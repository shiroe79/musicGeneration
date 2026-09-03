from __future__ import annotations

import base64
import datetime
import html
import json
import math
import shutil
import subprocess
import traceback
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import gradio as gr
import mido
import numpy as np

from aimusic.app.cli import _build_structural_diagnostics, _json_ready
from aimusic.core.config import (
    DecodeConfig,
    EDOConfig,
    MicrotonalRendering,
    StyleConfig,
)
from aimusic.core.diagnostics import RunManifest, SBDiagnostics
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import DEFAULT_GROOVE_FAMILIES, DEFAULT_METER_SIGNATURES
from aimusic.decode import decode_path_to_score
from aimusic.planning.plans import MethodARunConfig, run_method_a
from aimusic.render import render_midi
from aimusic.render.midi_render import TrackInstrumentConfig
from aimusic.theory.edo import EDO


OUTPUT_DIR = Path("./outputs")
SOUNDFONT_PATH = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
PREVIEW_SAMPLE_RATE = 22050
PREVIEW_TAIL_SECONDS = 0.35
MIDI_DRUM_CHANNEL = 9
PITCH_CLASS_NAMES_12 = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
DRUM_NOTE_NAMES = {
    35: "Acoustic kick",
    36: "Kick",
    38: "Snare",
    40: "Electric snare",
    42: "Closed hat",
    44: "Pedal hat",
    46: "Open hat",
}


@dataclass(frozen=True)
class GenerationParams:
    seed: int
    beats: int
    edo: int
    meter: str
    groove_family: str
    tempo_bpm: float
    sample_path: bool
    drum_density: float
    bass_density: float
    comping_density: float
    lead_density: float
    pitch_bend_range: int
    rendering_method: str
    bass_program: int
    comping_program: int
    lead_program: int
    drum_track: list[str]


@dataclass(frozen=True)
class GeneratedArtifacts:
    run_id: str
    score_path: Path
    midi_path: Path
    manifest_path: Path
    wav_path: Path


@dataclass(frozen=True)
class MidiPreviewNote:
    start_time: float
    end_time: float
    midi_note: int
    velocity: int
    channel: int
    pitch_bend: int = 0
    pitch_bend_range: float = 2.0


class MidiAudioConversionError(RuntimeError):
    """Raised when no usable MIDI-to-WAV converter is available."""


def _as_int(label: str, value: Any, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc

    if not numeric.is_integer():
        raise ValueError(f"{label} must be an integer.")

    integer = int(numeric)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    if maximum is not None and integer > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return integer


def _as_positive_float(label: str, value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc

    if numeric <= 0:
        raise ValueError(f"{label} must be > 0.")
    return numeric


def _as_unit_float(label: str, value: Any) -> float:
    numeric = _as_positive_or_zero_float(label, value)
    if numeric > 1:
        raise ValueError(f"{label} must be <= 1.")
    return numeric


def _as_positive_or_zero_float(label: str, value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc

    if numeric < 0:
        raise ValueError(f"{label} must be >= 0.")
    return numeric


def _normalize_inputs(
    seed: Any,
    beats: Any,
    edo: Any,
    meter: str,
    groove_family: str,
    tempo_bpm: Any,
    sample_path: bool,
    drum_density: Any,
    bass_density: Any,
    comping_density: Any,
    lead_density: Any,
    pitch_bend_range: Any,
    rendering_method: str,
    bass_program: Any,
    comping_program: Any,
    lead_program: Any,
    drum_track: list[str],
) -> GenerationParams:
    meter = str(meter).strip()
    groove_family = str(groove_family).strip()
    rendering_method = str(rendering_method).strip()

    if not meter:
        raise ValueError("meter must not be empty.")
    if not groove_family:
        raise ValueError("groove family must not be empty.")
    if not drum_track:
        raise ValueError("At least one track must be selected for drums.")
    supported_rendering_names = tuple(MicrotonalRendering.__members__)
    if rendering_method not in supported_rendering_names:
        raise ValueError(
            "rendering method must be one of "
            f"{', '.join(supported_rendering_names)}."
        )

    return GenerationParams(
        seed=_as_int("seed", seed),
        beats=_as_int("beats", beats, minimum=1),
        edo=_as_int("edo", edo, minimum=1),
        meter=meter,
        groove_family=groove_family,
        tempo_bpm=_as_positive_float("tempo bpm", tempo_bpm),
        sample_path=bool(sample_path),
        drum_density=_as_unit_float("drum density", drum_density),
        bass_density=_as_unit_float("bass density", bass_density),
        comping_density=_as_unit_float("comping density", comping_density),
        lead_density=_as_unit_float("lead density", lead_density),
        pitch_bend_range=_as_int("pitch bend range", pitch_bend_range, minimum=1),
        rendering_method=rendering_method,
        bass_program=_as_int("bass program", bass_program, minimum=0, maximum=127),
        comping_program=_as_int("comping program", comping_program, minimum=0, maximum=127),
        lead_program=_as_int("lead program", lead_program, minimum=0, maximum=127),
        drum_track=drum_track,
    )


def _drum_track_names(drum_track: list[str]) -> tuple[str, ...]:
    return tuple(name.strip().lower() for name in drum_track if name.strip())


def _build_track_instruments(params: GenerationParams) -> dict[str, TrackInstrumentConfig]:
    instruments = {
        "bass": TrackInstrumentConfig(program=params.bass_program),
        "comping": TrackInstrumentConfig(program=params.comping_program),
        "lead": TrackInstrumentConfig(program=params.lead_program),
    }
    for track_name in _drum_track_names(params.drum_track):
        existing = instruments.get(track_name)
        instruments[track_name] = TrackInstrumentConfig(
            program=None if existing is None else existing.program,
            is_drum=True,
        )
    return instruments


def _generate_artifacts(params: GenerationParams) -> GeneratedArtifacts:
    style_config = StyleConfig(
        allowed_meters=(params.meter,),
        groove_families=(params.groove_family,),
    )
    decode_config = DecodeConfig(
        drum_density=params.drum_density,
        bass_density=params.bass_density,
        comping_density=params.comping_density,
        lead_density=params.lead_density,
    )
    run_config = MethodARunConfig(
        total_beats=params.beats,
        seed=params.seed,
        use_sampling=params.sample_path,
        style_config=style_config,
        decode_config=decode_config,
        edo=params.edo,
    )

    plan_result, next_key = run_method_a(run_config, key=RNGKey(seed=params.seed))
    score, _ = decode_path_to_score(
        plan_result.path,
        decode_config=decode_config,
        vocabularies=plan_result.vocabularies,
        edo=params.edo,
        tempo_bpm=params.tempo_bpm,
        key=next_key,
    )
    structural_stats = _build_structural_diagnostics(
        plan_result.path,
        plan_result.vocabularies,
        edo=params.edo,
        sections=plan_result.endpoints.sections,
    )
    track_instruments = _build_track_instruments(params)
    manifest = RunManifest(
        seed=params.seed,
        config_dump=_json_ready(
            {
                "run_config": run_config,
                "meter": params.meter,
                "groove_family": params.groove_family,
                "tempo_bpm": params.tempo_bpm,
                "output_dir": str(OUTPUT_DIR),
                "pitch_bend_range": params.pitch_bend_range,
                "rendering_method": params.rendering_method,
                "track_instruments": track_instruments,
            }
        ),
        structural_stats=structural_stats,
        sb_stats=SBDiagnostics.from_solution(plan_result.sb_solution),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_path = OUTPUT_DIR / f"{manifest.run_id}_score.json"
    midi_path = OUTPUT_DIR / f"{manifest.run_id}.mid"
    manifest_path = OUTPUT_DIR / f"{manifest.run_id}_manifest.json"
    wav_path = OUTPUT_DIR / f"{manifest.run_id}.wav"

    with score_path.open("w", encoding="utf-8") as f:
        json.dump(score.to_dict(), f, indent=2)

    render_midi(
        score,
        EDO(
            EDOConfig(
                n=params.edo,
                base_tuning=0,
                pitch_bend_range=params.pitch_bend_range,
                microtonal_rendering_method=MicrotonalRendering[params.rendering_method],
            )
        ),
        str(midi_path),
        track_instruments=track_instruments,
    )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return GeneratedArtifacts(
        run_id=manifest.run_id,
        score_path=score_path,
        midi_path=midi_path,
        manifest_path=manifest_path,
        wav_path=wav_path,
    )


def _run_converter(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def _first_midi_tempo(midi_file: mido.MidiFile) -> int:
    for track in midi_file.tracks:
        for message in track:
            if message.type == "set_tempo":
                return int(message.tempo)
    return int(mido.bpm2tempo(120))


def _extract_midi_preview_notes(midi_path: Path) -> list[MidiPreviewNote]:
    midi_file = mido.MidiFile(midi_path)
    tempo = _first_midi_tempo(midi_file)
    seconds_per_tick = tempo / 1_000_000 / midi_file.ticks_per_beat
    preview_notes: list[MidiPreviewNote] = []

    for track_index, track in enumerate(midi_file.tracks):
        absolute_tick = 0
        active_notes: dict[
            tuple[int, int, int],
            list[tuple[int, int, int, float]],
        ] = {}
        pitch_bends: dict[int, int] = {}
        pitch_bend_ranges: dict[int, float] = {}
        rpn_selection: dict[int, tuple[int, int]] = {}
        rpn_msb: dict[int, int] = {}
        rpn_lsb: dict[int, int] = {}
        for message in track:
            absolute_tick += int(message.time)
            if hasattr(message, "channel") and message.type == "pitchwheel":
                pitch_bends[int(message.channel)] = int(message.pitch)
                continue
            if hasattr(message, "channel") and message.type == "control_change":
                channel = int(message.channel)
                if message.control == 101:
                    rpn_msb[channel] = int(message.value)
                elif message.control == 100:
                    rpn_lsb[channel] = int(message.value)
                rpn_selection[channel] = (
                    rpn_msb.get(channel, 127),
                    rpn_lsb.get(channel, 127),
                )
                if message.control == 6 and rpn_selection[channel] == (0, 0):
                    pitch_bend_ranges[channel] = float(message.value)
                continue
            if not hasattr(message, "channel") or not hasattr(message, "note"):
                continue

            channel = int(message.channel)
            key = (track_index, channel, int(message.note))
            if message.type == "note_on" and message.velocity > 0:
                active_notes.setdefault(key, []).append(
                    (
                        absolute_tick,
                        int(message.velocity),
                        pitch_bends.get(channel, 0),
                        pitch_bend_ranges.get(channel, 2.0),
                    )
                )
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                starts = active_notes.get(key)
                if not starts:
                    continue
                start_tick, velocity, pitch_bend, pitch_bend_range = starts.pop(0)
                if absolute_tick <= start_tick:
                    continue
                preview_notes.append(
                    MidiPreviewNote(
                        start_time=start_tick * seconds_per_tick,
                        end_time=absolute_tick * seconds_per_tick,
                        midi_note=int(message.note),
                        velocity=velocity,
                        channel=channel,
                        pitch_bend=pitch_bend,
                        pitch_bend_range=pitch_bend_range,
                    )
                )

    return preview_notes


def _midi_note_frequency(
    midi_note: int,
    pitch_bend: int = 0,
    pitch_bend_range: float = 2.0,
) -> float:
    bend_scale = 8191 if pitch_bend >= 0 else 8192
    sounding_pitch = midi_note + (pitch_bend / bend_scale) * pitch_bend_range
    return 440.0 * (2.0 ** ((sounding_pitch - 69) / 12.0))


def _note_envelope(sample_count: int, sample_rate: int) -> np.ndarray:
    envelope = np.ones(sample_count, dtype=np.float32)
    if sample_count <= 2:
        return envelope

    attack = min(max(1, int(sample_rate * 0.005)), sample_count // 3)
    release = min(max(1, int(sample_rate * 0.040)), sample_count // 3)
    envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    envelope[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32)
    return envelope


def _render_drum_preview(note: MidiPreviewNote, sample_rate: int) -> np.ndarray:
    duration = max(0.04, min(note.end_time - note.start_time, 0.45))
    sample_count = max(1, int(duration * sample_rate))
    t = np.arange(sample_count, dtype=np.float32) / sample_rate
    amplitude = 0.22 * (note.velocity / 127.0)

    if note.midi_note in (35, 36):
        wave_data = np.sin(2.0 * math.pi * 58.0 * t) * np.exp(-8.0 * t)
    elif note.midi_note in (38, 40):
        rng = np.random.default_rng(note.midi_note * 1009 + sample_count)
        noise = rng.uniform(-1.0, 1.0, sample_count).astype(np.float32)
        tone = np.sin(2.0 * math.pi * 180.0 * t)
        wave_data = ((0.65 * noise) + (0.35 * tone)) * np.exp(-14.0 * t)
    elif note.midi_note in (42, 44, 46):
        wave_data = (
            np.sin(2.0 * math.pi * 2300.0 * t)
            * np.sin(2.0 * math.pi * 3100.0 * t)
            * np.exp(-32.0 * t)
        )
    else:
        wave_data = np.sin(2.0 * math.pi * 440.0 * t) * np.exp(-16.0 * t)

    return (amplitude * wave_data).astype(np.float32)


def _render_melodic_preview(note: MidiPreviewNote, sample_rate: int) -> np.ndarray:
    duration = max(0.01, note.end_time - note.start_time)
    sample_count = max(1, int(duration * sample_rate))
    t = np.arange(sample_count, dtype=np.float32) / sample_rate
    frequency = _midi_note_frequency(
        note.midi_note,
        note.pitch_bend,
        note.pitch_bend_range,
    )
    amplitude = 0.13 * (note.velocity / 127.0)
    wave_data = (
        np.sin(2.0 * math.pi * frequency * t)
        + 0.25 * np.sin(2.0 * math.pi * frequency * 2.0 * t)
    )
    return (amplitude * wave_data * _note_envelope(sample_count, sample_rate)).astype(np.float32)


def _render_midi_preview_wav(
    midi_path: Path,
    wav_path: Path,
    *,
    sample_rate: int = PREVIEW_SAMPLE_RATE,
) -> None:
    preview_notes = _extract_midi_preview_notes(midi_path)
    if not preview_notes:
        raise MidiAudioConversionError("MIDI preview failed because the file contained no notes.")

    duration = max(note.end_time for note in preview_notes) + PREVIEW_TAIL_SECONDS
    samples = np.zeros(max(1, int(duration * sample_rate)), dtype=np.float32)

    for note in preview_notes:
        rendered = (
            _render_drum_preview(note, sample_rate)
            if note.channel == MIDI_DRUM_CHANNEL
            else _render_melodic_preview(note, sample_rate)
        )
        start_index = max(0, int(note.start_time * sample_rate))
        end_index = min(samples.size, start_index + rendered.size)
        if end_index <= start_index:
            continue
        samples[start_index:end_index] += rendered[: end_index - start_index]

    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = samples * min(0.95 / peak, 1.0)

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (samples * 32767.0).clip(-32768, 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _convert_midi_to_wav(midi_path: Path, wav_path: Path) -> str:
    midi_file = mido.MidiFile(midi_path)
    has_mts_tuning = any(
        message.type == "sysex"
        and tuple(message.data[:5]) == (0x7E, 0x7F, 0x08, 0x01, 0x00)
        for track in midi_file.tracks
        for message in track
    )
    if has_mts_tuning:
        raise MidiAudioConversionError(
            "The MIDI file uses MIDI Tuning Standard (MTS). Audio preview is "
            "disabled because the available preview converters cannot guarantee "
            "MTS tuning reproduction. Download the MIDI and play it with an "
            "MTS-compatible synthesizer."
        )

    conversion_errors: list[str] = []
    fluidsynth = shutil.which("fluidsynth")

    if fluidsynth is not None and SOUNDFONT_PATH.exists():
        try:
            _run_converter(
                [
                    fluidsynth,
                    "-ni",
                    str(SOUNDFONT_PATH),
                    str(midi_path),
                    "-F",
                    str(wav_path),
                    "-r",
                    "44100",
                ]
            )
            return "fluidsynth"
        except subprocess.CalledProcessError as exc:
            conversion_errors.append(f"fluidsynth failed: {exc.stderr or exc.stdout or exc}")
    elif fluidsynth is not None:
        conversion_errors.append(
            f"fluidsynth was found, but the soundfont is missing at {SOUNDFONT_PATH}."
        )

    timidity = shutil.which("timidity")
    if timidity is not None:
        try:
            _run_converter([timidity, str(midi_path), "-Ow", "-o", str(wav_path)])
            return "timidity"
        except subprocess.CalledProcessError as exc:
            conversion_errors.append(f"timidity failed: {exc.stderr or exc.stdout or exc}")

    try:
        _render_midi_preview_wav(midi_path, wav_path)
        return "built-in preview synth (MPE pitch bends applied)"
    except Exception as exc:
        conversion_errors.append(f"built-in preview synth failed: {exc}")

    details = "\n".join(conversion_errors)
    if details:
        details = f"\n\nConverter details:\n{details}"
    raise MidiAudioConversionError(
        "Could not create an audio preview for the generated MIDI. Install fluidsynth with "
        f"a General MIDI soundfont available at {SOUNDFONT_PATH}, or install timidity."
        "\n\nLinux example: sudo apt install fluidsynth fluid-soundfont-gm timidity"
        "\nmacOS example: brew install fluid-synth timidity"
        "\nWindows example: install FluidSynth and add fluidsynth.exe to PATH, or install TiMidity++."
        f"{details}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _note_label(note: dict[str, Any], edo: int) -> str:
    track = str(note.get("track", "")).lower()
    pitch_height = int(note.get("h", 0))
    if track == "drums":
        return DRUM_NOTE_NAMES.get(pitch_height, f"Drum {pitch_height}")
    if edo == 12:
        return f"{PITCH_CLASS_NAMES_12[pitch_height % 12]}{(pitch_height // 12) - 1}"
    return f"pc_{pitch_height % edo}"


def _dashboard_note_events(score_data: dict[str, Any], edo: int) -> list[dict[str, Any]]:
    ticks_per_beat = int(score_data.get("ticks_per_beat", 480))
    tempo_bpm = float(score_data.get("tempo_bpm", 120.0))
    seconds_per_tick = 60.0 / tempo_bpm / ticks_per_beat

    events = []
    for note in score_data.get("note_events", []):
        events.append(
            {
                "start": round(int(note.get("ton", 0)) * seconds_per_tick, 4),
                "end": round(int(note.get("toff", 0)) * seconds_per_tick, 4),
                "track": str(note.get("track", "default")),
                "label": _note_label(note, edo),
            }
        )
    return events


def _dashboard_chord_events(
    manifest_data: dict[str, Any],
    tempo_bpm: float,
) -> list[dict[str, Any]]:
    seconds_per_beat = 60.0 / tempo_bpm
    events = []
    for event in manifest_data.get("structure", {}).get("chord_timeline", []):
        events.append(
            {
                "start": round(float(event.get("start_time", 0.0)) * seconds_per_beat, 4),
                "end": round(float(event.get("end_time", 0.0)) * seconds_per_beat, 4),
                "label": str(event.get("label", "unknown")),
            }
        )
    return events


def _dashboard_tension_points(
    manifest_data: dict[str, Any],
    tempo_bpm: float,
) -> list[dict[str, float]]:
    """Realized tension curve, converted from beat-index time to seconds.

    Target-vs-realized is intentionally not wired in yet (planned as a later
    toggle) — this only surfaces `structure.tension_curve`.
    """
    seconds_per_beat = 60.0 / tempo_bpm
    points = []
    for beat_time, value in manifest_data.get("structure", {}).get("tension_curve", []):
        points.append(
            {
                "t": round(float(beat_time) * seconds_per_beat, 4),
                "v": round(float(value), 4),
            }
        )
    return points


def _build_playback_dashboard(
    wav_path: Path,
    score_path: Path,
    manifest_path: Path,
) -> str:
    score_data = _load_json(score_path)
    manifest_data = _load_json(manifest_path)
    tempo_bpm = float(score_data.get("tempo_bpm", 120.0))
    edo = int(manifest_data.get("config", {}).get("run_config", {}).get("edo", 12))
    notes = _dashboard_note_events(score_data, edo)
    chords = _dashboard_chord_events(manifest_data, tempo_bpm)
    tension = _dashboard_tension_points(manifest_data, tempo_bpm)
    audio_base64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")

    payload = json.dumps(
        {
            "audio": f"data:audio/wav;base64,{audio_base64}",
            "notes": notes,
            "chords": chords,
            "tension": tension,
            "secondsPerBeat": round(60.0 / tempo_bpm, 6),
        }
    )
    iframe_document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #f8fafc;
      background: #0f1115;
    }}
    .panel {{
      border: 1px solid #2b3340;
      border-radius: 8px;
      padding: 12px;
      background: #141820;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }}
    audio {{
      display: none;
    }}
    .controls {{
      outline: none;
    }}
    .transport {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 18px;
      margin-bottom: 8px;
    }}
    .ctrl-btn {{
      background: transparent;
      border: none;
      outline: none;
      -webkit-tap-highlight-color: transparent;
      color: #e5e9f0;
      cursor: pointer;
      font-size: 16px;
      width: 30px;
      height: 30px;
      border-radius: 999px;
      display: flex;
      align-items: center;
      justify-content: center;
      line-height: 1;
      padding: 0;
    }}
    .ctrl-btn:focus,
    .ctrl-btn:active {{
      outline: none;
      background: transparent;
    }}
    .ctrl-btn:hover {{
      background: #232a36;
    }}
    .ctrl-btn-main {{
      font-size: 15px;
    }}
    .seek-row {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .time-label {{
      color: #9aa4b2;
      font-size: 11px;
      font-variant-numeric: tabular-nums;
      min-width: 30px;
      text-align: center;
    }}
    .seek-track {{
      position: relative;
      flex: 1;
      height: 4px;
      border-radius: 99px;
      background: #2a303a;
      cursor: pointer;
    }}
    .seek-fill {{
      position: absolute;
      top: 0;
      left: 0;
      height: 100%;
      width: 0%;
      border-radius: 99px;
      background: #f35620;
      pointer-events: none;
    }}
    .seek-handle {{
      position: absolute;
      top: 50%;
      left: 0%;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #f35620;
      transform: translate(-50%, -50%);
      pointer-events: none;
    }}
    .tension-wrap {{
      margin-top: 26px;
      padding-top: 14px;
      border-top: 1px solid #232a36;
    }}
    .tension-label {{
      color: #9aa4b2;
      font-size: 11px;
      text-transform: uppercase;
      margin-bottom: 4px;
    }}
    .tension-svg {{
      width: 100%;
      height: 56px;
      display: block;
      border-radius: 6px;
      background: #10141b;
      cursor: pointer;
    }}
    .tension-grid-line {{
      stroke: #232a36;
      stroke-width: 1;
    }}
    .tension-line {{
      fill: none;
      stroke: #f35620;
      stroke-width: 2;
    }}
    .tension-playhead {{
      stroke: #f8fafc;
      stroke-width: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr 2fr;
      gap: 8px;
      margin-top: 10px;
    }}
    .cell {{
      min-height: 54px;
      border: 1px solid #2b3340;
      border-radius: 6px;
      padding: 8px;
      background: #1b212c;
    }}
    .label {{
      color: #9aa4b2;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
      margin-bottom: 4px;
    }}
    .value {{
      font-size: 18px;
      line-height: 1.25;
      font-weight: 650;
    }}
    .notes {{
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }}
    .chip {{
      border-radius: 6px;
      padding: 3px 6px;
      background: #243b63;
      color: #d7e7ff;
      font-size: 12px;
      line-height: 1.2;
      white-space: nowrap;
    }}
    .track {{
      color: #9fb5d4;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <audio id="player" src=""></audio>

    <div class="controls" id="controls" tabindex="0">
      <div class="transport">
        <button id="back" class="ctrl-btn" title="Previous beat (Left arrow)" aria-label="Previous beat">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M6 6h2v12H6zM20 18V6l-10 6z"/></svg>
        </button>
        <button id="playpause" class="ctrl-btn ctrl-btn-main" title="Play / Pause" aria-label="Play or pause">
          <svg id="playIcon" viewBox="0 0 24 24" width="15" height="15" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
        </button>
        <button id="fwd" class="ctrl-btn" title="Next beat (Right arrow)" aria-label="Next beat">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M18 6h-2v12h2zM4 6v12l10-6z"/></svg>
        </button>
      </div>
      <div class="seek-row">
        <span id="curTime" class="time-label">0:00</span>
        <div id="seekTrack" class="seek-track">
          <div id="seekFill" class="seek-fill"></div>
          <div id="seekHandle" class="seek-handle"></div>
        </div>
        <span id="durTime" class="time-label">0:00</span>
      </div>
    </div>

    <div class="tension-wrap" id="tensionWrap">
      <div class="tension-label">Tension</div>
      <svg id="tensionSvg" class="tension-svg" viewBox="0 0 1000 56" preserveAspectRatio="none"></svg>
    </div>

    <div class="grid">
      <div class="cell">
        <div class="label">Time</div>
        <div id="time" class="value">0.00s</div>
      </div>
      <div class="cell">
        <div class="label">Chord</div>
        <div id="chord" class="value">-</div>
      </div>
      <div class="cell">
        <div class="label">Notes</div>
        <div id="notes" class="notes"><span class="chip">-</span></div>
      </div>
    </div>
  </div>
  <script>
    const data = {payload};
    const player = document.getElementById("player");
    const controls = document.getElementById("controls");
    const playPauseBtn = document.getElementById("playpause");
    const backBtn = document.getElementById("back");
    const fwdBtn = document.getElementById("fwd");
    const curTimeEl = document.getElementById("curTime");
    const durTimeEl = document.getElementById("durTime");
    const seekTrack = document.getElementById("seekTrack");
    const seekFill = document.getElementById("seekFill");
    const seekHandle = document.getElementById("seekHandle");
    const timeEl = document.getElementById("time");
    const chordEl = document.getElementById("chord");
    const notesEl = document.getElementById("notes");
    const tensionWrap = document.getElementById("tensionWrap");
    const tensionSvg = document.getElementById("tensionSvg");
    player.src = data.audio;

    const NS = "http://www.w3.org/2000/svg";
    let beatSeconds = [];
    let playheadLine = null;

    function formatTime(seconds) {{
      const s = Math.max(0, seconds || 0);
      const mins = Math.floor(s / 60);
      const secs = Math.floor(s % 60);
      return mins + ":" + String(secs).padStart(2, "0");
    }}

    function currentChord(t) {{
      return data.chords.find((item) => item.start <= t && t < item.end);
    }}

    function currentNotes(t) {{
      return data.notes.filter((item) => item.start <= t && t < item.end).slice(0, 14);
    }}

    function nearestBeatIndex(t) {{
      let idx = 0;
      for (let i = 0; i < beatSeconds.length; i++) {{
        if (beatSeconds[i] <= t + 0.02) idx = i;
      }}
      return idx;
    }}

    function skip(direction) {{
      if (!beatSeconds.length) return;
      const t = player.currentTime || 0;
      const idx = nearestBeatIndex(t);
      const targetIdx = Math.min(beatSeconds.length - 1, Math.max(0, idx + direction));
      player.currentTime = beatSeconds[targetIdx];
      render();
    }}

    function seekToClientX(clientX) {{
      const duration = player.duration || 0;
      if (!duration) return;
      const rect = seekTrack.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      player.currentTime = frac * duration;
      render();
    }}

    function seekTensionToClientX(clientX) {{
      const duration = player.duration || 0;
      if (!duration) return;
      const rect = tensionSvg.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      player.currentTime = frac * duration;
      render();
    }}

    function buildTensionChart() {{
      const duration = player.duration || 0;
      tensionSvg.innerHTML = "";
      if (!duration || !data.tension.length) {{
        tensionWrap.style.display = "none";
        return;
      }}
      tensionWrap.style.display = "block";

      beatSeconds.forEach((t) => {{
        const x = (t / duration) * 1000;
        const gridLine = document.createElementNS(NS, "line");
        gridLine.setAttribute("x1", x);
        gridLine.setAttribute("x2", x);
        gridLine.setAttribute("y1", 0);
        gridLine.setAttribute("y2", 56);
        gridLine.setAttribute("class", "tension-grid-line");
        tensionSvg.appendChild(gridLine);
      }});

      const points = data.tension
        .map((p) => {{
          const x = (p.t / duration) * 1000;
          const y = 54 - Math.min(1, Math.max(0, p.v)) * 50;
          return x.toFixed(2) + "," + y.toFixed(2);
        }})
        .join(" ");
      const polyline = document.createElementNS(NS, "polyline");
      polyline.setAttribute("points", points);
      polyline.setAttribute("class", "tension-line");
      tensionSvg.appendChild(polyline);

      playheadLine = document.createElementNS(NS, "line");
      playheadLine.setAttribute("x1", 0);
      playheadLine.setAttribute("x2", 0);
      playheadLine.setAttribute("y1", 0);
      playheadLine.setAttribute("y2", 56);
      playheadLine.setAttribute("class", "tension-playhead");
      tensionSvg.appendChild(playheadLine);
    }}

    function updatePosition() {{
      const t = player.currentTime || 0;
      const duration = player.duration || 0;
      const frac = duration ? Math.min(1, t / duration) : 0;
      curTimeEl.textContent = formatTime(t);
      seekFill.style.width = (frac * 100) + "%";
      seekHandle.style.left = (frac * 100) + "%";
      if (playheadLine) {{
        playheadLine.setAttribute("x1", frac * 1000);
        playheadLine.setAttribute("x2", frac * 1000);
      }}
    }}

    function render() {{
      const t = player.currentTime || 0;
      const duration = player.duration || 0;
      const chord = currentChord(t);
      const notes = currentNotes(t);
      timeEl.textContent = t.toFixed(2) + "s";
      chordEl.textContent = chord ? chord.label : "-";
      notesEl.innerHTML = notes.length
        ? notes.map((note) => `<span class="chip">${{note.label}} <span class="track">${{note.track}}</span></span>`).join("")
        : `<span class="chip">-</span>`;
      durTimeEl.textContent = formatTime(duration);
      updatePosition();
    }}

    let rafId = null;
    function positionLoop() {{
      updatePosition();
      if (!player.paused && !player.ended) {{
        rafId = window.requestAnimationFrame(positionLoop);
      }}
    }}
    function startPositionLoop() {{
      if (rafId !== null) return;
      rafId = window.requestAnimationFrame(positionLoop);
    }}
    function stopPositionLoop() {{
      if (rafId !== null) {{
        window.cancelAnimationFrame(rafId);
        rafId = null;
      }}
    }}

    const PLAY_ICON = '<svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
    const PAUSE_ICON = '<svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg>';

    player.addEventListener("loadedmetadata", () => {{
      const duration = player.duration || 0;
      const spb = data.secondsPerBeat || 1;
      beatSeconds = [];
      for (let t = 0; t <= duration + 0.001; t += spb) {{
        beatSeconds.push(Math.min(t, duration));
      }}
      if (!beatSeconds.length) beatSeconds = [0];
      buildTensionChart();
      render();
    }});
    player.addEventListener("timeupdate", render);
    player.addEventListener("seeked", render);
    player.addEventListener("play", () => {{
      playPauseBtn.innerHTML = PAUSE_ICON;
      startPositionLoop();
    }});
    player.addEventListener("pause", () => {{
      playPauseBtn.innerHTML = PLAY_ICON;
      stopPositionLoop();
      render();
    }});
    player.addEventListener("ended", () => {{
      playPauseBtn.innerHTML = PLAY_ICON;
      stopPositionLoop();
      render();
    }});

    playPauseBtn.addEventListener("click", () => {{
      if (player.paused) {{
        player.play();
      }} else {{
        player.pause();
      }}
    }});
    backBtn.addEventListener("click", () => skip(-1));
    fwdBtn.addEventListener("click", () => skip(1));
    seekTrack.addEventListener("click", (e) => seekToClientX(e.clientX));
    tensionSvg.addEventListener("click", (e) => seekTensionToClientX(e.clientX));

    controls.addEventListener("click", () => controls.focus());
    controls.addEventListener("keydown", (e) => {{
      if (e.key === "ArrowLeft") {{
        e.preventDefault();
        skip(-1);
      }} else if (e.key === "ArrowRight") {{
        e.preventDefault();
        skip(1);
      }} else if (e.key === " ") {{
        e.preventDefault();
        if (player.paused) player.play(); else player.pause();
      }}
    }});

    render();
  </script>
</body>
</html>"""
    return (
        '<iframe title="MIDI playback dashboard" '
        'style="width:100%;height:270px;border:0;border-radius:8px;" '
        f'srcdoc="{html.escape(iframe_document, quote=True)}"></iframe>'
    )


def _score_summary(score_path: Path, manifest_path: Path) -> str:
    score_data = _load_json(score_path)
    manifest_data = _load_json(manifest_path)

    notes = score_data.get("note_events", [])
    ticks_per_beat = int(score_data.get("ticks_per_beat", 480))
    tempo_bpm = float(score_data.get("tempo_bpm", 120.0))
    track_counts = score_data.get("track_event_counts", {})
    max_tick = max((int(note.get("toff", 0)) for note in notes), default=0)
    duration_beats = max_tick / ticks_per_beat
    duration_seconds = duration_beats * 60.0 / tempo_bpm
    meter = manifest_data.get("config", {}).get("meter", "unknown")

    return (
        "### Analysis\n"
        f"- Tempo: {tempo_bpm:g} BPM\n"
        f"- Meter: {meter}\n"
        f"- Track count: {len(track_counts)}\n"
        f"- Duration: {duration_seconds:.2f}s ({duration_beats:.2f} beats)"
    )


def _score_table(score_path: Path, manifest_path: Path) -> str:
    score_data = _load_json(score_path)
    track_counts = score_data.get("track_event_counts", {})
    track_lines = "\n".join(
        f"| {track} | {count} |" for track, count in sorted(track_counts.items())
    )
    if not track_lines:
        track_lines = "| none | 0 |"
    return (
        "| Track | Notes |\n"
        "| --- | ---: |\n"
        f"{track_lines}"
    )


def _error_markdown(message: str, *, include_traceback: bool = False) -> str:
    if include_traceback:
        return f"### Error\n```text\n{message}\n```"
    return f"### Error\n{message}"


MAX_HISTORY_ENTRIES = 25


class HistoryEntry(TypedDict):
    run_id: str
    label: str
    dashboard: str
    summary: str
    table: str
    midi_path: str
    score_path: str
    manifest_path: str


def _history_label(params: GenerationParams, run_id: str) -> str:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    return f"seed={params.seed} · {params.beats} beats · {stamp} · {run_id[:8]}"


def _history_entry(
    params: GenerationParams,
    artifacts: GeneratedArtifacts,
    dashboard: str,
    summary: str,
    table: str,
) -> HistoryEntry:
    return HistoryEntry(
        run_id=artifacts.run_id,
        label=_history_label(params, artifacts.run_id),
        dashboard=dashboard,
        summary=summary,
        table=table,
        midi_path=str(artifacts.midi_path),
        score_path=str(artifacts.score_path),
        manifest_path=str(artifacts.manifest_path),
    )


def generate_music(
    seed: Any,
    beats: Any,
    edo: Any,
    meter: str,
    groove_family: str,
    tempo_bpm: Any,
    sample_path: bool,
    drum_density: Any,
    bass_density: Any,
    comping_density: Any,
    lead_density: Any,
    pitch_bend_range: Any,
    rendering_method: str,
    bass_program: Any,
    comping_program: Any,
    lead_program: Any,
    drum_track: list[str],
    history: list[HistoryEntry],
) -> tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    str,
    list[HistoryEntry],
]:
    history = list(history or [])
    try:
        params = _normalize_inputs(
            seed,
            beats,
            edo,
            meter,
            groove_family,
            tempo_bpm,
            sample_path,
            drum_density,
            bass_density,
            comping_density,
            lead_density,
            pitch_bend_range,
            rendering_method,
            bass_program,
            comping_program,
            lead_program,
            drum_track,
        )
        artifacts = _generate_artifacts(params)
        summary = _score_summary(artifacts.score_path, artifacts.manifest_path)
        table = _score_table(artifacts.score_path, artifacts.manifest_path)
    except Exception:
        return (
            "",
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            _error_markdown(traceback.format_exc(), include_traceback=True),
            history,
        )

    try:
        _ = _convert_midi_to_wav(artifacts.midi_path, artifacts.wav_path)
        dashboard = _build_playback_dashboard(
            artifacts.wav_path,
            artifacts.score_path,
            artifacts.manifest_path,
        )
    except MidiAudioConversionError as exc:
        return (
            "",
            gr.update(value=summary, visible=True),
            gr.update(value=table, visible=True),
            _error_markdown(str(exc)),
            history,
        )

    entry = _history_entry(params, artifacts, dashboard, summary, table)
    history = ([entry] + history)[:MAX_HISTORY_ENTRIES]

    return (
        dashboard,
        gr.update(value=summary, visible=True),
        gr.update(value=table, visible=True),
        "",
        history,
    )


def select_history_entry(
    entry: HistoryEntry,
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
    return (
        entry["dashboard"],
        gr.update(value=entry["summary"], visible=True),
        gr.update(value=entry["table"], visible=True),
        "",
    )


def _history_file_download(label: str, path: str) -> gr.components.Component:
    if hasattr(gr, "DownloadButton"):
        return gr.DownloadButton(
            label=label,
            value=path,
            size="sm",
            elem_classes="history-file-btn",
        )
    return gr.File(value=path, label=label, interactive=False)


css = """
    .block { padding: 0 8px !important; }
    .form { gap: 2px !important; }
    .wrap { gap: 2px !important; padding: 0 !important; }
    label { margin-bottom: 0 !important; font-size: 11px !important; }
    .density-box { border: 1px solid #374151 !important; border-radius: 6px !important; padding: 2px 6px !important; }
    .density-box input[type=range] { height: 3px !important; }
    input[type=number] { padding: 1px 4px !important; }
    .drum-check .wrap { display: flex !important; flex-direction: row !important; flex-wrap: wrap !important; gap: 12px !important; }
    .history-row { gap: 4px !important; margin-bottom: 2px !important; }
    .history-select { text-align: left !important; justify-content: flex-start !important; font-size: 11px !important; }
    .history-icon-btn { min-width: 30px !important; max-width: 34px !important; padding: 0 !important; font-size: 13px !important; }
    .history-downloads { gap: 4px !important; margin: 0 0 8px 0 !important; padding-left: 4px !important; }
    .history-file-btn { font-size: 10px !important; padding: 2px 6px !important; min-width: 0 !important; }
"""

with gr.Blocks(title="MIDI Generator", fill_height=True) as demo:
    gr.Markdown("## MIDI Generator <small style='font-weight:700;color:#9aa4b2;font-size:14px'>  | write configs and click 'generate'</small>")
    with gr.Row(equal_height=False):
        with gr.Column(scale=0):
            seed = gr.Number(label="seed", value=11, precision=0)
            beats = gr.Number(label="beats", value=8, precision=0)
            edo = gr.Number(label="edo", value=12, precision=0)
            meter = gr.Dropdown(
                label="meter",
                choices=list(DEFAULT_METER_SIGNATURES),
                value="4/4",
                allow_custom_value=True,
            )
            groove_family = gr.Dropdown(
                label="groove-family",
                choices=list(DEFAULT_GROOVE_FAMILIES),
                value="straight",
            )
            tempo_bpm = gr.Number(label="tempo-bpm", value=120)
            pitch_bend_range = gr.Number(label="pitch-bend-range", value=2, precision=0)
            rendering_method = gr.Dropdown(
                label="rendering-method",
                choices=[method.name for method in MicrotonalRendering],
                value=MicrotonalRendering.MPE.name,
            )

            gr.Markdown("### History")
            history_state = gr.State([])
            history_list = gr.Column()

        with gr.Column(scale=2):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    bass_program = gr.Number(label="track-program bass", value=34, precision=0)
                    comping_program = gr.Number(label="track-program comping", value=5, precision=0)
                    lead_program = gr.Number(label="track-program lead", value=88, precision=0)
                    drum_track = gr.CheckboxGroup(
                        label="drum-track",
                        choices=["drums", "bass", "comping", "lead"],
                        value=["drums"],
                        elem_classes="drum-check",
                    )
                with gr.Column(scale=2):
                    drum_density = gr.Slider(
                        label="drum-density",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.75,
                        elem_classes="density-box",
                    )
                    bass_density = gr.Slider(
                        label="bass-density",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.60,
                        elem_classes="density-box",
                    )
                    comping_density = gr.Slider(
                        label="comping-density",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.55,
                        elem_classes="density-box",
                    )
                    lead_density = gr.Slider(
                        label="lead-density",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.45,
                        elem_classes="density-box",
                    )
                    sample_path = gr.Checkbox(label="Choose Sample Path", value=False)
            generate_button = gr.Button("Generate", variant="primary")
            dashboard = gr.HTML()
            status = gr.Markdown()
            with gr.Row(equal_height=True):
                summary = gr.Markdown(visible=False, scale=1)
                table = gr.Markdown(visible=False, scale=1)

    with history_list:

        @gr.render(inputs=history_state)
        def render_history(history: list[HistoryEntry]) -> None:
            if not history:
                gr.Markdown("<small style='color:#9aa4b2'>No runs yet.</small>")
                return
            for entry in history:
                downloads_open = gr.State(False)
                with gr.Row(elem_classes="history-row", equal_height=True):
                    select_btn = gr.Button(
                        entry["label"],
                        elem_classes="history-select",
                        size="sm",
                        scale=4,
                    )
                    downloads_toggle = gr.Button(
                        "\u2b07",
                        elem_classes="history-icon-btn",
                        size="sm",
                        scale=0,
                    )
                with gr.Row(elem_classes="history-downloads", visible=False) as downloads_row:
                    _history_file_download("MIDI", entry["midi_path"])
                    _history_file_download("Score JSON", entry["score_path"])
                    _history_file_download("Manifest JSON", entry["manifest_path"])

                def _toggle(is_open: bool) -> tuple[dict[str, Any], bool]:
                    new_state = not is_open
                    return gr.update(visible=new_state), new_state

                downloads_toggle.click(  # type: ignore[attr-defined]
                    fn=_toggle,
                    inputs=[downloads_open],
                    outputs=[downloads_row, downloads_open],
                    show_progress="hidden",
                )

                select_btn.click(  # type: ignore[attr-defined]
                    fn=lambda selected=entry: select_history_entry(selected),
                    inputs=None,
                    outputs=[
                        dashboard,
                        summary,
                        table,
                        status,
                    ],
                    show_progress="hidden",
                )

    generate_button.click(  # type: ignore[attr-defined]
        fn=generate_music,
        inputs=[
            seed,
            beats,
            edo,
            meter,
            groove_family,
            tempo_bpm,
            sample_path,
            drum_density,
            bass_density,
            comping_density,
            lead_density,
            pitch_bend_range,
            rendering_method,
            bass_program,
            comping_program,
            lead_program,
            drum_track,
            history_state,
        ],
        outputs=[
            dashboard,
            summary,
            table,
            status,
            history_state,
        ],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="localhost", server_port=7860, inbrowser=True, css=css)