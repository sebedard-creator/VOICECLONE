from __future__ import annotations

import shutil
import subprocess
import sys
import os
import warnings
from datetime import datetime
from pathlib import Path

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)
    from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from .audio_utils import convert_file_to_wav_24bit, export_segment_wav_24bit, normalize_peak
from .cache_env import build_subprocess_env
from .config import Settings, ensure_runtime_layout
from .errors import VoiceCloneError
from .ffmpeg_utils import ensure_ffmpeg_available
from .naming import safe_name


def clean_dataset(
    settings: Settings,
    actor_name: str,
    input_files: list[str],
    split_only: bool = False,
) -> Path:
    actor_slug = safe_name(actor_name)
    if not input_files:
        raise VoiceCloneError("Ajoute au moins un extrait rough a nettoyer.")

    paths = ensure_runtime_layout(settings)
    ensure_ffmpeg_available(settings)
    work_dir = paths.temp / "cleaning" / actor_slug
    demucs_out = work_dir / "demucs"
    rough_dir = work_dir / "rough"
    cleaned_dir = paths.dataset_cleaned / actor_slug

    if work_dir.exists():
        shutil.rmtree(work_dir)
    if cleaned_dir.exists():
        shutil.rmtree(cleaned_dir)
    rough_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    rough_paths = [_copy_to_workdir(Path(path), rough_dir, index) for index, path in enumerate(input_files, 1)]
    sources_to_slice = rough_paths if split_only else [_run_demucs(settings, source, demucs_out) for source in rough_paths]

    exported_count = 0
    for source_index, source_path in enumerate(sources_to_slice, 1):
        exported_count += _slice_and_export(settings, source_path, cleaned_dir, actor_slug, source_index, exported_count)

    if exported_count == 0:
        raise VoiceCloneError(
            "Aucun segment valide de 4 a 10 secondes n'a ete detecte. "
            "Essaie des sources plus longues ou ajuste les seuils de silence."
        )

    return cleaned_dir


def clean_single_file(settings: Settings, input_file: str, quality: int = 4) -> Path:
    if not input_file:
        raise VoiceCloneError("Ajoute un fichier audio a nettoyer.")

    paths = ensure_runtime_layout(settings)
    source = Path(input_file)
    if not source.exists():
        raise VoiceCloneError(f"Fichier introuvable: {source}")

    timestamp = source.stat().st_mtime_ns
    file_slug = safe_name(source.stem)
    work_dir = paths.temp / "clean_only" / f"{file_slug}_{timestamp}"
    rough_dir = work_dir / "rough"
    demucs_out = work_dir / "demucs"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    rough_dir.mkdir(parents=True, exist_ok=True)

    rough_path = _copy_to_workdir(source, rough_dir, 1)
    quality = max(1, min(10, int(quality)))
    overlap = 0.25 + ((quality - 1) / 9) * 0.5

    vocal_path = _run_demucs(
        settings,
        rough_path,
        demucs_out,
        shifts=quality,
        overlap=overlap,
    )

    output_slug = safe_name(f"{source.stem}_demucs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_path = _dedupe_path(paths.clean_only / f"{output_slug}.wav")
    convert_file_to_wav_24bit(
        vocal_path,
        output_path,
        sample_rate=settings.audio.sample_rate,
        target_peak_dbfs=settings.audio.target_peak_dbfs,
    )
    return output_path


def _copy_to_workdir(source: Path, rough_dir: Path, index: int) -> Path:
    if not source.exists():
        raise VoiceCloneError(f"Fichier introuvable: {source}")
    destination = rough_dir / f"{index:03d}_{safe_name(source.stem)}{source.suffix.lower()}"
    shutil.copy2(source, destination)
    return destination


def _run_demucs(
    settings: Settings,
    source: Path,
    output_dir: Path,
    shifts: int | None = None,
    overlap: float | None = None,
) -> Path:
    ensure_ffmpeg_available(settings)

    command = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        settings.audio.demucs_model,
        "--device",
        settings.audio.demucs_device,
        "--out",
        str(output_dir),
    ]
    if shifts is not None:
        command.extend(["--shifts", str(shifts)])
    if overlap is not None:
        command.extend(["--overlap", str(overlap)])
    command.append(str(source))
    env = build_subprocess_env(settings)
    project_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    env["VOICECLONE_PRELOAD_TORCHCODEC"] = "1"

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=project_root,
        env=env,
    )
    if completed.returncode != 0:
        raise VoiceCloneError(
            "Demucs a echoue.\n"
            f"Commande: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    vocal_matches = sorted(output_dir.glob(f"{settings.audio.demucs_model}/**/{source.stem}/vocals.wav"))
    if not vocal_matches:
        vocal_matches = sorted(output_dir.glob(f"{settings.audio.demucs_model}/**/vocals.wav"))
    if not vocal_matches:
        raise VoiceCloneError(f"Demucs n'a pas produit de vocals.wav pour {source.name}.")
    return vocal_matches[-1]


def _dedupe_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise VoiceCloneError(f"Impossible de trouver un nom de sortie libre pour {path.name}.")


def _slice_and_export(
    settings: Settings,
    vocal_path: Path,
    cleaned_dir: Path,
    actor_slug: str,
    source_index: int,
    existing_count: int,
) -> int:
    audio = AudioSegment.from_file(vocal_path)
    audio = audio.set_channels(1).set_frame_rate(settings.audio.sample_rate)
    audio = normalize_peak(audio, settings.audio.target_peak_dbfs)

    min_ms = settings.audio.min_segment_seconds * 1000
    max_ms = settings.audio.max_segment_seconds * 1000
    silence_thresh = settings.audio.silence_thresh_dbfs
    if silence_thresh is None:
        silence_thresh = max(audio.dBFS - 18, -45)

    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=settings.audio.silence_min_len_ms,
        silence_thresh=silence_thresh,
        seek_step=20,
    )
    if not nonsilent_ranges:
        nonsilent_ranges = [[0, len(audio)]]

    segments = _compose_segments(nonsilent_ranges, min_ms, max_ms)
    exported = 0
    for local_index, (start_ms, end_ms) in enumerate(segments, 1):
        chunk = audio[start_ms:end_ms]
        if len(chunk) < min_ms:
            continue
        output_path = cleaned_dir / f"{actor_slug}_{source_index:03d}_{existing_count + exported + 1:04d}.wav"
        export_segment_wav_24bit(chunk, output_path, settings.audio.sample_rate)
        exported += 1
    return exported


def _compose_segments(ranges: list[list[int]], min_ms: int, max_ms: int) -> list[tuple[int, int]]:
    composed: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None

    for start, end in ranges:
        if current_start is None:
            current_start, current_end = start, end
            continue

        assert current_end is not None
        proposed_duration = end - current_start
        if proposed_duration <= max_ms or (current_end - current_start) < min_ms:
            current_end = end
        else:
            composed.extend(_split_long_segment(current_start, current_end, min_ms, max_ms))
            current_start, current_end = start, end

    if current_start is not None and current_end is not None:
        composed.extend(_split_long_segment(current_start, current_end, min_ms, max_ms))

    return composed


def _split_long_segment(start: int, end: int, min_ms: int, max_ms: int) -> list[tuple[int, int]]:
    duration = end - start
    if duration <= max_ms:
        return [(start, end)] if duration >= min_ms else []

    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        piece_end = min(cursor + max_ms, end)
        if piece_end - cursor >= min_ms:
            pieces.append((cursor, piece_end))
        elif pieces:
            previous_start, _ = pieces[-1]
            pieces[-1] = (previous_start, end)
        cursor = piece_end
    return pieces
