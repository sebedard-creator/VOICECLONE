from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from .audio_utils import convert_file_to_wav_24bit
from .cache_env import build_subprocess_env
from .config import Settings, ensure_runtime_layout
from .errors import ConfigurationError, VoiceCloneError
from .ffmpeg_utils import ensure_ffmpeg_available
from .naming import safe_name


UVR_MODELS: dict[str, str] = {
    "RoFormer vocal - qualite": "vocals_mel_band_roformer.ckpt",
    "BS-RoFormer vocal - alternatif": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "MDX vocal - rapide/leger": "UVR-MDX-NET_Main_406.onnx",
}

VOICEFIXER_MODES: dict[str, int] = {
    "Mode 0 - restauration normale": 0,
    "Mode 1 - preprocessing spectral": 1,
    "Mode 2 - voix tres degradee": 2,
}

def run_deepfilter_denoise(settings: Settings, input_file: str, strength: float = 60.0) -> Path:
    paths = ensure_runtime_layout(settings)
    source = _validate_input(input_file)
    runtime_python = Path(settings.deepfilter.runtime_python)
    wrapper_path = Path(settings.deepfilter.wrapper_path)
    _validate_runtime(runtime_python, wrapper_path, "DeepFilterNet")
    ensure_ffmpeg_available(settings)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_slug = safe_name(f"{source.stem}_deepfilternet_{timestamp}")
    prepared_input = _copy_input(source, paths.temp / "deepfilter_inputs", output_slug)
    raw_output = paths.temp / "deepfilter" / f"{output_slug}_raw.wav"
    final_output = _dedupe_path(paths.deepfilter_outputs / f"{output_slug}.wav")

    try:
        _run_wrapper(
            runtime_python=runtime_python,
            wrapper_path=wrapper_path,
            args=[
                "--input_path",
                str(prepared_input),
                "--output_path",
                str(raw_output),
                "--strength",
                str(max(1.0, min(100.0, float(strength)))),
            ],
            label="DeepFilterNet",
            settings=settings,
        )
        if not raw_output.exists():
            raise VoiceCloneError(f"DeepFilterNet n'a pas produit le fichier attendu: {raw_output}")
        convert_file_to_wav_24bit(
            raw_output,
            final_output,
            sample_rate=settings.audio.sample_rate,
            target_peak_dbfs=settings.audio.target_peak_dbfs,
        )
        return final_output
    finally:
        _safe_delete(prepared_input)


def run_voicefixer_restoration(settings: Settings, input_file: str, mode_label: str) -> Path:
    paths = ensure_runtime_layout(settings)
    source = _validate_input(input_file)
    runtime_python = Path(settings.voicefixer.runtime_python)
    wrapper_path = Path(settings.voicefixer.wrapper_path)
    _validate_runtime(runtime_python, wrapper_path, "VoiceFixer")
    ensure_ffmpeg_available(settings)

    mode = VOICEFIXER_MODES.get(mode_label, 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_slug = safe_name(f"{source.stem}_voicefixer_{timestamp}")
    prepared_input = _copy_input(source, paths.temp / "voicefixer_inputs", output_slug)
    raw_output = paths.temp / "voicefixer" / f"{output_slug}_raw.wav"
    final_output = _dedupe_path(paths.voicefixer_outputs / f"{output_slug}.wav")

    try:
        _run_wrapper(
            runtime_python=runtime_python,
            wrapper_path=wrapper_path,
            args=[
                "--input_path",
                str(prepared_input),
                "--output_path",
                str(raw_output),
                "--mode",
                str(mode),
            ],
            label="VoiceFixer",
            settings=settings,
        )
        if not raw_output.exists():
            raise VoiceCloneError(f"VoiceFixer n'a pas produit le fichier attendu: {raw_output}")
        convert_file_to_wav_24bit(
            raw_output,
            final_output,
            sample_rate=settings.audio.sample_rate,
            target_peak_dbfs=settings.audio.target_peak_dbfs,
        )
        return final_output
    finally:
        _safe_delete(prepared_input)


def run_uvr_isolation(settings: Settings, input_file: str, model_label: str, quality: int = 4) -> Path:
    paths = ensure_runtime_layout(settings)
    source = _validate_input(input_file)
    runtime_python = Path(settings.uvr.runtime_python)
    wrapper_path = Path(settings.uvr.wrapper_path)
    _validate_runtime(runtime_python, wrapper_path, "UVR")
    ensure_ffmpeg_available(settings)

    model_filename = UVR_MODELS.get(model_label) or settings.uvr.model_filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_slug = safe_name(f"{source.stem}_uvr_{timestamp}")
    prepared_input = _copy_input(source, paths.temp / "uvr_inputs", output_slug)
    raw_output = paths.temp / "uvr" / f"{output_slug}_raw.wav"
    final_output = _dedupe_path(paths.uvr_outputs / f"{output_slug}.wav")

    try:
        _run_wrapper(
            runtime_python=runtime_python,
            wrapper_path=wrapper_path,
            args=[
                "--input_path",
                str(prepared_input),
                "--output_path",
                str(raw_output),
                "--model_filename",
                model_filename,
                "--quality",
                str(max(1, min(10, int(quality)))),
            ],
            label="UVR",
            settings=settings,
        )
        if not raw_output.exists():
            raise VoiceCloneError(f"UVR n'a pas produit le fichier attendu: {raw_output}")
        convert_file_to_wav_24bit(
            raw_output,
            final_output,
            sample_rate=settings.audio.sample_rate,
            target_peak_dbfs=settings.audio.target_peak_dbfs,
        )
        return final_output
    finally:
        _safe_delete(prepared_input)


def _validate_input(input_file: str) -> Path:
    if not input_file:
        raise VoiceCloneError("Ajoute un fichier audio.")
    source = Path(input_file)
    if not source.exists():
        raise VoiceCloneError(f"Fichier introuvable: {source}")
    return source


def _validate_runtime(runtime_python: Path, wrapper_path: Path, label: str) -> None:
    if not runtime_python.exists():
        raise ConfigurationError(f"Runtime {label} introuvable: {runtime_python}")
    if not wrapper_path.exists():
        raise ConfigurationError(f"Wrapper {label} introuvable: {wrapper_path}")


def _copy_input(source: Path, work_dir: Path, output_slug: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    destination = work_dir / f"{output_slug}_input{source.suffix.lower() or '.wav'}"
    shutil.copy2(source, destination)
    return destination


def _run_wrapper(
    runtime_python: Path,
    wrapper_path: Path,
    args: list[str],
    label: str,
    settings: Settings,
) -> None:
    command = [str(runtime_python), str(wrapper_path), *args]
    env = build_subprocess_env(settings)

    completed = subprocess.run(
        command,
        cwd=str(wrapper_path.parent),
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise VoiceCloneError(
            f"{label} a echoue.\n"
            f"Code retour: {completed.returncode}\n"
            f"Commande: {_format_command(command)}\n"
            f"STDOUT:\n{completed.stdout or '(vide)'}\n"
            f"STDERR:\n{completed.stderr or '(vide)'}"
        )


def _safe_delete(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        return


def _dedupe_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise VoiceCloneError(f"Impossible de trouver un nom de sortie libre pour {path.name}.")


def _format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)
