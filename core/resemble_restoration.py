from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from .cache_env import build_subprocess_env
from .config import Settings, ensure_runtime_layout
from .errors import ConfigurationError, VoiceCloneError
from .naming import safe_name


VALID_SOLVERS = {"midpoint", "euler", "rk4"}


def restore_audio(
    settings: Settings,
    input_path: Path,
    solver: str = "midpoint",
    quality: int = 4,
) -> Path:
    paths = ensure_runtime_layout(settings)
    source = Path(input_path)
    if not source.exists():
        raise VoiceCloneError(f"Fichier introuvable: {source}")
    if source.suffix.lower() != ".wav":
        raise VoiceCloneError("Seuls les fichiers .wav sont acceptes pour la restauration.")

    solver = (solver or "midpoint").strip().lower()
    if solver not in VALID_SOLVERS:
        raise VoiceCloneError(f"Solver invalide: {solver}. Choisis midpoint, euler ou rk4.")

    quality = max(1, min(8, int(quality)))
    runtime_python = Path(settings.resemble.runtime_python)
    wrapper_path = Path(settings.resemble.wrapper_path)
    if not runtime_python.exists():
        raise ConfigurationError(f"Runtime Resemble introuvable: {runtime_python}")
    if not wrapper_path.exists():
        raise ConfigurationError(f"Wrapper Resemble introuvable: {wrapper_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_slug = safe_name(f"{source.stem}_resemble_enhance_{timestamp}")
    prepared_input = _copy_input_to_workdir(source, paths.temp / "resemble_inputs", output_slug)
    output_path = _dedupe_path(paths.restoration_outputs / f"{output_slug}.wav")

    try:
        _run_resemble_command(
            runtime_python=runtime_python,
            wrapper_path=wrapper_path,
            input_path=prepared_input,
            output_path=output_path,
            solver=solver,
            quality=quality,
            settings=settings,
        )
        if not output_path.exists():
            raise VoiceCloneError(
                "Resemble Enhance s'est termine sans produire le fichier attendu: "
                f"{output_path}"
            )
        _delete_if_temporary_gradio_source(source)
        return output_path
    finally:
        _safe_delete(prepared_input)
        _clear_cuda_cache()


def _copy_input_to_workdir(source: Path, work_dir: Path, output_slug: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = work_dir / f"{output_slug}_input.wav"
    shutil.copy2(source, prepared_path)
    return prepared_path


def _run_resemble_command(
    runtime_python: Path,
    wrapper_path: Path,
    input_path: Path,
    output_path: Path,
    solver: str,
    quality: int,
    settings: Settings,
) -> None:
    command = [
        str(runtime_python),
        str(wrapper_path),
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--solver",
        solver,
        "--quality",
        str(quality),
        "--device",
        "cpu",
    ]
    env = build_subprocess_env(settings, {"CUDA_VISIBLE_DEVICES": ""})

    completed = subprocess.run(
        command,
        cwd=str(wrapper_path.parent),
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        stdout = completed.stdout or "(vide)"
        stderr = completed.stderr or "(vide)"
        raise VoiceCloneError(
            "La restauration Resemble Enhance a echoue.\n"
            f"Code retour: {completed.returncode}\n"
            f"Commande: {_format_command(command)}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )


def _delete_if_temporary_gradio_source(path: Path) -> None:
    try:
        resolved = path.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        parts = {part.lower() for part in resolved.parts}
        if "gradio" in parts and resolved.is_relative_to(temp_root):
            _safe_delete(resolved)
    except Exception:
        return


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


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        return
