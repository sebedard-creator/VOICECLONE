from __future__ import annotations

import shlex
import shutil
import subprocess
import os
from datetime import datetime
from pathlib import Path

from .audio_utils import convert_file_to_wav_24bit
from .cache_env import build_subprocess_env
from .config import Settings, ensure_runtime_layout
from .errors import ConfigurationError, VoiceCloneError
from .model_bank import find_index_for_model
from .naming import safe_name


def convert_voice(
    settings: Settings,
    model_path: Path,
    guide_path: Path,
    transpose: int = 0,
    index_rate: float = 0.75,
    protect: float = 0.33,
) -> Path:
    paths = ensure_runtime_layout(settings)
    if not model_path.exists():
        raise VoiceCloneError(f"Modele .pth introuvable: {model_path}")
    if not guide_path.exists():
        raise VoiceCloneError(f"Fichier guide introuvable: {guide_path}")

    index_path = find_index_for_model(settings, model_path)
    if index_path is None:
        raise VoiceCloneError(f"Aucun fichier .index correspondant a {model_path.name}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_slug = safe_name(f"{model_path.stem}_{guide_path.stem}_{timestamp}")
    raw_output = paths.temp / "rvc" / f"{output_slug}_raw.wav"
    final_output = paths.outputs / f"{output_slug}.wav"
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    prepared_guide_path = _prepare_guide_file(guide_path, paths.temp / "rvc_inputs", output_slug)

    try:
        _run_rvc_command(
            settings=settings,
            model_path=model_path,
            index_path=index_path,
            guide_path=prepared_guide_path,
            output_path=raw_output,
            transpose=transpose,
            index_rate=index_rate,
            protect=protect,
        )
        if not raw_output.exists():
            raise VoiceCloneError(
                "La commande RVC s'est terminee sans produire le fichier attendu: "
                f"{raw_output}"
            )
        convert_file_to_wav_24bit(
            raw_output,
            final_output,
            sample_rate=settings.rvc.output_sample_rate,
            target_peak_dbfs=-3.0,
        )
        return final_output
    finally:
        _clear_cuda_cache()


def _prepare_guide_file(guide_path: Path, work_dir: Path, output_slug: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    safe_suffix = guide_path.suffix.lower() or ".wav"
    prepared_path = work_dir / f"{output_slug}_input{safe_suffix}"
    shutil.copy2(guide_path, prepared_path)
    return prepared_path


def _run_rvc_command(
    settings: Settings,
    model_path: Path,
    index_path: Path,
    guide_path: Path,
    output_path: Path,
    transpose: int,
    index_rate: float,
    protect: float,
) -> None:
    template = settings.rvc.command_template.strip()
    if not template:
        raise ConfigurationError(
            "Le moteur RVC n'est pas encore configure. "
            "Renseigne rvc.command_template dans config/settings.json pour ton repo RVC."
        )

    repo_path = Path(settings.rvc.repo_path).expanduser() if settings.rvc.repo_path else None
    if repo_path and not repo_path.exists():
        raise ConfigurationError(f"rvc.repo_path introuvable: {repo_path}")

    command_text = template.format(
        model_path=str(model_path),
        index_path=str(index_path),
        input_path=str(guide_path),
        output_path=str(output_path),
        f0_method=settings.rvc.f0_method or "rmvpe",
        device=settings.rvc.device,
        transpose=transpose,
        index_rate=index_rate,
        protect=protect,
    )
    command = shlex.split(command_text, posix=True)
    executable = command[0]
    if shutil.which(executable) is None and not Path(executable).exists():
        raise ConfigurationError(f"Executable introuvable pour RVC: {executable}")

    env = build_subprocess_env(settings)

    completed = subprocess.run(
        command,
        cwd=str(repo_path) if repo_path else None,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        stdout = completed.stdout or "(vide)"
        stderr = completed.stderr or "(vide)"
        raise VoiceCloneError(
            "La conversion RVC a echoue.\n"
            f"Code retour: {completed.returncode}\n"
            f"Commande: {command_text}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        return
