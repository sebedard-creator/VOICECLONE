from __future__ import annotations

import os
from pathlib import Path

from .config import Settings, get_runtime_paths


def cache_dirs(settings: Settings) -> dict[str, Path]:
    cache_root = get_runtime_paths(settings).cache
    return {
        "root": cache_root,
        "home": cache_root / "home",
        "hf_home": cache_root / "huggingface",
        "hf_hub": cache_root / "huggingface" / "hub",
        "hf_datasets": cache_root / "huggingface" / "datasets",
        "transformers": cache_root / "huggingface" / "transformers",
        "torch": cache_root / "torch",
        "torch_extensions": cache_root / "torch_extensions",
        "xdg": cache_root / "xdg",
        "matplotlib": cache_root / "matplotlib",
        "numba": cache_root / "numba",
        "localappdata": cache_root / "localappdata",
        "pip": cache_root / "pip",
        "gradio": cache_root / "gradio",
        "tmp": cache_root / "tmp",
        "keras": cache_root / "keras",
    }


def configure_cache_environment(settings: Settings) -> None:
    _apply_cache_environment(os.environ, settings)


def build_subprocess_env(settings: Settings, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    _apply_cache_environment(env, settings)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    return env


def _apply_cache_environment(env: dict[str, str], settings: Settings) -> None:
    dirs = cache_dirs(settings)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    env["VOICECLONE_CACHE_DIR"] = str(dirs["root"])
    env["HF_HOME"] = str(dirs["hf_home"])
    env["HUGGINGFACE_HUB_CACHE"] = str(dirs["hf_hub"])
    env["HF_HUB_CACHE"] = str(dirs["hf_hub"])
    env["HF_DATASETS_CACHE"] = str(dirs["hf_datasets"])
    env["TRANSFORMERS_CACHE"] = str(dirs["transformers"])
    env["TORCH_HOME"] = str(dirs["torch"])
    env["TORCH_EXTENSIONS_DIR"] = str(dirs["torch_extensions"])
    env["XDG_CACHE_HOME"] = str(dirs["xdg"])
    env["MPLCONFIGDIR"] = str(dirs["matplotlib"])
    env["NUMBA_CACHE_DIR"] = str(dirs["numba"])
    env["PIP_CACHE_DIR"] = str(dirs["pip"])
    env["LOCALAPPDATA"] = str(dirs["localappdata"])
    env["GRADIO_TEMP_DIR"] = str(dirs["gradio"])
    env["TEMP"] = str(dirs["tmp"])
    env["TMP"] = str(dirs["tmp"])
    env["KERAS_HOME"] = str(dirs["keras"])
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Some older packages, notably VoiceFixer, hard-code os.path.expanduser("~/.cache").
    env["HOME"] = str(dirs["home"])
    env["USERPROFILE"] = str(dirs["home"])
