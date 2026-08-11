from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent


def configure_cache_env(env: dict[str, str]) -> None:
    cache_root = RUNTIME_ROOT.parent / "cache"
    dirs = {
        "home": cache_root / "home",
        "hf_home": cache_root / "huggingface",
        "hf_hub": cache_root / "huggingface" / "hub",
        "hf_datasets": cache_root / "huggingface" / "datasets",
        "transformers": cache_root / "huggingface" / "transformers",
        "torch": cache_root / "torch",
        "torch_extensions": cache_root / "torch_extensions",
        "xdg": cache_root / "xdg",
        "localappdata": cache_root / "localappdata",
        "matplotlib": cache_root / "matplotlib",
        "numba": cache_root / "numba",
        "pip": cache_root / "pip",
        "gradio": cache_root / "gradio",
        "tmp": cache_root / "tmp",
        "keras": cache_root / "keras",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    env["VOICECLONE_CACHE_DIR"] = str(cache_root)
    env["HF_HOME"] = str(dirs["hf_home"])
    env["HUGGINGFACE_HUB_CACHE"] = str(dirs["hf_hub"])
    env["HF_HUB_CACHE"] = str(dirs["hf_hub"])
    env["HF_DATASETS_CACHE"] = str(dirs["hf_datasets"])
    env["TRANSFORMERS_CACHE"] = str(dirs["transformers"])
    env["TORCH_HOME"] = str(dirs["torch"])
    env["TORCH_EXTENSIONS_DIR"] = str(dirs["torch_extensions"])
    env["XDG_CACHE_HOME"] = str(dirs["xdg"])
    env["LOCALAPPDATA"] = str(dirs["localappdata"])
    env["MPLCONFIGDIR"] = str(dirs["matplotlib"])
    env["NUMBA_CACHE_DIR"] = str(dirs["numba"])
    env["PIP_CACHE_DIR"] = str(dirs["pip"])
    env["GRADIO_TEMP_DIR"] = str(dirs["gradio"])
    env["TEMP"] = str(dirs["tmp"])
    env["TMP"] = str(dirs["tmp"])
    env["KERAS_HOME"] = str(dirs["keras"])
    env["HOME"] = str(dirs["home"])
    env["USERPROFILE"] = str(dirs["home"])
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main() -> int:
    parser = argparse.ArgumentParser(description="VOICECLONE-QC DeepFilterNet wrapper")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--strength", type=float, default=60.0)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    strength = int(round(max(1.0, min(100.0, float(args.strength)))))
    work_dir = RUNTIME_ROOT / "work"
    output_dir = work_dir / "deepfilter_out"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    deepfilter_exe = RUNTIME_ROOT / ".venv" / "Scripts" / "deepFilter.exe"
    if not deepfilter_exe.exists():
        raise FileNotFoundError(deepfilter_exe)

    command = [
        str(deepfilter_exe),
        "--output-dir",
        str(output_dir),
        "--atten-lim",
        str(strength),
        "--no-suffix",
        str(input_path),
    ]

    print("VOICECLONE-QC DeepFilterNet", flush=True)
    print(f"Input: {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Strength: {strength:g} dB", flush=True)

    env = os.environ.copy()
    configure_cache_env(env)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            "DeepFilterNet failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    candidates = sorted(
        [path for path in output_dir.glob("*") if path.suffix.lower() in {".wav", ".flac", ".mp3"}],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"DeepFilterNet did not produce an audio file in {output_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[-1], output_path)
    print("DeepFilterNet complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
