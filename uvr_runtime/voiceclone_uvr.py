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
    parser = argparse.ArgumentParser(description="VOICECLONE-QC UVR/audio-separator wrapper")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_filename", default="vocals_mel_band_roformer.ckpt")
    parser.add_argument("--quality", type=int, default=4)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    quality = max(1, min(10, int(args.quality)))
    mdxc_overlap = min(50, 2 + quality * 3)
    mdx_overlap = min(0.95, 0.05 + quality * 0.05)

    work_dir = RUNTIME_ROOT / "work"
    output_dir = work_dir / "uvr_out"
    model_dir = RUNTIME_ROOT / "models"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    separator_exe = RUNTIME_ROOT / ".venv" / "Scripts" / "audio-separator.exe"
    if not separator_exe.exists():
        raise FileNotFoundError(separator_exe)

    command = [
        str(separator_exe),
        str(input_path),
        "--model_filename",
        args.model_filename,
        "--model_file_dir",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--output_format",
        "WAV",
        "--single_stem",
        "Vocals",
        "--sample_rate",
        "48000",
        "--normalization",
        "0.9",
        "--mdxc_overlap",
        str(mdxc_overlap),
        "--mdx_overlap",
        f"{mdx_overlap:.2f}",
    ]

    print("VOICECLONE-QC UVR audio-separator", flush=True)
    print(f"Input: {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Model: {args.model_filename}", flush=True)
    print(f"Quality: {quality}/10", flush=True)

    env = os.environ.copy()
    configure_cache_env(env)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["AUDIO_SEPARATOR_MODEL_DIR"] = str(model_dir)
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            "UVR/audio-separator failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    candidates = sorted(
        [path for path in output_dir.glob("*") if path.suffix.lower() in {".wav", ".flac", ".mp3"}],
        key=lambda path: path.stat().st_mtime,
    )
    vocal_candidates = [path for path in candidates if "vocal" in path.stem.lower()]
    if vocal_candidates:
        candidates = vocal_candidates
    if not candidates:
        raise RuntimeError(f"UVR did not produce an audio file in {output_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[-1], output_path)
    print("UVR complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
