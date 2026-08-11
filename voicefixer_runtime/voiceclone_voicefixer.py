from __future__ import annotations

import argparse
import os
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent


def configure_cache_env() -> None:
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
    os.environ["VOICECLONE_CACHE_DIR"] = str(cache_root)
    os.environ["HF_HOME"] = str(dirs["hf_home"])
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(dirs["hf_hub"])
    os.environ["HF_HUB_CACHE"] = str(dirs["hf_hub"])
    os.environ["HF_DATASETS_CACHE"] = str(dirs["hf_datasets"])
    os.environ["TRANSFORMERS_CACHE"] = str(dirs["transformers"])
    os.environ["TORCH_HOME"] = str(dirs["torch"])
    os.environ["TORCH_EXTENSIONS_DIR"] = str(dirs["torch_extensions"])
    os.environ["XDG_CACHE_HOME"] = str(dirs["xdg"])
    os.environ["LOCALAPPDATA"] = str(dirs["localappdata"])
    os.environ["MPLCONFIGDIR"] = str(dirs["matplotlib"])
    os.environ["NUMBA_CACHE_DIR"] = str(dirs["numba"])
    os.environ["PIP_CACHE_DIR"] = str(dirs["pip"])
    os.environ["GRADIO_TEMP_DIR"] = str(dirs["gradio"])
    os.environ["TEMP"] = str(dirs["tmp"])
    os.environ["TMP"] = str(dirs["tmp"])
    os.environ["KERAS_HOME"] = str(dirs["keras"])
    os.environ["HOME"] = str(dirs["home"])
    os.environ["USERPROFILE"] = str(dirs["home"])
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main() -> int:
    parser = argparse.ArgumentParser(description="VOICECLONE-QC VoiceFixer wrapper")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["MPLBACKEND"] = "Agg"
    configure_cache_env()

    from voicefixer import VoiceFixer

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("VOICECLONE-QC VoiceFixer", flush=True)
    print(f"Input: {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Mode: {args.mode}", flush=True)

    voicefixer = VoiceFixer()
    voicefixer.restore(
        input=str(input_path),
        output=str(output_path),
        cuda=False,
        mode=args.mode,
    )

    if not output_path.exists():
        raise RuntimeError(f"VoiceFixer did not produce the expected file: {output_path}")
    print("VoiceFixer complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
