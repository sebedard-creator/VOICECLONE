from __future__ import annotations

import argparse
import gc
import os
import pathlib
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONUNBUFFERED", "1")

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


configure_cache_env()

# The pretrained Resemble Enhance hparams serialize a few relative paths as
# pathlib.PosixPath. Windows cannot instantiate that class directly.
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

import torch
import torchaudio

from resemble_enhance.enhancer.inference import enhance


def main() -> int:
    parser = argparse.ArgumentParser(description="VOICECLONE-QC Resemble Enhance CPU wrapper")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--solver", choices=["midpoint", "euler", "rk4"], default="midpoint")
    parser.add_argument("--quality", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if input_path.suffix.lower() != ".wav":
        raise ValueError("Only .wav files are supported.")
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    quality = max(1, min(8, int(args.quality)))
    nfe = quality * 16
    device = "cpu"
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    print("VOICECLONE-QC Resemble Enhance", flush=True)
    print(f"Input: {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Solver: {args.solver}", flush=True)
    print(f"Quality: {quality}/8 ({nfe} NFE)", flush=True)
    print("Loading audio...", flush=True)

    dwav, sr = torchaudio.load(str(input_path))
    dwav = dwav.mean(0)

    print("Restoration in progress. This can take a while on CPU...", flush=True)
    hwav, out_sr = enhance(
        dwav=dwav,
        sr=sr,
        device=device,
        nfe=nfe,
        solver=args.solver,
        lambd=1.0,
        tau=0.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), hwav[None], out_sr)
    print("Restoration complete.", flush=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
