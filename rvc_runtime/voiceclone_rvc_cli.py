from __future__ import annotations

import os
import sys
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


configure_cache_env()
os.chdir(RUNTIME_ROOT)
os.environ["RVC_CLI_ASSET_ROOT"] = str(RUNTIME_ROOT / ".rvc_cli")
os.environ["PATH"] = str(RUNTIME_ROOT) + os.pathsep + os.environ.get("PATH", "")

from rvc_cli.core import main  # noqa: E402


if __name__ == "__main__":
    # Optional repeatable noise for controlled comparisons; normal runs are unchanged.
    if os.environ.get("VOICECLONE_COMPARE_SEED"):
        import random
        import numpy as np
        import torch

        comparison_seed = int(os.environ["VOICECLONE_COMPARE_SEED"])
        random.seed(comparison_seed)
        np.random.seed(comparison_seed)
        torch.manual_seed(comparison_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(comparison_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    sys.argv = ["rvc-cli", "infer", *sys.argv[1:]]
    main()
