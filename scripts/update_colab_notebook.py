"""One-time structured update for the VOICECLONE-QC Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "VOICECLONE_QC_RVC_Bridge.ipynb"


def source(text: str) -> list[str]:
    return [line + "\n" for line in text.strip().splitlines()]


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def old_cell(cells: list[dict], title: str) -> dict:
    for cell in cells:
        content = "".join(cell.get("source", []))
        if title in content:
            cell["execution_count"] = None
            cell["outputs"] = []
            return cell
    raise RuntimeError(f"Notebook cell not found: {title}")


def title(cell: dict, value: str) -> dict:
    lines = cell["source"]
    lines[0] = f"#@title {value}\n"
    return cell


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    old = notebook["cells"]

    config = code(
        '''#@title 1. Configuration VOICECLONE-QC v1.2.3
from pathlib import Path

NOTEBOOK_VERSION = "1.2.3"
MODEL_NAME = "Alertes_Stephanie"  #@param {type:"string"}
RUN_MODE = "new"  #@param ["new", "resume"]
TARGET_SAMPLE_RATE = "40k"
MODEL_ARCHITECTURE = "v2"
PRETRAIN_TYPE = "OV2"
PITCH_METHOD = "rmvpe"
TOTAL_EPOCHS = 200  #@param {type:"integer"}
SAVE_FREQUENCY = 10  #@param {type:"integer"}
BATCH_SIZE = 7  #@param {type:"integer"}
CHECKPOINT_SYNC_SECONDS = 120  #@param {type:"integer"}

# Pin the exact RVC code and model assets used by this notebook.
RVC_REPOSITORY_COMMIT = "d618280cdef162c39bf74d03362592db5c41ad80"
TORCHCREPE_COMMIT = "19e2ec3d494c0797a5ff2a11408ec5838fba6681"
ORVC_REVISION = "425f8006582161af571e0d7f5ce646a535a14b65"

DRIVE_DATASET_DIR = "/content/drive/MyDrive/VOICECLONE_Datasets"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/RVC_Output"
DRIVE_TRAINING_DIR = "/content/drive/MyDrive/VOICECLONE_Training"
DRIVE_RUN_DIR = Path(DRIVE_TRAINING_DIR) / MODEL_NAME
DRIVE_EXPERIMENT_DIR = DRIVE_RUN_DIR / "experiment"
DRIVE_CHECKPOINT_DIR = DRIVE_RUN_DIR / "checkpoints"

ZIP_PATH = f"{DRIVE_DATASET_DIR}/{MODEL_NAME}_dataset_cleaned.zip"
NOW_DIR = "/content/Mangio-RVC-Fork"
EXP_DIR = f"{NOW_DIR}/logs/{MODEL_NAME}"
DATASET_DIR = f"/content/voiceclone_qc/{MODEL_NAME}/dataset"

if not MODEL_NAME.strip():
    raise ValueError("MODEL_NAME cannot be empty.")
if RUN_MODE not in {"new", "resume"}:
    raise ValueError("RUN_MODE must be 'new' or 'resume'.")
if (TARGET_SAMPLE_RATE, MODEL_ARCHITECTURE, PRETRAIN_TYPE) != ("40k", "v2", "OV2"):
    raise ValueError("This validated notebook is pinned to 40k, v2 and OV2 assets.")
if PITCH_METHOD != "rmvpe":
    raise ValueError("Use RMVPE to keep the production quality setting consistent.")
if TOTAL_EPOCHS < 1 or SAVE_FREQUENCY < 1 or BATCH_SIZE < 1:
    raise ValueError("Training values must be positive integers.")

print(
    f"VOICECLONE-QC RVC Bridge v{NOTEBOOK_VERSION} | "
    f"Model: {MODEL_NAME} | mode: {RUN_MODE} | target epochs: {TOTAL_EPOCHS}"
)'''
    )

    mount_drive = code(
        '''#@title 2. Mount Google Drive - run this immediately
import os
from google.colab import drive

if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive")
else:
    print("Google Drive is already mounted.")

for directory in (DRIVE_DATASET_DIR, DRIVE_OUTPUT_DIR, DRIVE_TRAINING_DIR, DRIVE_RUN_DIR):
    Path(directory).mkdir(parents=True, exist_ok=True)

print("Google Drive is ready. You can now leave Colab to continue the setup.")'''
    )

    dependencies = code(
        '''#@title 3. Install Dependencies (Colab Python 3.12 compatible)
import os
import subprocess
import sys

# A failed/restarted repository cell can leave the process in a deleted folder.
# pip calls os.getcwd(), so always restore a valid Colab working directory first.
os.chdir("/content")

system_packages = ["build-essential", "python3-dev", "ffmpeg", "aria2"]
python_packages = [
    "faiss-cpu",
    "ffmpeg-python",
    "praat-parselmouth",
    "pyworld",
    "numpy",
    "numba",
    "librosa",
    "tensorboardX",
    "tensorboard",
    "onnx",
    "onnxruntime-gpu",
    "torchcrepe",
    "python-dotenv",
    "av",
    "scikit-learn",
]

def run_command(command, label):
    print(f"Installing: {label}", flush=True)
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"INSTALLATION FAILED: {label} (exit code {completed.returncode})."
        )

print("Updating package list...", flush=True)
run_command(["apt-get", "update", "-qq"], "system package list")

for package in system_packages:
    run_command(["apt-get", "install", "-qq", "-y", package], f"system package {package}")

# Colab owns its system pip. Do not upgrade pip, setuptools, or wheel here.
# The flag is required by the current Python 3.12 Debian environment.
pip_prefix = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--disable-pip-version-check",
    "--no-input",
    "--prefer-binary",
    "--break-system-packages",
    "--upgrade",
]

for package in python_packages:
    run_command(pip_prefix + [package], f"Python package {package}")

run_command(pip_prefix + ["fairseq-fixed"], "Python package fairseq-fixed")
print("Dependencies ready.", flush=True)'''
    )

    clone = code(
        '''#@title 4. Download RVC Source Code (pinned)
import os
import shutil
import subprocess
import zipfile

# Never delete a repository while Python is still using it as the current path.
os.chdir("/content")

def download_github_archive(repository, revision, destination):
    archive_path = Path("/content") / f"{destination.name}-{revision[:12]}.zip"
    extract_dir = Path("/content") / f"{destination.name}-{revision[:12]}-extract"
    for path in (destination, archive_path, extract_dir):
        if path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()

    url = f"https://github.com/{repository}/archive/{revision}.zip"
    completed = subprocess.run(
        [
            "curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
            "--connect-timeout", "30", "--output", str(archive_path), url,
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Unable to download {repository} at {revision[:12]}.\\n"
            f"curl output:\\n{completed.stderr or completed.stdout}"
        )
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_dir)
    source_dirs = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(source_dirs) != 1:
        raise RuntimeError(f"Unexpected archive layout for {repository}: {source_dirs}")
    shutil.move(str(source_dirs[0]), str(destination))
    archive_path.unlink()
    shutil.rmtree(extract_dir)

repo_dir = Path("/content/Mangio-RVC-Fork")
torchcrepe_dir = Path("/content/torchcrepe")
download_github_archive("Mangio621/Mangio-RVC-Fork", RVC_REPOSITORY_COMMIT, repo_dir)
download_github_archive("maxrmorrison/torchcrepe", TORCHCREPE_COMMIT, torchcrepe_dir)
shutil.copytree(torchcrepe_dir / "torchcrepe", repo_dir / "torchcrepe", dirs_exist_ok=True)
os.chdir(repo_dir)
print(f"RVC source ready: {repo_dir} @ {RVC_REPOSITORY_COMMIT[:12]}")'''
    )

    download_models = code(
        '''#@title 6. Download verified pretrained models and RMVPE
import hashlib
import json
import subprocess

assets = {
    Path(NOW_DIR) / "pretrained_v2" / "f0G40k_OV2.pth": [
        f"https://huggingface.co/ORVC/Ov2Super/resolve/{ORVC_REVISION}/f0Ov2Super40kG.pth",
        "https://huggingface.co/ORVC/Ov2Super/resolve/main/f0Ov2Super40kG.pth",
    ],
    Path(NOW_DIR) / "pretrained_v2" / "f0D40k_OV2.pth": [
        f"https://huggingface.co/ORVC/Ov2Super/resolve/{ORVC_REVISION}/f0Ov2Super40kD.pth",
        "https://huggingface.co/ORVC/Ov2Super/resolve/main/f0Ov2Super40kD.pth",
    ],
    Path(NOW_DIR) / "hubert_base.pt": [
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
    ],
    Path(NOW_DIR) / "rmvpe.pt": [
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    ],
    Path(NOW_DIR) / "configs" / "40k.json": [
        f"https://raw.githubusercontent.com/Mangio621/Mangio-RVC-Fork/{RVC_REPOSITORY_COMMIT}/configs/40k.json",
    ],
}

def download_verified(destination, urls):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    failures = []
    for url in urls:
        if temporary.exists():
            temporary.unlink()
        try:
            subprocess.check_call([
                "curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
                "--connect-timeout", "30", "--output", str(temporary), url,
            ])
            if temporary.stat().st_size < 100:
                raise RuntimeError("downloaded file is unexpectedly small")
            temporary.replace(destination)
            return url, hashlib.sha256(destination.read_bytes()).hexdigest()
        except Exception as error:
            failures.append(f"{url}: {error}")
    raise RuntimeError(
        f"Unable to download {destination.name}. Tried:\\n" + "\\n".join(failures)
    )

manifest = {}
for destination, urls in assets.items():
    used_url, checksum = download_verified(destination, urls)
    manifest[str(destination.relative_to(NOW_DIR))] = {
        "url": used_url,
        "sha256": checksum,
        "bytes": destination.stat().st_size,
    }

manifest_path = Path(NOW_DIR) / "voiceclone_asset_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Verified {len(manifest)} assets: {manifest_path}")'''
    )

    restore = code(
        '''#@title 7. Restore a resumable training run from Drive
import shutil

local_experiment = Path(EXP_DIR)
if RUN_MODE == "resume":
    if not DRIVE_EXPERIMENT_DIR.exists():
        raise FileNotFoundError(
            f"No Drive checkpoint backup found for {MODEL_NAME}: {DRIVE_EXPERIMENT_DIR}"
        )
    if local_experiment.exists():
        shutil.rmtree(local_experiment)
    shutil.copytree(DRIVE_EXPERIMENT_DIR, local_experiment)
    print(f"Restored experiment: {DRIVE_EXPERIMENT_DIR} -> {local_experiment}")
else:
    if local_experiment.exists():
        shutil.rmtree(local_experiment)
    print("New run selected. Existing Drive backups are preserved until this run creates new ones.")'''
    )

    dataset = code(
        '''#@title 8. Load VOICECLONE-QC Dataset ZIP
import os
import shutil
import zipfile

if RUN_MODE == "resume":
    print("Resume mode: dataset import is skipped.")
else:
    dataset_dir = Path(DATASET_DIR)
    if not Path(ZIP_PATH).exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {ZIP_PATH}")
    if dataset_dir.parent.exists():
        shutil.rmtree(dataset_dir.parent)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(dataset_dir.parent)
    readme = dataset_dir / "README.txt"
    if readme.exists():
        readme.unlink()
    wav_files = sorted(dataset_dir.glob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No WAV files found in {dataset_dir}")
    print(f"Dataset ready: {dataset_dir} ({len(wav_files)} WAV files)")'''
    )

    preprocess = code(
        '''#@title 10. Preprocess Dataset
import os
import subprocess

if RUN_MODE == "resume":
    print("Resume mode: preprocessing is skipped.")
else:
    cpu_threads = min(2, os.cpu_count() or 1)
    Path(EXP_DIR).mkdir(parents=True, exist_ok=True)
    command = [
        "python", "trainset_preprocess_pipeline_print.py", DATASET_DIR, "40000",
        str(cpu_threads), EXP_DIR, "1",
    ]
    print(" ".join(command))
    subprocess.check_call(command)
    print("Preprocessing complete.")'''
    )

    features = code(
        '''#@title 12. Feature Extraction RMVPE
import os
import subprocess

if RUN_MODE == "resume":
    print("Resume mode: RMVPE and HuBERT extraction are skipped.")
else:
    cpu_threads = min(2, os.cpu_count() or 1)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{NOW_DIR}:{environment.get('PYTHONPATH', '')}"
    f0_command = ["python", "extract_f0_print.py", EXP_DIR, str(cpu_threads), "rmvpe", "128"]
    feature_command = ["python", "extract_feature_print.py", "device", "1", "0", "0", EXP_DIR, "v2"]
    print(" ".join(f0_command))
    subprocess.check_call(f0_command, env=environment)
    print(" ".join(feature_command))
    subprocess.check_call(feature_command, env=environment)
    print("Feature extraction complete.")'''
    )

    backup = code(
        '''#@title 13. Save preprocessed data and checkpoints to Drive
import shutil
import time

def copy_file_atomically(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    shutil.copy2(source, temporary)
    temporary.replace(destination)

def sync_checkpoints_to_drive():
    local = Path(EXP_DIR)
    copied = []
    for pattern in ("G_*.pth", "D_*.pth", "filelist.txt", "config.json"):
        for item in local.glob(pattern):
            copy_file_atomically(item, DRIVE_CHECKPOINT_DIR / item.name)
            copied.append(item.name)
    if copied:
        print(f"Checkpoint backup: {', '.join(sorted(copied))}")
    return copied

def sync_experiment_to_drive():
    local = Path(EXP_DIR)
    if not local.exists():
        raise FileNotFoundError(f"Experiment is missing: {local}")
    staging = DRIVE_RUN_DIR / "experiment_staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(local, staging)
    if DRIVE_EXPERIMENT_DIR.exists():
        shutil.rmtree(DRIVE_EXPERIMENT_DIR)
    staging.replace(DRIVE_EXPERIMENT_DIR)
    sync_checkpoints_to_drive()
    print(f"Full experiment backup complete: {DRIVE_EXPERIMENT_DIR}")

if RUN_MODE == "new":
    sync_experiment_to_drive()
else:
    print("Resume mode: the restored backup remains the source of truth.")'''
    )

    train = code(
        '''#@title 14. Train RVC Model with automatic checkpoint backups
import os
import subprocess
import time

os.chdir(NOW_DIR)
sync_checkpoints_to_drive()
environment = os.environ.copy()
environment["PYTHONPATH"] = f"{NOW_DIR}:{environment.get('PYTHONPATH', '')}"
command = [
    "python", "train_nsf_sim_cache_sid_load_pretrain.py",
    "-e", MODEL_NAME, "-sr", "40k", "-f0", "1", "-bs", str(BATCH_SIZE),
    "-g", "0", "-te", str(TOTAL_EPOCHS), "-se", str(SAVE_FREQUENCY),
    "-pg", "pretrained_v2/f0G40k_OV2.pth", "-pd", "pretrained_v2/f0D40k_OV2.pth",
    "-l", "1", "-c", "0", "-sw", "1", "-v", "v2", "-li", "3",
]
print(" ".join(command))
process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, env=environment,
)
last_sync = time.monotonic()
for line in process.stdout:
    print(line, end="")
    if time.monotonic() - last_sync >= CHECKPOINT_SYNC_SECONDS:
        sync_checkpoints_to_drive()
        last_sync = time.monotonic()
exit_code = process.wait()
sync_checkpoints_to_drive()
if exit_code != 0:
    raise RuntimeError(f"Training stopped with exit code {exit_code}. Checkpoints remain on Drive.")
sync_experiment_to_drive()
print("Training complete and fully backed up to Drive.")'''
    )

    export = code(
        '''#@title 16. Export model and index to RVC_Output
import shutil
from datetime import datetime

log_dir = Path(EXP_DIR)
model_files = sorted(log_dir.glob("*.pth"), key=lambda item: item.stat().st_mtime)
index_files = sorted(log_dir.glob("*.index"), key=lambda item: item.stat().st_mtime)
if not model_files or not index_files:
    raise FileNotFoundError("Model .pth or .index is missing. Run training and index creation first.")

source_model, source_index = model_files[-1], index_files[-1]
destination_model = Path(DRIVE_OUTPUT_DIR) / f"{MODEL_NAME}.pth"
destination_index = Path(DRIVE_OUTPUT_DIR) / f"{MODEL_NAME}.index"
if destination_model.exists() or destination_index.exists():
    archive = Path(DRIVE_OUTPUT_DIR) / "archive" / MODEL_NAME / datetime.now().strftime("%Y%m%d_%H%M%S")
    archive.mkdir(parents=True, exist_ok=True)
    for existing in (destination_model, destination_index):
        if existing.exists():
            shutil.copy2(existing, archive / existing.name)
    print(f"Previous export archived: {archive}")

shutil.copy2(source_model, destination_model)
shutil.copy2(source_index, destination_index)
if destination_model.stat().st_size == 0 or destination_index.stat().st_size == 0:
    raise RuntimeError("Export verification failed: an output file is empty.")
print(f"Exported: {destination_model}")
print(f"Exported: {destination_index}")'''
    )

    cells = [
        markdown("# VOICECLONE-QC RVC Bridge\n\nRun the Drive authorization cell immediately after configuration. This notebook is pinned to the validated RVC v2 / 40k / RMVPE pipeline and saves resumable checkpoints to Drive."),
        config,
        mount_drive,
        dependencies,
        clone,
        title(old_cell(old, "GPU Check"), "5. GPU Check"),
        download_models,
        restore,
        dataset,
        title(old_cell(old, "Setup CSVDB"), "9. Setup CSVDB"),
        preprocess,
        title(old_cell(old, "Python 3.12 / PyTorch Compatibility Patch"), "11. Python 3.12 / PyTorch Compatibility Patch"),
        features,
        title(old_cell(old, "Matplotlib / NumPy Compatibility Patch"), "12b. Matplotlib / NumPy Compatibility Patch"),
        title(old_cell(old, "RGB Canvas Patch"), "12c. RGB Canvas Patch"),
        backup,
        train,
        title(old_cell(old, "Train Index"), "15. Train Index"),
        export,
        title(old_cell(old, "Auto-disconnect runtime"), "17. Auto-disconnect runtime after successful export"),
    ]
    notebook["cells"] = cells
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated notebook safely: {NOTEBOOK}")


if __name__ == "__main__":
    main()
