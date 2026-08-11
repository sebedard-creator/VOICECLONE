from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, RootPathUnavailable


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = PROJECT_DIR / "config" / "settings.json"


DEFAULT_SETTINGS: dict[str, Any] = {
    "root_path": r"Y:\VOICECLONE",
    "colab_url": "https://colab.research.google.com/drive/1aDEnG2W3t0cJo2P-IMoOie8cyy9jIBZf",
    "gradio": {"host": "0.0.0.0", "port": 7860},
    "audio": {
        "sample_rate": 48000,
        "target_peak_dbfs": -3.0,
        "min_segment_seconds": 4,
        "max_segment_seconds": 10,
        "silence_min_len_ms": 450,
        "silence_thresh_dbfs": None,
        "demucs_model": "htdemucs",
        "demucs_device": "cpu",
        "ffmpeg_path": "",
    },
    "google_drive": {
        "auth_mode": "oauth",
        "service_account_path": "config/service_account.json",
        "oauth_client_path": "config/oauth_client.json",
        "oauth_token_path": "config/oauth_token.json",
        "upload_folder_id": "",
        "return_folder_id": "",
        "return_folder_name": "RVC_Output",
    },
    "rvc": {
        "repo_path": "",
        "python_executable": "python",
        "command_template": "",
        "f0_method": "rmvpe",
        "device": "cuda:0",
        "output_sample_rate": 48000,
    },
    "resemble": {
        "runtime_python": r"Y:\VOICECLONE\resemble_runtime\.venv\Scripts\python.exe",
        "wrapper_path": r"Y:\VOICECLONE\resemble_runtime\voiceclone_resemble_enhance.py",
        "device": "cpu",
    },
    "deepfilter": {
        "runtime_python": r"Y:\VOICECLONE\deepfilter_runtime\.venv\Scripts\python.exe",
        "wrapper_path": r"Y:\VOICECLONE\deepfilter_runtime\voiceclone_deepfilter.py",
        "device": "cpu",
    },
    "uvr": {
        "runtime_python": r"Y:\VOICECLONE\uvr_runtime\.venv\Scripts\python.exe",
        "wrapper_path": r"Y:\VOICECLONE\uvr_runtime\voiceclone_uvr.py",
        "model_filename": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "device": "cpu",
    },
    "voicefixer": {
        "runtime_python": r"Y:\VOICECLONE\voicefixer_runtime\.venv\Scripts\python.exe",
        "wrapper_path": r"Y:\VOICECLONE\voicefixer_runtime\voiceclone_voicefixer.py",
        "device": "cpu",
    },
}


COLAB_BRIDGE_README = """VOICECLONE-QC 1.01 - COMPATIBILITE COLAB

Cette application locale ne pilote pas Google Colab.
Elle agit comme un serveur de fichiers intelligent entre ce PC, Google Drive et un notebook RVC.

Workflow attendu:

1. Dans l'onglet Nettoyage dataset, preparer un comedien.
   Les fichiers nettoyes sont crees dans:
   dataset_cleaned/[Nom_Comedien]/

2. Dans l'onglet Cloud / Colab, zipper et uploader le dataset.
   Le ZIP envoye sur Drive contient un dossier:
   dataset/

   Ce format est volontaire: un notebook RVC standard doit pouvoir decompresser le ZIP
   et utiliser directement le dossier dataset/ comme dossier d'entrainement.

3. Ouvrir un notebook Google Colab RVC standard qui permet:
   - de monter Google Drive;
   - de decompresser le ZIP envoye par VOICECLONE-QC;
   - d'entrainer un modele RVC avec le dossier dataset/;
   - d'exporter les fichiers finaux .pth et .index.

4. A la fin de l'entrainement Colab, placer les fichiers .pth et .index dans le dossier Drive:
   RVC_Output/

   Le nom des fichiers doit contenir le nom du comedien utilise dans VOICECLONE-QC.
   Exemple:
   Jean_Dupont.pth
   Jean_Dupont.index

5. Dans VOICECLONE-QC, lancer le Watchdog.
   Il scanne RVC_Output/ toutes les 30 secondes et rapatrie automatiquement le couple
   .pth/.index dans:
   models_bank/

Notes:
- Le fichier service_account.json doit etre place dans config/service_account.json.
- Le dossier RVC_Output doit etre visible par le Service Account Google.
- Si plusieurs dossiers RVC_Output existent, renseigner google_drive.return_folder_id
  dans config/settings.json pour eviter toute ambiguite.
"""


@dataclass(frozen=True)
class GradioSettings:
    host: str
    port: int


@dataclass(frozen=True)
class AudioSettings:
    sample_rate: int
    target_peak_dbfs: float
    min_segment_seconds: int
    max_segment_seconds: int
    silence_min_len_ms: int
    silence_thresh_dbfs: float | None
    demucs_model: str
    demucs_device: str
    ffmpeg_path: str


@dataclass(frozen=True)
class GoogleDriveSettings:
    auth_mode: str
    service_account_path: str
    oauth_client_path: str
    oauth_token_path: str
    upload_folder_id: str
    return_folder_id: str
    return_folder_name: str


@dataclass(frozen=True)
class RvcSettings:
    repo_path: str
    python_executable: str
    command_template: str
    f0_method: str
    device: str
    output_sample_rate: int


@dataclass(frozen=True)
class ResembleSettings:
    runtime_python: str
    wrapper_path: str
    device: str


@dataclass(frozen=True)
class DeepFilterSettings:
    runtime_python: str
    wrapper_path: str
    device: str


@dataclass(frozen=True)
class UvrSettings:
    runtime_python: str
    wrapper_path: str
    model_filename: str
    device: str


@dataclass(frozen=True)
class VoiceFixerSettings:
    runtime_python: str
    wrapper_path: str
    device: str


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    cache: Path
    config: Path
    dataset_cleaned: Path
    clean_only: Path
    models_bank: Path
    outputs: Path
    restoration_outputs: Path
    deepfilter_outputs: Path
    uvr_outputs: Path
    voicefixer_outputs: Path
    staging: Path
    temp: Path
    voice_bank_json: Path
    colab_readme: Path


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    root_path: Path
    gradio: GradioSettings
    audio: AudioSettings
    google_drive: GoogleDriveSettings
    rvc: RvcSettings
    resemble: ResembleSettings
    deepfilter: DeepFilterSettings
    uvr: UvrSettings
    voicefixer: VoiceFixerSettings


def ensure_default_settings_file(path: Path = DEFAULT_SETTINGS_PATH) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_SETTINGS, indent=2), encoding="utf-8")


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
    if not path.exists():
        raise ConfigurationError(f"Fichier de configuration introuvable: {path}")

    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    merged = _deep_merge(DEFAULT_SETTINGS, raw)

    return Settings(
        raw=merged,
        root_path=Path(merged["root_path"]),
        gradio=GradioSettings(**merged["gradio"]),
        audio=AudioSettings(**merged["audio"]),
        google_drive=GoogleDriveSettings(**merged["google_drive"]),
        rvc=RvcSettings(**merged["rvc"]),
        resemble=ResembleSettings(**merged["resemble"]),
        deepfilter=DeepFilterSettings(**merged["deepfilter"]),
        uvr=UvrSettings(**merged["uvr"]),
        voicefixer=VoiceFixerSettings(**merged["voicefixer"]),
    )


def get_runtime_paths(settings: Settings) -> RuntimePaths:
    root = settings.root_path.expanduser()
    return RuntimePaths(
        root=root,
        cache=root / "cache",
        config=root / "config",
        dataset_cleaned=root / "dataset_cleaned",
        clean_only=root / "Nettoyage Seul",
        models_bank=root / "models_bank",
        outputs=root / "outputs",
        restoration_outputs=root / "outputs" / "restoration",
        deepfilter_outputs=root / "outputs" / "deepfilternet",
        uvr_outputs=root / "outputs" / "uvr",
        voicefixer_outputs=root / "outputs" / "voicefixer",
        staging=root / "staging",
        temp=root / "temp",
        voice_bank_json=root / "config" / "voice_bank.json",
        colab_readme=root / "README.txt",
    )


def ensure_runtime_layout(settings: Settings) -> RuntimePaths:
    paths = get_runtime_paths(settings)
    validate_root_path(paths.root)
    for folder in (
        paths.root,
        paths.cache,
        paths.config,
        paths.dataset_cleaned,
        paths.clean_only,
        paths.models_bank,
        paths.outputs,
        paths.restoration_outputs,
        paths.deepfilter_outputs,
        paths.uvr_outputs,
        paths.voicefixer_outputs,
        paths.staging,
        paths.temp,
    ):
        folder.mkdir(parents=True, exist_ok=True)
    if not paths.voice_bank_json.exists():
        paths.voice_bank_json.write_text('{"voices": []}\n', encoding="utf-8")
    paths.colab_readme.write_text(COLAB_BRIDGE_README, encoding="utf-8")
    return paths


def validate_root_path(root: Path) -> None:
    drive = root.drive
    if drive:
        drive_root = Path(f"{drive}\\")
        if not drive_root.exists():
            raise RootPathUnavailable(
                f"Le disque {drive} est introuvable. "
                f"Reconnecte le disque ou change root_path dans {DEFAULT_SETTINGS_PATH}."
            )
    if root.exists() and not root.is_dir():
        raise RootPathUnavailable(f"Le root_path existe mais n'est pas un dossier: {root}")


def resolve_runtime_file(settings: Settings, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return get_runtime_paths(settings).root / path


def _deep_merge(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(defaults)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
