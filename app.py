from __future__ import annotations

import json
import shutil
import traceback
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import gradio as gr

from core.audio_cleaning import clean_dataset, clean_single_file
from core.cache_env import configure_cache_environment
from core.config import (
    DEFAULT_SETTINGS_PATH,
    Settings,
    ensure_default_settings_file,
    ensure_runtime_layout,
    load_settings,
)
from core.drive_sync import GoogleDriveClient, zip_cleaned_dataset
from core.drive_archive import DriveArchive, snapshot_summary
from core.errors import VoiceCloneError
from core.ffmpeg_utils import ensure_ffmpeg_available
from core.local_audio_tools import (
    UVR_MODELS,
    VOICEFIXER_MODES,
    run_deepfilter_denoise,
    run_uvr_isolation,
    run_voicefixer_restoration,
)
from core.model_bank import discover_voice_models, find_index_for_model, load_voice_bank
from core.resemble_restoration import restore_audio
from core.rvc_inference import convert_voice
from core.task_queue import TASK_GATE


ensure_default_settings_file(DEFAULT_SETTINGS_PATH)
SETTINGS = load_settings(DEFAULT_SETTINGS_PATH)
configure_cache_environment(SETTINGS)
COLAB_URL = str(SETTINGS.raw["colab_url"])

APP_CSS = """
.vc-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
}

.vc-topbar h1 {
    margin: 0;
}

.vc-colab-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 34px;
    padding: 7px 12px;
    border: 1px solid #2563eb;
    border-radius: 6px;
    color: #1d4ed8 !important;
    background: #eff6ff;
    font-weight: 650;
    text-decoration: none !important;
}

.vc-colab-link:hover {
    background: #dbeafe;
}

.vc-nav-button button {
    min-height: 34px !important;
    padding: 7px 12px !important;
    border: 1px solid #0f766e !important;
    border-radius: 6px !important;
    color: #0f766e !important;
    background: #f0fdfa !important;
    font-weight: 650 !important;
}

.vc-nav-button button:hover {
    background: #ccfbf1 !important;
}

.vc-colab-config-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 34px !important;
    padding: 7px 12px !important;
    border: 1px solid #7c3aed !important;
    border-radius: 6px !important;
    color: #5b21b6 !important;
    background: #f5f3ff !important;
    font-weight: 650 !important;
    text-decoration: none !important;
}

.vc-colab-config-link:hover {
    background: #ede9fe !important;
}

.vc-home-button button {
    min-height: 34px !important;
    padding: 7px 12px !important;
    border-radius: 6px !important;
}

.vc-danger-button,
.vc-danger-button button,
.vc-danger-button > button {
    min-height: 34px !important;
    padding: 7px 12px !important;
    border: 1px solid #991b1b !important;
    border-radius: 6px !important;
    color: #ffffff !important;
    background: #dc2626 !important;
    font-weight: 700 !important;
}

.vc-danger-button:hover,
.vc-danger-button button:hover,
.vc-danger-button > button:hover {
    background: #b91c1c !important;
    border-color: #7f1d1d !important;
    color: #ffffff !important;
}

.vc-danger-panel {
    border: 1px solid #dc2626 !important;
    border-left: 6px solid #dc2626 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

.vc-help {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    margin-left: 6px;
    border-radius: 999px;
    background: #e5e7eb;
    color: #374151;
    font-size: 12px;
    font-weight: 800;
    cursor: help;
    position: relative;
}

.vc-help:hover {
    background: #d1d5db;
}

.vc-help::after {
    content: attr(data-help);
    display: none;
    position: absolute;
    left: 0;
    bottom: calc(100% + 10px);
    transform: none;
    width: min(320px, calc(100vw - 48px));
    padding: 10px 12px;
    border-radius: 6px;
    background: #111827;
    color: white;
    font-size: 13px;
    font-weight: 500;
    line-height: 1.35;
    text-align: left;
    white-space: normal;
    z-index: 1000;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
}

.vc-help:hover::after {
    display: block;
}

.vc-busy {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background: #f9fafb;
    font-weight: 650;
}

.vc-busy-diamond {
    width: 18px;
    height: 18px;
    background: #2563eb;
    transform: rotate(45deg);
    animation: vc-spin-diamond 0.9s linear infinite;
}

@keyframes vc-spin-diamond {
    from {
        transform: rotate(45deg);
    }
    to {
        transform: rotate(405deg);
    }
}

#vc-toast-container {
    position: fixed;
    top: 18px;
    right: 18px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    z-index: 9999;
    pointer-events: none;
}

.vc-toast {
    width: min(360px, calc(100vw - 36px));
    padding: 12px 14px;
    border: 1px solid #16a34a;
    border-left: 6px solid #16a34a;
    border-radius: 8px;
    background: #f0fdf4;
    color: #14532d;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
    animation: vc-toast-in 0.18s ease-out;
}

.vc-toast-title {
    font-weight: 800;
    margin-bottom: 4px;
}

.vc-toast-body {
    font-size: 13px;
    line-height: 1.35;
    word-break: break-word;
}

@keyframes vc-toast-in {
    from {
        opacity: 0;
        transform: translateY(-8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
"""

BUSY_HTML = """
<div class="vc-busy">
    <span class="vc-busy-diamond"></span>
    <span>{message}</span>
</div>
"""

NOTIFICATION_HELPER_JS = r"""
function vcEnsureNotify() {
    if (window.VC_NOTIFY) {
        return window.VC_NOTIFY;
    }

    const helper = {
        enabled: false,
        soundEnabled: false,
        audioContext: null,

        basename(value) {
            if (!value) {
                return "";
            }
            let text = "";
            if (typeof value === "string") {
                text = value;
            } else if (Array.isArray(value) && value.length) {
                return this.basename(value[0]);
            } else if (typeof value === "object") {
                text = value.name || value.path || value.orig_name || value.url || "";
            }
            const normalized = String(text).replaceAll("\\", "/");
            return normalized.split("/").filter(Boolean).pop() || normalized;
        },

        cleanStatus(status) {
            return String(status || "").replace(/^\[OK\]\s*/, "").trim();
        },

        ensureContainer() {
            let container = document.getElementById("vc-toast-container");
            if (!container) {
                container = document.createElement("div");
                container.id = "vc-toast-container";
                document.body.appendChild(container);
            }
            return container;
        },

        toast(title, body) {
            const container = this.ensureContainer();
            const toast = document.createElement("div");
            toast.className = "vc-toast";
            const safeTitle = String(title || "VOICECLONE-QC").replace(/[<>&]/g, "");
            const safeBody = String(body || "").replace(/[<>&]/g, "");
            toast.innerHTML = `<div class="vc-toast-title">${safeTitle}</div><div class="vc-toast-body">${safeBody}</div>`;
            container.appendChild(toast);
            window.setTimeout(() => {
                toast.style.opacity = "0";
                toast.style.transform = "translateY(-8px)";
                toast.style.transition = "opacity 180ms ease, transform 180ms ease";
                window.setTimeout(() => toast.remove(), 220);
            }, 9000);
        },

        async unlockSound() {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) {
                return false;
            }
            if (!this.audioContext) {
                this.audioContext = new AudioContextClass();
            }
            await this.audioContext.resume();
            this.soundEnabled = true;
            return true;
        },

        async beep() {
            if (!this.soundEnabled) {
                return;
            }
            try {
                await this.unlockSound();
                const ctx = this.audioContext;
                const now = ctx.currentTime;
                [880, 1175].forEach((freq, index) => {
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = "sine";
                    osc.frequency.value = freq;
                    gain.gain.setValueAtTime(0.0001, now + index * 0.13);
                    gain.gain.exponentialRampToValueAtTime(0.12, now + index * 0.13 + 0.015);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + index * 0.13 + 0.12);
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.start(now + index * 0.13);
                    osc.stop(now + index * 0.13 + 0.14);
                });
            } catch (error) {
                console.warn("VOICECLONE-QC sound notification failed", error);
            }
        },

        async enable() {
            this.enabled = true;
            let browserStatus = "non supportees";
            if ("Notification" in window) {
                try {
                    if (Notification.permission === "default") {
                        await Notification.requestPermission();
                    }
                    browserStatus = Notification.permission;
                } catch (error) {
                    browserStatus = "bloquees";
                }
            }
            const soundOk = await this.unlockSound();
            const message = `Notifications visuelles actives. Navigateur: ${browserStatus}. Son: ${soundOk ? "actif" : "non supporte"}.`;
            this.toast("VOICECLONE-QC", message);
            this.beep();
            return message;
        },

        notify(toolName, status, outputValue) {
            const statusText = String(status || "").trim();
            if (!statusText.startsWith("[OK]")) {
                return;
            }
            const fileName = this.basename(outputValue);
            const detail = fileName ? `Fichier pret: ${fileName}` : this.cleanStatus(statusText);
            const title = `VOICECLONE-QC - ${toolName}`;
            this.toast(title, detail);
            if (this.enabled && "Notification" in window && Notification.permission === "granted") {
                try {
                    new Notification(title, {
                        body: detail,
                        tag: `voiceclone-qc-${toolName}`,
                    });
                } catch (error) {
                    console.warn("VOICECLONE-QC browser notification failed", error);
                }
            }
            this.beep();
        },
    };

    window.VC_NOTIFY = helper;
    return helper;
}
"""

def notify_success_js(tool_name: str) -> str:
    return (
        f"""
(status, outputValue) => {{
{NOTIFICATION_HELPER_JS}
    const helper = vcEnsureNotify();
    helper.notify({json.dumps(tool_name)}, status, outputValue);
}}
"""
    )


def _ok(message: str) -> str:
    return f"[OK] {message}"


def _format_exception(exc: Exception) -> str:
    if isinstance(exc, VoiceCloneError):
        return f"[ERREUR] {exc}"
    return "[ERREUR] Une erreur inattendue est arrivee:\n" + traceback.format_exc()


def _safe_call(label: str, fn, *args, **kwargs):
    try:
        return TASK_GATE.run(label, fn, *args, **kwargs)
    except Exception as exc:  # Gradio should show clear messages, not raw crashes.
        return _format_exception(exc)


def bootstrap_runtime() -> str:
    try:
        paths = ensure_runtime_layout(SETTINGS)
        ffmpeg_path = ensure_ffmpeg_available(SETTINGS)
        return _ok(f"Structure prete dans {paths.root}\nFFmpeg: {ffmpeg_path}")
    except Exception as exc:
        return _format_exception(exc)


def refresh_models_choices() -> tuple[gr.Dropdown, str]:
    try:
        records = discover_voice_models(SETTINGS)
        choices = [
            (f"{record.actor_name} | {Path(record.model_path).name}", record.model_path)
            for record in records
        ]
        status = _ok(f"{len(records)} modele(s) trouve(s).")
        return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None), status
    except Exception as exc:
        return gr.Dropdown(choices=[], value=None), _format_exception(exc)


def show_voice_bank() -> str:
    try:
        discover_voice_models(SETTINGS)
        bank = load_voice_bank(SETTINGS)
        return json.dumps(bank, indent=2, ensure_ascii=False)
    except Exception as exc:
        return _format_exception(exc)


def run_cleaning(actor_name: str, files: list[str] | None, split_only: bool = False) -> tuple[str, str | None]:
    def _work() -> tuple[str, str | None]:
        cleaned_dir = clean_dataset(SETTINGS, actor_name, files or [], split_only=bool(split_only))
        mode = "Split Only - sans Demucs" if split_only else "Demucs + slicing"
        return _ok(f"Dataset prepare ({mode}): {cleaned_dir}"), str(cleaned_dir)

    result = _safe_call("nettoyage audio", _work)
    if isinstance(result, tuple):
        return result
    return result, None


def run_clean_only(file_path: str | None, quality: int = 4) -> tuple[str, str | None, str | None]:
    def _work() -> tuple[str, str | None, str | None]:
        if not file_path:
            raise VoiceCloneError("Ajoute un fichier audio a nettoyer.")
        output_path = clean_single_file(SETTINGS, file_path, quality=int(quality))
        return _ok(f"Nettoyage termine: {output_path}"), str(output_path), str(output_path)

    result = _safe_call("nettoyage seul", _work)
    if isinstance(result, tuple):
        return result
    return result, None, None


def run_clean_only_with_progress(file_path: str | None, quality: int):
    yield (
        "[INFO] Nettoyage en cours. Cela peut prendre un moment.",
        None,
        None,
        gr.update(value=BUSY_HTML.format(message="Demucs - Isolation vocale en cours..."), visible=True),
    )
    status, output_path, audio_path = run_clean_only(file_path, quality)
    yield status, output_path, audio_path, gr.update(visible=False)


def run_deepfilter_with_progress(file_path: str | None, strength: float):
    yield (
        "[INFO] DeepFilterNet en cours. Traitement local sur CPU.",
        None,
        None,
        gr.update(value=BUSY_HTML.format(message="DeepFilterNet - Denoise vocal en cours..."), visible=True),
    )

    def _work() -> tuple[str, str | None, str | None]:
        if not file_path:
            raise VoiceCloneError("Ajoute un fichier audio a traiter.")
        output_path = run_deepfilter_denoise(SETTINGS, file_path, strength=float(strength))
        return _ok(f"DeepFilterNet termine: {output_path}"), str(output_path), str(output_path)

    result = _safe_call("DeepFilterNet", _work)
    if isinstance(result, tuple):
        yield (*result, gr.update(visible=False))
    else:
        yield result, None, None, gr.update(visible=False)


def run_uvr_with_progress(file_path: str | None, model_label: str, quality: int):
    yield (
        "[INFO] UVR en cours. Le premier lancement peut telecharger le modele choisi.",
        None,
        None,
        gr.update(value=BUSY_HTML.format(message="UVR - Isolation vocale alternative en cours..."), visible=True),
    )

    def _work() -> tuple[str, str | None, str | None]:
        if not file_path:
            raise VoiceCloneError("Ajoute un fichier audio a isoler.")
        output_path = run_uvr_isolation(
            SETTINGS,
            file_path,
            model_label=model_label,
            quality=int(quality),
        )
        return _ok(f"UVR termine: {output_path}"), str(output_path), str(output_path)

    result = _safe_call("UVR", _work)
    if isinstance(result, tuple):
        yield (*result, gr.update(visible=False))
    else:
        yield result, None, None, gr.update(visible=False)


def run_voicefixer_with_progress(file_path: str | None, mode_label: str):
    yield (
        "[INFO] VoiceFixer en cours. Traitement local sur CPU.",
        None,
        None,
        gr.update(value=BUSY_HTML.format(message="VoiceFixer - Restauration vocale en cours..."), visible=True),
    )

    def _work() -> tuple[str, str | None, str | None]:
        if not file_path:
            raise VoiceCloneError("Ajoute un fichier audio a restaurer.")
        output_path = run_voicefixer_restoration(SETTINGS, file_path, mode_label=mode_label)
        return _ok(f"VoiceFixer termine: {output_path}"), str(output_path), str(output_path)

    result = _safe_call("VoiceFixer", _work)
    if isinstance(result, tuple):
        yield (*result, gr.update(visible=False))
    else:
        yield result, None, None, gr.update(visible=False)


def run_resemble_restoration(file_path: str | None, solver: str, quality: int):
    yield "[INFO] Resemble Enhance en cours sur CPU. La premiere execution peut etre longue.", None, None

    def _work() -> tuple[str, str | None, str | None]:
        if not file_path:
            raise VoiceCloneError("Ajoute un fichier WAV a restaurer.")
        output_path = restore_audio(
            SETTINGS,
            input_path=Path(file_path),
            solver=solver,
            quality=int(quality),
        )
        return _ok(f"Resemble Enhance termine: {output_path}"), str(output_path), str(output_path)

    result = _safe_call("restauration Resemble Enhance", _work)
    if isinstance(result, tuple):
        yield result
    else:
        yield result, None, None


def run_resemble_restoration_with_progress(file_path: str | None, solver: str, quality: int):
    yield (
        "[INFO] Resemble Enhance en cours sur CPU. La premiere execution peut etre longue.",
        None,
        None,
        gr.update(value=BUSY_HTML.format(message="Resemble Enhance en cours..."), visible=True),
    )

    def _work() -> tuple[str, str | None, str | None]:
        if not file_path:
            raise VoiceCloneError("Ajoute un fichier WAV a restaurer.")
        output_path = restore_audio(
            SETTINGS,
            input_path=Path(file_path),
            solver=solver,
            quality=int(quality),
        )
        return _ok(f"Resemble Enhance termine: {output_path}"), str(output_path), str(output_path)

    result = _safe_call("restauration Resemble Enhance", _work)
    if isinstance(result, tuple):
        yield (*result, gr.update(visible=False))
    else:
        yield result, None, None, gr.update(visible=False)


def run_zip_upload(actor_name: str) -> str:
    def _work() -> str:
        paths = ensure_runtime_layout(SETTINGS)
        zip_path = zip_cleaned_dataset(SETTINGS, actor_name)
        client = GoogleDriveClient.from_settings(SETTINGS)
        uploaded = client.upload_file(zip_path, SETTINGS.google_drive.upload_folder_id)
        return _ok(
            "Zip cree et uploade: "
            f"{zip_path.name} -> Google Drive id={uploaded.get('id', 'inconnu')}"
        )

    return _safe_call("upload Google Drive", _work)


def run_drive_test() -> str:
    def _work() -> str:
        client = GoogleDriveClient.from_settings(SETTINGS)
        return _ok("Connexion Google Drive validee.\n" + client.test_connection())

    return _safe_call("test Google Drive", _work)


def run_drive_watch(actor_name: str, timeout_minutes: int, interval_seconds: int) -> str:
    def _work() -> str:
        client = GoogleDriveClient.from_settings(SETTINGS)
        downloaded = client.wait_and_download_model_pair(
            actor_name=actor_name,
            destination_dir=ensure_runtime_layout(SETTINGS).models_bank,
            timeout_minutes=int(timeout_minutes),
            interval_seconds=int(interval_seconds),
        )
        discover_voice_models(SETTINGS)
        names = ", ".join(path.name for path in downloaded)
        return _ok(f"Modele recupere depuis Drive: {names}")

    return _safe_call("surveillance Google Drive", _work)


def archive_choices():
    folder = SETTINGS.root_path / "archives_drive" / "snapshots"
    return sorted((p.name for p in folder.glob("*.json") if not p.name.endswith(".removed.json")), reverse=True)


def run_archive_drive():
    def work():
        archive = DriveArchive(SETTINGS, GoogleDriveClient.from_settings(SETTINGS))
        path = archive.snapshot()
        _, data = archive.load(path)
        return _ok(snapshot_summary(data) + f"\nArchive: {path}\nAucun fichier Drive supprime."), path.name
    result = _safe_call("archivage Drive", work)
    if isinstance(result, tuple):
        return result[0], gr.update(choices=archive_choices(), value=result[1])
    return result, gr.update(choices=archive_choices())


def run_release_drive(snapshot, confirmed_idle):
    def work():
        if not snapshot:
            raise VoiceCloneError("Cree et selectionne une archive verifiee d'abord.")
        archive = DriveArchive(SETTINGS, GoogleDriveClient.from_settings(SETTINGS))
        removed = archive.remove_archived(snapshot, bool(confirmed_idle))
        return _ok(f"{removed / 1e9:.3f} Go retires de Drive. Archives locales conservees; modeles utilisables localement.")
    return _safe_call("liberation Drive", work), gr.update(value=False)


def run_restore_archive(snapshot, actor_name):
    def work():
        if not snapshot:
            raise VoiceCloneError("Selectionne une archive locale.")
        archive = DriveArchive(SETTINGS, GoogleDriveClient.from_settings(SETTINGS))
        count = archive.restore_actor(snapshot, actor_name)
        return _ok(f"{count} fichier(s) verifies sur Drive. Dans Colab, choisis ce comedien, RUN_MODE=resume et un total d'epochs superieur au precedent.")
    return _safe_call("restauration Drive", work)


def run_conversion(
    model_path: str | None,
    guide_file: str | None,
    transpose: int,
    index_rate: float,
    protect: float,
) -> tuple[str, str | None]:
    def _work() -> tuple[str, str | None]:
        if not model_path:
            raise VoiceCloneError("Aucun modele .pth selectionne.")
        if not guide_file:
            raise VoiceCloneError("Aucun fichier guide fourni.")
        selected_model = Path(model_path)
        selected_index = find_index_for_model(SETTINGS, selected_model)
        output_path = convert_voice(
            SETTINGS,
            model_path=selected_model,
            guide_path=Path(guide_file),
            transpose=int(transpose),
            index_rate=float(index_rate),
            protect=float(protect),
        )
        index_name = selected_index.name if selected_index else "aucun index trouve"
        return (
            _ok(
                "Conversion terminee.\n"
                f"Output: {output_path}\n"
                f"Parametres: RMVPE, transpose={int(transpose)}, "
                f"index_rate={float(index_rate):.2f}, protect={float(protect):.2f}\n"
                f"Index utilise: {index_name}"
            ),
            str(output_path),
        )

    result = _safe_call("conversion RVC", _work)
    if isinstance(result, tuple):
        return result
    return result, None


def settings_as_json() -> str:
    return json.dumps(SETTINGS.raw, indent=2, ensure_ascii=False)


def _colab_link_html(url: str) -> str:
    safe_url = escape(url, quote=True)
    return (
        f'<a class="vc-colab-link" href="{safe_url}" target="_blank" '
        'rel="noopener noreferrer">Ouvrir Colab</a>'
    )


def _normalize_notebook_url(url: str) -> str:
    """Accept either a Colab URL or the shared Google Drive file URL."""
    parsed = urlsplit((url or "").strip())
    if parsed.scheme != "https":
        raise VoiceCloneError("Utilise un lien https Google Drive ou Google Colab.")

    notebook_id = ""
    if parsed.netloc == "colab.research.google.com" and parsed.path.startswith("/drive/"):
        notebook_id = parsed.path.removeprefix("/drive/").split("/", 1)[0]
    elif parsed.netloc in {"drive.google.com", "www.drive.google.com"}:
        marker = "/file/d/"
        if parsed.path.startswith(marker):
            notebook_id = parsed.path.removeprefix(marker).split("/", 1)[0]
        else:
            notebook_id = parse_qs(parsed.query).get("id", [""])[0]

    if not notebook_id or not all(char.isalnum() or char in "_-" for char in notebook_id):
        raise VoiceCloneError(
            "Colle le lien du fichier notebook sur Google Drive, par exemple "
            "https://drive.google.com/file/d/..."
        )
    return f"https://colab.research.google.com/drive/{notebook_id}"


def save_colab_url(url: str) -> tuple[str, str]:
    global COLAB_URL

    try:
        value = _normalize_notebook_url(url)
        raw = json.loads(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8-sig"))
        raw["colab_url"] = value
        temporary = DEFAULT_SETTINGS_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(DEFAULT_SETTINGS_PATH)
        SETTINGS.raw["colab_url"] = value
        COLAB_URL = value
        return _ok("Lien Drive converti et URL Colab enregistree."), _colab_link_html(value)
    except Exception as exc:
        return _format_exception(exc), _colab_link_html(COLAB_URL)


def show_main_page():
    return gr.update(visible=True), gr.update(visible=False)


def show_local_tools_page():
    return gr.update(visible=False), gr.update(visible=True)


def show_clear_cache_confirm():
    return gr.update(visible=True), ""


def hide_clear_cache_confirm():
    return gr.update(visible=False), "[INFO] Vidage de cache annule."


def run_clear_cache() -> tuple[gr.Group, str]:
    def _work() -> str:
        paths = ensure_runtime_layout(SETTINGS)
        targets = [
            paths.temp,
            paths.dataset_cleaned,
            paths.clean_only,
            paths.outputs,
        ]
        lines = ["Cache vide. Contenu supprime uniquement dans:"]
        for target in targets:
            deleted = _delete_folder_contents_safely(paths.root, target)
            lines.append(f"- {target}: {deleted} element(s)")
        ensure_runtime_layout(SETTINGS)
        return _ok("\n".join(lines))

    result = _safe_call("vidage de cache", _work)
    return gr.update(visible=False), result


def _delete_folder_contents_safely(root: Path, folder: Path) -> int:
    root_resolved = root.resolve()
    folder_resolved = folder.resolve()
    allowed_names = {
        "temp",
        "dataset_cleaned",
        "Nettoyage Seul",
        "outputs",
    }
    if folder.name not in allowed_names:
        raise VoiceCloneError(f"Dossier non autorise pour le vidage de cache: {folder}")
    if folder_resolved == root_resolved or not folder_resolved.is_relative_to(root_resolved):
        raise VoiceCloneError(f"Chemin refuse hors root VOICECLONE: {folder}")

    folder.mkdir(parents=True, exist_ok=True)
    deleted = 0
    for child in folder.iterdir():
        _delete_child_safely(folder_resolved, child)
        deleted += 1
    return deleted


def _delete_child_safely(parent_resolved: Path, child: Path) -> None:
    child_resolved = child.resolve()
    is_junction = getattr(child, "is_junction", lambda: False)()
    if child.is_symlink() or is_junction:
        if child.is_dir():
            child.rmdir()
        else:
            child.unlink()
        return

    if not child_resolved.is_relative_to(parent_resolved):
        raise VoiceCloneError(f"Suppression refusee hors dossier autorise: {child}")

    if child.is_dir():
        shutil.rmtree(child)
    else:
        child.unlink()


def build_ui(settings: Settings) -> gr.Blocks:
    # Populate the selector at startup so the normal conversion workflow does
    # not begin with an empty model field.
    initial_model_dropdown, initial_model_status = refresh_models_choices()

    with gr.Blocks(title="VOICECLONE-QC") as demo:
        with gr.Row(elem_classes=["vc-topbar"]):
            gr.HTML("<h1>VOICECLONE-QC</h1>")
            home_nav = gr.Button("Page principale", elem_classes=["vc-home-button"])
            local_tools_nav = gr.Button(
                "Fonctionnalites hors Colab",
                elem_classes=["vc-nav-button"],
            )
            clear_cache_nav = gr.Button(
                "Vider la cache",
                variant="stop",
                elem_classes=["vc-danger-button"],
            )
            gr.HTML(
                '<a class="vc-colab-config-link" href="#colab-url-config">'
                'Configurer URL Colab</a>'
            )
            colab_link = gr.HTML(_colab_link_html(COLAB_URL))

        startup_status = gr.Textbox(
            label="Etat runtime",
            value=bootstrap_runtime(),
            interactive=False,
            lines=3,
        )
        with gr.Accordion("Notebook Google Drive", open=False, elem_id="colab-url-config"):
            colab_url_input = gr.Textbox(
                label="Lien du fichier notebook sur Google Drive",
                value=COLAB_URL,
            )
            with gr.Row():
                save_colab_url_button = gr.Button("Enregistrer", variant="primary")
            colab_url_status = gr.Textbox(label="Etat URL Colab", interactive=False, lines=2)

        with gr.Group(visible=False, elem_classes=["vc-danger-panel"]) as clear_cache_confirm:
            gr.Markdown(
                "## Confirmation - vider la cache\n"
                "Cette action supprimera uniquement le contenu de `temp`, `dataset_cleaned`, "
                "`Nettoyage Seul` et `outputs`. Les modeles, la configuration et les runtimes "
                "ne seront pas touches."
            )
            with gr.Row():
                clear_cache_confirm_button = gr.Button("Oui, vider la cache", variant="stop")
                clear_cache_cancel_button = gr.Button("Annuler")
            clear_cache_status = gr.Textbox(label="Journal cache", lines=7, interactive=False)

        with gr.Column(visible=True) as main_page:
            gr.Markdown("Workflow local: nettoyage dataset, synchronisation Drive, conversion RVC.")

            gr.HTML(
                '<h2 class="vc-section-title">Nettoyage'
                '<span class="vc-help" data-help="Duree recommandee pour un modele RVC: minimum 10 a 15 minutes de voix propre, bon resultat autour de 25 a 45 minutes, idealement 45 a 75 minutes par comedien. Au-dela de 2 heures, le gain devient souvent faible. La qualite et la variete des sources comptent plus que la duree brute.">?</span>'
                '</h2>'
            )
            with gr.Group():
                actor_name = gr.Textbox(label="Nom du comedien", placeholder="Ex: Jean_Dupont")
                rough_files = gr.File(
                    label="Extraits rough",
                    file_count="multiple",
                    type="filepath",
                )
                split_only = gr.Checkbox(
                    label="Split Only - sauter Demucs",
                    value=False,
                    info=(
                        "Utilise ce mode seulement si tes sources sont deja propres: "
                        "voix seule, sans musique ni bruit de fond."
                    ),
                )
                clean_button = gr.Button("Nettoyer le dataset", variant="primary")
                clean_status = gr.Textbox(label="Journal", lines=7, interactive=False)
                cleaned_dir = gr.Textbox(label="Dossier nettoye", interactive=False)
                clean_done = clean_button.click(
                    run_cleaning,
                    inputs=[actor_name, rough_files, split_only],
                    outputs=[clean_status, cleaned_dir],
                )
                clean_done.success(
                    fn=None,
                    inputs=[clean_status, cleaned_dir],
                    js=notify_success_js("Nettoyage dataset"),
                    queue=False,
                )

            gr.Markdown("## Cloud")
            with gr.Group():
                drive_actor = gr.Textbox(label="Nom du comedien", placeholder="Meme nom que le dataset")
                with gr.Row():
                    drive_test_button = gr.Button("Tester Drive")
                    zip_upload_button = gr.Button("Zipper et uploader", variant="primary")
                    watch_button = gr.Button("Surveiller le retour modele")
                with gr.Row():
                    timeout_minutes = gr.Number(label="Timeout minutes", value=180, precision=0)
                    interval_seconds = gr.Number(label="Intervalle secondes", value=30, precision=0)
                drive_status = gr.Textbox(label="Journal Drive", lines=8, interactive=False)
                drive_test_button.click(run_drive_test, outputs=[drive_status])
                zip_upload_done = zip_upload_button.click(
                    run_zip_upload,
                    inputs=[drive_actor],
                    outputs=[drive_status],
                )
                zip_upload_done.success(
                    fn=None,
                    inputs=[drive_status],
                    js=notify_success_js("Upload Drive"),
                    queue=False,
                )
                watch_done = watch_button.click(
                    run_drive_watch,
                    inputs=[drive_actor, timeout_minutes, interval_seconds],
                    outputs=[drive_status],
                )
                watch_done.success(
                    fn=None,
                    inputs=[drive_status],
                    js=notify_success_js("Retour modele Drive"),
                    queue=False,
                )

                with gr.Accordion("Espace Drive et archives locales", open=False):
                    gr.Markdown(
                        "Archive les trois dossiers VOICECLONE sur ce disque, avec verification et une seule copie des fichiers identiques. "
                        "Les archives sont conservees dans `archives_drive` et protegees du bouton Vider la cache. "
                        "Apres un entrainement termine, arrete Colab, archive, puis libere Drive."
                    )
                    archive_button = gr.Button("1. Archiver Drive localement", variant="primary")
                    snapshots = gr.Dropdown(label="Archive locale", choices=archive_choices())
                    archive_status = gr.Textbox(label="Archivage et verification", lines=9, interactive=False)
                    release_confirm = gr.Checkbox(
                        label="Colab est arrete. Je confirme la suppression definitive des copies Drive de TOUS les fichiers de cette archive, apres verification locale.",
                        value=False,
                    )
                    release_button = gr.Button("2. Liberer Drive avec cette archive", variant="stop")
                    gr.Markdown("Pour continuer une voix plus tard : selectionne son archive et renseigne son nom dans Nom du comedien ci-dessus. Attends la fin de la restauration avant de lancer Colab en mode resume.")
                    restore_button = gr.Button("Remettre cette voix sur Drive pour reprendre l'entrainement")
                    archive_button.click(run_archive_drive, outputs=[archive_status, snapshots])
                    release_button.click(run_release_drive, inputs=[snapshots, release_confirm], outputs=[archive_status, release_confirm])
                    restore_button.click(run_restore_archive, inputs=[snapshots, drive_actor], outputs=[archive_status])

            gr.Markdown("## RVC")
            with gr.Group():
                with gr.Row():
                    refresh_button = gr.Button("Rafraichir les modeles")
                    model_dropdown = gr.Dropdown(
                        label="Modele RVC (.pth)",
                        choices=initial_model_dropdown.choices,
                        value=initial_model_dropdown.value,
                    )
                model_status = gr.Textbox(
                    label="Etat banque de voix",
                    value=initial_model_status,
                    interactive=False,
                )
                guide_file = gr.File(label="Fichier guide WAV", file_count="single", type="filepath")
                with gr.Row():
                    with gr.Column():
                        gr.HTML(
                            '<div class="vc-label">Transpose'
                            '<span class="vc-help" data-help="Change la hauteur de la voix en demi-tons. 0 garde la hauteur originale. Utilise +12 pour une octave plus haut, -12 pour une octave plus bas. Utile si la voix source et la voix cible ont des tessitures tres differentes.">?</span>'
                            '</div>'
                        )
                        transpose = gr.Slider(-24, 24, value=0, step=1, label=None)
                    with gr.Column():
                        gr.HTML(
                            '<div class="vc-label">Index Rate'
                            '<span class="vc-help" data-help="Dose l influence du fichier .index, donc la ressemblance avec la voix entrainee. Plus haut = plus proche du modele, mais parfois plus d artefacts. 0.6 a 0.8 est souvent un bon depart.">?</span>'
                            '</div>'
                        )
                        index_rate = gr.Slider(0, 1, value=0.75, step=0.05, label=None)
                    with gr.Column():
                        gr.HTML(
                            '<div class="vc-label">Protect'
                            '<span class="vc-help" data-help="Protege les consonnes et respirations contre les deformations liees a l index. Dans ce moteur, une valeur plus basse renforce cette protection. 0.5 la desactive. Compare 0.15, 0.25 et 0.33 selon la replique.">?</span>'
                            '</div>'
                        )
                        protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label=None)
                convert_button = gr.Button("Convertir en RMVPE", variant="primary")
                convert_status = gr.Textbox(label="Journal conversion", lines=7, interactive=False)
                converted_audio = gr.Audio(label="Output WAV", type="filepath")

                refresh_button.click(refresh_models_choices, outputs=[model_dropdown, model_status])
                convert_done = convert_button.click(
                    run_conversion,
                    inputs=[model_dropdown, guide_file, transpose, index_rate, protect],
                    outputs=[convert_status, converted_audio],
                )
                convert_done.success(
                    fn=None,
                    inputs=[convert_status, converted_audio],
                    js=notify_success_js("Conversion RVC"),
                    queue=False,
                )

            gr.Markdown("## Voix")
            with gr.Group():
                bank_button = gr.Button("Scanner la banque")
                bank_json = gr.Code(label="voice_bank.json", language="json", value=show_voice_bank)
                bank_button.click(show_voice_bank, outputs=[bank_json])

            gr.Markdown("## Config")
            with gr.Group():
                gr.Textbox(label="Fichier settings", value=str(DEFAULT_SETTINGS_PATH), interactive=False)
                gr.Code(label="Configuration active", language="json", value=settings_as_json)
                recheck_button = gr.Button("Revalider la structure")
                recheck_button.click(bootstrap_runtime, outputs=[startup_status])

        with gr.Column(visible=False) as local_tools_page:
            gr.Markdown("## Fonctionnalites hors Colab")

            with gr.Group():
                gr.Markdown("## Demucs - Isolation vocale")
                clean_only_file = gr.File(
                    label="Fichier audio a nettoyer",
                    file_count="single",
                    type="filepath",
                )
                gr.HTML(
                    '<div class="vc-label">Qualite / lenteur Demucs'
                    '<span class="vc-help" data-help="Controle le nombre de passes Demucs. Plus haut = separation potentiellement plus propre, mais beaucoup plus lente. 4 est un bon depart; 8 a 10 pour les fichiers importants quand le temps ne compte pas.">?</span>'
                    '</div>'
                )
                clean_only_quality = gr.Slider(
                    1,
                    10,
                    value=4,
                    step=1,
                    label=None,
                )
                clean_only_button = gr.Button("Isoler la voix avec Demucs", variant="primary")
                clean_only_status = gr.Textbox(label="Journal Demucs", lines=7, interactive=False)
                clean_only_path = gr.Textbox(label="Fichier nettoye", interactive=False)
                clean_only_audio = gr.Audio(label="Preview WAV nettoye", type="filepath")
                clean_only_busy = gr.HTML(visible=False)
                clean_only_done = clean_only_button.click(
                    run_clean_only_with_progress,
                    inputs=[clean_only_file, clean_only_quality],
                    outputs=[clean_only_status, clean_only_path, clean_only_audio, clean_only_busy],
                )
                clean_only_done.success(
                    fn=None,
                    inputs=[clean_only_status, clean_only_path],
                    js=notify_success_js("Demucs"),
                    queue=False,
                )

            with gr.Group():
                gr.Markdown("## DeepFilterNet - Denoise vocal")
                deepfilter_file = gr.File(
                    label="Fichier audio a denoiser",
                    file_count="single",
                    type="filepath",
                )
                gr.HTML(
                    '<div class="vc-label">Force denoise DeepFilterNet'
                    '<span class="vc-help" data-help="Limite d attenuation du bruit en dB. Plus haut = nettoyage plus agressif, avec plus de risque d artefacts ou de voix amincie. Essaie 40 a 60 avant de monter davantage.">?</span>'
                    '</div>'
                )
                deepfilter_strength = gr.Slider(
                    1,
                    100,
                    value=60,
                    step=1,
                    label=None,
                )
                deepfilter_button = gr.Button("Denoiser avec DeepFilterNet", variant="primary")
                deepfilter_status = gr.Textbox(label="Journal DeepFilterNet", lines=7, interactive=False)
                deepfilter_path = gr.Textbox(label="Fichier denoise", interactive=False)
                deepfilter_audio = gr.Audio(label="Output DeepFilterNet WAV", type="filepath")
                deepfilter_busy = gr.HTML(visible=False)
                deepfilter_done = deepfilter_button.click(
                    run_deepfilter_with_progress,
                    inputs=[deepfilter_file, deepfilter_strength],
                    outputs=[deepfilter_status, deepfilter_path, deepfilter_audio, deepfilter_busy],
                )
                deepfilter_done.success(
                    fn=None,
                    inputs=[deepfilter_status, deepfilter_path],
                    js=notify_success_js("DeepFilterNet"),
                    queue=False,
                )

            with gr.Group():
                gr.Markdown("## UVR - Isolation vocale alternative")
                uvr_file = gr.File(
                    label="Fichier audio a isoler",
                    file_count="single",
                    type="filepath",
                )
                gr.HTML(
                    '<div class="vc-label">Modele UVR'
                    '<span class="vc-help" data-help="Choisit la famille de modele d isolation. RoFormer est le choix qualite par defaut, BS-RoFormer est une bonne alternative si le resultat sonne moins naturel, MDX est plus leger.">?</span>'
                    '</div>'
                )
                uvr_model = gr.Dropdown(
                    label=None,
                    choices=list(UVR_MODELS.keys()),
                    value="RoFormer vocal - qualite",
                )
                gr.HTML(
                    '<div class="vc-label">Qualite / lenteur UVR'
                    '<span class="vc-help" data-help="Augmente le chevauchement d analyse du separateur. Plus haut = isolation parfois plus stable, mais plus lente et plus lourde. 4 est raisonnable; 7 a 10 pour tenter de sauver un fichier difficile.">?</span>'
                    '</div>'
                )
                uvr_quality = gr.Slider(
                    1,
                    10,
                    value=4,
                    step=1,
                    label=None,
                )
                uvr_button = gr.Button("Isoler la voix avec UVR", variant="primary")
                uvr_status = gr.Textbox(label="Journal UVR", lines=7, interactive=False)
                uvr_path = gr.Textbox(label="Fichier voix isolee", interactive=False)
                uvr_audio = gr.Audio(label="Output UVR WAV", type="filepath")
                uvr_busy = gr.HTML(visible=False)
                uvr_done = uvr_button.click(
                    run_uvr_with_progress,
                    inputs=[uvr_file, uvr_model, uvr_quality],
                    outputs=[uvr_status, uvr_path, uvr_audio, uvr_busy],
                )
                uvr_done.success(
                    fn=None,
                    inputs=[uvr_status, uvr_path],
                    js=notify_success_js("UVR"),
                    queue=False,
                )

            with gr.Group():
                gr.Markdown("## Resemble Enhance - Restauration haute fidelite")
                restoration_file = gr.File(
                    label="Fichier WAV a restaurer",
                    file_count="single",
                    type="filepath",
                )
                with gr.Row():
                    with gr.Column():
                        gr.HTML(
                            '<div class="vc-label">Solver'
                            '<span class="vc-help" data-help="Methode mathematique utilisee par Resemble Enhance pour reconstruire le signal. midpoint est le meilleur choix par defaut. euler peut etre un peu plus rapide mais parfois moins propre. rk4 peut etre plus stable sur certains fichiers, mais plus lent.">?</span>'
                            '</div>'
                        )
                        restoration_solver = gr.Dropdown(
                            label=None,
                            choices=["midpoint", "euler", "rk4"],
                            value="midpoint",
                        )
                    with gr.Column():
                        gr.HTML(
                            '<div class="vc-label">Qualite reconstruction'
                            '<span class="vc-help" data-help="Controle le nombre d etapes de reconstruction de Resemble Enhance. Plus haut = restauration potentiellement plus fine, mais beaucoup plus lente sur CPU. 4 est un bon point de depart; 6 a 8 pour les fichiers importants quand le temps ne compte pas.">?</span>'
                            '</div>'
                        )
                        restoration_quality = gr.Slider(
                            1,
                            8,
                            value=4,
                            step=1,
                            label=None,
                        )
                restoration_button = gr.Button("Restaurer avec Resemble Enhance", variant="primary")
                restoration_status = gr.Textbox(label="Journal Resemble Enhance", lines=7, interactive=False)
                restoration_path = gr.Textbox(label="Fichier Resemble Enhance", interactive=False)
                restoration_audio = gr.Audio(label="Output Resemble Enhance WAV", type="filepath")
                restoration_busy = gr.HTML(visible=False)
                restoration_done = restoration_button.click(
                    run_resemble_restoration_with_progress,
                    inputs=[restoration_file, restoration_solver, restoration_quality],
                    outputs=[restoration_status, restoration_path, restoration_audio, restoration_busy],
                )
                restoration_done.success(
                    fn=None,
                    inputs=[restoration_status, restoration_path],
                    js=notify_success_js("Resemble Enhance"),
                    queue=False,
                )

            with gr.Group():
                gr.Markdown("## VoiceFixer - Restauration vocale")
                voicefixer_file = gr.File(
                    label="Fichier audio a restaurer",
                    file_count="single",
                    type="filepath",
                )
                gr.HTML(
                    '<div class="vc-label">Mode VoiceFixer'
                    '<span class="vc-help" data-help="Mode 0 est le choix normal. Mode 1 ajoute un preprocessing spectral utile si les hautes frequences sont sales. Mode 2 peut aider sur une voix tres degradee, mais peut aussi changer davantage le timbre.">?</span>'
                    '</div>'
                )
                voicefixer_mode = gr.Dropdown(
                    label=None,
                    choices=list(VOICEFIXER_MODES.keys()),
                    value="Mode 0 - restauration normale",
                )
                voicefixer_button = gr.Button("Restaurer avec VoiceFixer", variant="primary")
                voicefixer_status = gr.Textbox(label="Journal VoiceFixer", lines=7, interactive=False)
                voicefixer_path = gr.Textbox(label="Fichier VoiceFixer", interactive=False)
                voicefixer_audio = gr.Audio(label="Output VoiceFixer WAV", type="filepath")
                voicefixer_busy = gr.HTML(visible=False)
                voicefixer_done = voicefixer_button.click(
                    run_voicefixer_with_progress,
                    inputs=[voicefixer_file, voicefixer_mode],
                    outputs=[voicefixer_status, voicefixer_path, voicefixer_audio, voicefixer_busy],
                )
                voicefixer_done.success(
                    fn=None,
                    inputs=[voicefixer_status, voicefixer_path],
                    js=notify_success_js("VoiceFixer"),
                    queue=False,
                )

        save_colab_url_button.click(
            save_colab_url,
            inputs=[colab_url_input],
            outputs=[colab_url_status, colab_link],
            queue=False,
        )
        home_nav.click(show_main_page, outputs=[main_page, local_tools_page])
        local_tools_nav.click(show_local_tools_page, outputs=[main_page, local_tools_page])
        clear_cache_nav.click(
            show_clear_cache_confirm,
            outputs=[clear_cache_confirm, clear_cache_status],
        )
        clear_cache_cancel_button.click(
            hide_clear_cache_confirm,
            outputs=[clear_cache_confirm, clear_cache_status],
        )
        clear_cache_confirm_button.click(
            run_clear_cache,
            outputs=[clear_cache_confirm, clear_cache_status],
        )

    demo.queue(default_concurrency_limit=1)
    return demo


if __name__ == "__main__":
    ui = build_ui(SETTINGS)
    runtime_paths = ensure_runtime_layout(SETTINGS)
    ui.launch(
        server_name=SETTINGS.gradio.host,
        server_port=SETTINGS.gradio.port,
        show_error=False,
        # Only output folders need to be exposed to Gradio's file server.  The
        # project root also contains OAuth credentials and runtime settings.
        allowed_paths=[
            str(runtime_paths.outputs),
            str(runtime_paths.clean_only),
        ],
        css=APP_CSS,
    )

