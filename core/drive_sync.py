from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

from .config import Settings, ensure_runtime_layout, resolve_runtime_file
from .errors import ConfigurationError, DependencyMissing, VoiceCloneError
from .naming import safe_name


DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def zip_cleaned_dataset(settings: Settings, actor_name: str) -> Path:
    actor_slug = safe_name(actor_name)
    paths = ensure_runtime_layout(settings)
    source_dir = paths.dataset_cleaned / actor_slug
    if not source_dir.exists():
        raise VoiceCloneError(f"Dataset nettoye introuvable: {source_dir}")
    if not any(source_dir.glob("*.wav")):
        raise VoiceCloneError(f"Aucun WAV trouve dans: {source_dir}")

    zip_path = paths.staging / f"{actor_slug}_dataset_cleaned.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(source_dir.glob("*.wav")):
            if file_path.is_file():
                archive.write(file_path, Path("dataset") / file_path.name)

    return zip_path


def _load_credentials(settings: Settings):
    auth_mode = (settings.google_drive.auth_mode or "oauth").lower()
    if auth_mode == "oauth":
        return _load_oauth_credentials(settings)
    if auth_mode == "service_account":
        return _load_service_account_credentials(settings)
    raise ConfigurationError("google_drive.auth_mode doit etre 'oauth' ou 'service_account'.")


def _load_oauth_credentials(settings: Settings):
    try:
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise DependencyMissing(
            "OAuth Google Drive n'est pas installe. Lance: "
            ".\\.venv\\Scripts\\python.exe -m pip install google-auth-oauthlib"
        ) from exc

    client_path = resolve_runtime_file(settings, settings.google_drive.oauth_client_path)
    token_path = resolve_runtime_file(settings, settings.google_drive.oauth_token_path)
    if not client_path.exists():
        raise ConfigurationError(
            "Client OAuth introuvable: "
            f"{client_path}. Telecharge le JSON OAuth Google Cloud et renomme-le oauth_client.json."
        )

    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), DRIVE_SCOPES)

    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except RefreshError:
            try:
                token_path.unlink()
            except OSError:
                pass
            credentials = None

    if not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_path), DRIVE_SCOPES)
        credentials = flow.run_local_server(port=0)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def _load_service_account_credentials(settings: Settings):
    try:
        from google.oauth2 import service_account
    except ImportError as exc:
        raise DependencyMissing("google-auth est requis pour le Service Account Google.") from exc

    service_account_path = resolve_runtime_file(settings, settings.google_drive.service_account_path)
    if not service_account_path.exists():
        raise ConfigurationError(
            "Cle service_account.json introuvable: "
            f"{service_account_path}. Place la cle dans ce chemin ou modifie settings.json."
        )
    return service_account.Credentials.from_service_account_file(
        str(service_account_path),
        scopes=DRIVE_SCOPES,
    )


class GoogleDriveClient:
    def __init__(
        self,
        service,
        upload_folder_id: str,
        return_folder_id: str,
        return_folder_name: str,
    ):
        self.service = service
        self.upload_folder_id = upload_folder_id
        self.return_folder_id = return_folder_id
        self.return_folder_name = return_folder_name

    @classmethod
    def from_settings(cls, settings: Settings) -> "GoogleDriveClient":
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise DependencyMissing(
                "Les dependances Google Drive ne sont pas installees. "
                "Lance scripts/setup_windows.bat avant d'utiliser le Cloud."
            ) from exc

        credentials = _load_credentials(settings)
        if not settings.google_drive.upload_folder_id:
            raise ConfigurationError("google_drive.upload_folder_id est vide dans settings.json.")
        if not settings.google_drive.return_folder_id and not settings.google_drive.return_folder_name:
            raise ConfigurationError(
                "Configure google_drive.return_folder_id ou google_drive.return_folder_name."
            )

        return cls(
            build("drive", "v3", credentials=credentials, cache_discovery=False),
            settings.google_drive.upload_folder_id,
            settings.google_drive.return_folder_id,
            settings.google_drive.return_folder_name,
        )

    def upload_file(self, file_path: Path, folder_id: str) -> dict:
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as exc:
            raise DependencyMissing("google-api-python-client est requis pour uploader sur Drive.") from exc

        metadata = {"name": file_path.name, "parents": [folder_id]}
        media = MediaFileUpload(str(file_path), resumable=True)
        return (
            self.service.files()
            .create(
                body=metadata,
                media_body=media,
                fields="id,name,webViewLink",
                supportsAllDrives=True,
            )
            .execute()
        )

    def test_connection(self) -> str:
        upload_folder = (
            self.service.files()
            .get(
                fileId=self.upload_folder_id,
                fields="id,name,mimeType",
                supportsAllDrives=True,
            )
            .execute()
        )
        return_folder_id = self._get_return_folder_id()
        return_folder = (
            self.service.files()
            .get(
                fileId=return_folder_id,
                fields="id,name,mimeType",
                supportsAllDrives=True,
            )
            .execute()
        )
        return (
            f"Upload OK: {upload_folder.get('name')} ({upload_folder.get('id')})\n"
            f"Retour OK: {return_folder.get('name')} ({return_folder.get('id')})"
        )

    def wait_and_download_model_pair(
        self,
        actor_name: str,
        destination_dir: Path,
        timeout_minutes: int,
        interval_seconds: int,
    ) -> list[Path]:
        actor_slug = safe_name(actor_name)
        deadline = time.monotonic() + max(timeout_minutes, 1) * 60

        while time.monotonic() < deadline:
            pth = self._find_first_file(actor_slug, ".pth")
            index = self._find_first_file(actor_slug, ".index")
            if pth and index:
                destination_dir.mkdir(parents=True, exist_ok=True)
                return [
                    self.download_file(pth["id"], destination_dir / pth["name"]),
                    self.download_file(index["id"], destination_dir / index["name"]),
                ]
            time.sleep(max(interval_seconds, 10))

        raise VoiceCloneError(
            f"Aucun couple .pth/.index trouve pour {actor_slug} apres {timeout_minutes} minutes."
        )

    def download_file(self, file_id: str, destination: Path) -> Path:
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as exc:
            raise DependencyMissing("google-api-python-client est requis pour telecharger depuis Drive.") from exc

        request = self.service.files().get_media(fileId=file_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with io.FileIO(destination, "wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return destination

    def _find_first_file(self, actor_slug: str, extension: str) -> dict | None:
        folder_id = self._escape_query_literal(self._get_return_folder_id())
        actor_query = self._escape_query_literal(actor_slug)
        ext_query = self._escape_query_literal(extension)
        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{actor_query}' and name contains '{ext_query}'"
        )
        response = (
            self.service.files()
            .list(
                q=query,
                fields="files(id,name,modifiedTime)",
                orderBy="modifiedTime desc",
                pageSize=10,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        files = response.get("files", [])
        return files[0] if files else None

    def _get_return_folder_id(self) -> str:
        if self.return_folder_id:
            return self.return_folder_id

        folder_name = self._escape_query_literal(self.return_folder_name or "RVC_Output")
        query = (
            "mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false and name='{folder_name}'"
        )
        response = (
            self.service.files()
            .list(
                q=query,
                fields="files(id,name,modifiedTime)",
                orderBy="modifiedTime desc",
                pageSize=10,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        files = response.get("files", [])
        if not files:
            raise ConfigurationError(
                f"Dossier Drive '{self.return_folder_name}' introuvable. "
                "Cree-le, partage-le avec le compte Google connecte, ou renseigne return_folder_id."
            )
        self.return_folder_id = files[0]["id"]
        return self.return_folder_id

    @staticmethod
    def _escape_query_literal(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")
