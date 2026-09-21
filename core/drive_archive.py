"""Verified local archives of the three VOICECLONE Drive folders.

Cloud removal is a separate, explicit operation. Only unchanged files from a
complete, reverified snapshot can be removed; folders are never deleted.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .errors import VoiceCloneError

FOLDER = "application/vnd.google-apps.folder"
FIELDS = "id,name,mimeType,parents,size,md5Checksum,modifiedTime,version,trashed"
ROOT_NAMES = ("VOICECLONE_Datasets", "RVC_Output", "VOICECLONE_Training")


def digest(path: Path, algorithm: str = "md5") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def list_files(service, query: str) -> list[dict]:
    result, page = [], None
    while True:
        data = service.files().list(
            q=query, pageSize=1000, pageToken=page,
            fields=f"nextPageToken,incompleteSearch,files({FIELDS})",
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute(num_retries=2)
        if data.get("incompleteSearch"):
            raise VoiceCloneError("Inventaire Drive incomplet; operation interrompue.")
        result.extend(data.get("files", []))
        page = data.get("nextPageToken")
        if not page:
            return result


def same_file(left: dict, right: dict) -> bool:
    return all(left.get(k) == right.get(k) for k in
               ("id", "name", "parents", "size", "md5Checksum", "version", "modifiedTime"))


def safe_component(name: str) -> str:
    # These names are restored on Windows and in Colab. Refuse ambiguous paths.
    if (not name or name in {".", ".."} or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name.endswith((".", " "))
            or name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}):
        raise VoiceCloneError(f"Nom de fichier non portable: {name!r}")
    return name


class DriveArchive:
    def __init__(self, settings, client):
        self.settings, self.client = settings, client
        self.service = client.service
        self.root = settings.root_path / "archives_drive"

    def roots(self) -> list[dict]:
        roots = []
        configured = (self.client.upload_folder_id, self.client._get_return_folder_id(), "")
        for name, fid in zip(ROOT_NAMES, configured):
            if fid:
                item = self.service.files().get(fileId=fid, fields=FIELDS, supportsAllDrives=True).execute()
                if item["mimeType"] != FOLDER or item.get("trashed"):
                    raise VoiceCloneError(f"Dossier Drive invalide: {name}")
            else:
                matches = list_files(self.service, f"'root' in parents and trashed=false and mimeType='{FOLDER}' and name='{name}'")
                if len(matches) > 1:
                    raise VoiceCloneError(f"Plusieurs dossiers {name}: resoudre l'ambiguite dans Drive.")
                if not matches:
                    continue
                item = matches[0]
            roots.append({"id": item["id"], "name": name})
        if len({r['id'] for r in roots}) != len(roots):
            raise VoiceCloneError("Les dossiers Drive configures doivent etre distincts.")
        return roots

    def inventory(self, roots=None) -> list[dict]:
        files, seen = [], set()
        pending = [(r["id"], [r["name"]]) for r in (self.roots() if roots is None else roots)]
        while pending:
            fid, parts = pending.pop()
            if fid in seen:
                raise VoiceCloneError("Structure Drive ambigue.")
            seen.add(fid)
            for item in list_files(self.service, f"'{fid}' in parents and trashed=false"):
                child_parts = parts + [safe_component(item["name"])]
                if item["mimeType"] == FOLDER:
                    pending.append((item["id"], child_parts))
                else:
                    if not item.get("md5Checksum") or "size" not in item:
                        raise VoiceCloneError(f"Fichier non archivable automatiquement: {'/'.join(child_parts)}")
                    if not re.fullmatch(r"[0-9a-f]{32}", item['md5Checksum']) or int(item['size']) < 0:
                        raise VoiceCloneError("Empreinte ou taille Drive invalide.")
                    item["parts"] = child_parts
                    files.append(item)
        return sorted(files, key=lambda f: f["id"])

    def snapshot(self, progress=lambda message: None) -> Path:
        roots = self.roots()
        entries = self.inventory(roots)
        if not entries:
            raise VoiceCloneError("Aucun fichier VOICECLONE a archiver sur Drive.")
        objects = self.root / "objects"
        objects.mkdir(parents=True, exist_ok=True)
        unique = {f"{f['md5Checksum']}_{f['size']}": int(f["size"]) for f in entries}
        needed = sum(size for name, size in unique.items() if not (objects / name).exists())
        if shutil.disk_usage(self.root).free < needed + 256 * 1024 * 1024:
            raise VoiceCloneError("Espace local insuffisant pour une archive complete.")
        verified = {}
        for number, entry in enumerate(entries, 1):
            progress(f"Archivage {number}/{len(entries)} : {'/'.join(entry['parts'])}")
            blob_name = f"{entry['md5Checksum']}_{entry['size']}"
            blob = objects / blob_name
            if blob_name not in verified:
                if not (blob.is_file() and blob.stat().st_size == int(entry["size"]) and digest(blob) == entry["md5Checksum"]):
                    temporary = objects / f"{blob_name}.{uuid4().hex}.part"
                    try:
                        local = self.settings.root_path / "models_bank" / entry["name"]
                        if local.is_file() and local.stat().st_size == int(entry["size"]) and digest(local) == entry["md5Checksum"]:
                            shutil.copy2(local, temporary)
                        else:
                            self.client.download_file(entry["id"], temporary)
                        if temporary.stat().st_size != int(entry["size"]) or digest(temporary) != entry["md5Checksum"]:
                            raise VoiceCloneError("Verification du telechargement echouee; rien ne sera supprime.")
                        temporary.replace(blob)
                    finally:
                        temporary.unlink(missing_ok=True)
                verified[blob_name] = digest(blob, "sha256")
            entry.update(blob=blob_name, sha256=verified[blob_name])
        current = self.inventory(roots)
        if len(current) != len(entries) or any(not same_file(a, b) for a, b in zip(entries, current)):
            raise VoiceCloneError("Drive a change pendant l'archivage. Relancer apres l'arret de Colab; les copies deja faites seront reutilisees.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        path = self.root / "snapshots" / f"{stamp}.json"
        atomic_json(path, {"format": 1, "created": stamp, "roots": roots, "files": entries})
        return path

    def load(self, snapshot: str | Path) -> tuple[Path, dict]:
        directory = (self.root / "snapshots").resolve()
        path = (directory / Path(snapshot).name).resolve()
        if path.parent != directory or path.suffix != ".json":
            raise VoiceCloneError("Archive locale invalide.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != 1 or not data.get("files"):
            raise VoiceCloneError("Archive incomplete ou inconnue.")
        return path, data

    def verify(self, snapshot: str | Path) -> dict:
        _, data = self.load(snapshot)
        checked = set()
        objects = (self.root / "objects").resolve()
        for entry in data["files"]:
            blob = (objects / entry["blob"]).resolve()
            if blob.parent != objects:
                raise VoiceCloneError("Chemin d'archive invalide.")
            if blob not in checked:
                if (not blob.is_file() or blob.stat().st_size != int(entry["size"])
                        or digest(blob, "sha256") != entry["sha256"] or digest(blob) != entry["md5Checksum"]):
                    raise VoiceCloneError(f"Archive locale endommagee: {entry['name']}. Aucune suppression autorisee.")
                checked.add(blob)
        return data

    def remove_archived(self, snapshot: str | Path, confirmed_idle: bool = False) -> int:
        if not confirmed_idle:
            raise VoiceCloneError("Confirme que Colab est arrete et que tu veux supprimer definitivement les copies Drive archivees.")
        path, _ = self.load(snapshot)
        data = self.verify(snapshot)
        if sorted(self.roots(), key=lambda r: r['id']) != sorted(data['roots'], key=lambda r: r['id']):
            raise VoiceCloneError("Les dossiers Drive ont change depuis l'archive.")
        expected = {f['id']: f for f in data['files']}
        current = self.inventory(data['roots'])
        # Allow retries after partial deletion, but never delete new or changed files.
        if any(f['id'] not in expected or not same_file(expected[f['id']], f) for f in current):
            raise VoiceCloneError("Drive contient des fichiers nouveaux ou modifies. Cree une nouvelle archive avant de nettoyer.")
        journal_path = path.with_suffix(".removed.json")
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {"removed": []}
        removed = 0
        for entry in current:
            fresh = self.service.files().get(fileId=entry['id'], fields=FIELDS, supportsAllDrives=True).execute()
            if not same_file(entry, fresh) or fresh.get('trashed'):
                raise VoiceCloneError("Fichier modifie pendant le nettoyage; nettoyage interrompu.")
            # File-only deletion. Never delete a folder or empty the user's trash.
            self.service.files().delete(fileId=entry['id'], supportsAllDrives=True).execute()
            removed += int(entry['size'])
            journal['removed'].append(entry['id'])
            atomic_json(journal_path, journal)
        return removed

    def restore_actor(self, snapshot: str | Path, actor_name: str) -> int:
        from .naming import safe_name
        actor = safe_name(actor_name)
        data = self.verify(snapshot)
        # Resume uses the prepared experiment, not the original dataset ZIP.
        selected = [f for f in data['files'] if
                    f['parts'][0] == 'VOICECLONE_Training' and len(f['parts']) > 2 and f['parts'][1] == actor]
        if not all(any(f['name'].startswith(prefix) and f['name'].endswith('.pth') for f in selected)
                   for prefix in ('G_', 'D_')):
            raise VoiceCloneError("Cette archive ne contient pas de sauvegarde d'entrainement pour cette voix.")
        by_path = {}
        for entry in selected:
            key = tuple(entry['parts'])
            if key in by_path and by_path[key]['sha256'] != entry['sha256']:
                raise VoiceCloneError("Plusieurs versions d'un fichier d'entrainement dans cette archive; restauration automatique refusee.")
            by_path[key] = entry
        selected = sorted(by_path.values(), key=lambda f: (f['name'] == 'current.json', f['name'] == 'data_manifest.json', f['parts']))
        root_ids = {r['name']: r['id'] for r in self.roots()}
        if 'VOICECLONE_Training' not in root_ids:
            root_ids['VOICECLONE_Training'] = self.service.files().create(body={'name': 'VOICECLONE_Training', 'mimeType': FOLDER}, fields='id').execute()['id']
        folders = {(name,): fid for name, fid in root_ids.items()}
        for entry in selected:
            parts = [safe_component(p) for p in entry['parts']]
            for depth in range(2, len(parts)):
                key = tuple(parts[:depth])
                if key in folders:
                    continue
                parent = folders[key[:-1]]
                escaped = self.client._escape_query_literal(key[-1])
                matches = list_files(self.service, f"'{parent}' in parents and trashed=false and name='{escaped}'")
                if len(matches) > 1 or (matches and matches[0]['mimeType'] != FOLDER):
                    raise VoiceCloneError("Conflit de dossiers lors de la restauration.")
                folders[key] = matches[0]['id'] if matches else self.service.files().create(body={'name': key[-1], 'parents': [parent], 'mimeType': FOLDER}, fields='id').execute()['id']
            self.client.upload_file(self.root / 'objects' / entry['blob'], folders[tuple(parts[:-1])], name=parts[-1])
        return len(selected)


def snapshot_summary(data: dict) -> str:
    groups = {}
    unique = {}
    for f in data['files']:
        groups[f['parts'][0]] = groups.get(f['parts'][0], 0) + int(f['size'])
        unique[f['blob']] = int(f['size'])
    return '\n'.join([f"{len(data['files'])} fichiers verifies."] +
                     [f"{name}: {size / 1e9:.3f} Go" for name, size in groups.items()] +
                     [f"Archive locale sans doublons: {sum(unique.values()) / 1e9:.3f} Go"])
