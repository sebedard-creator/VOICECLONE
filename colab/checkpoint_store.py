#@title VOICECLONE-QC - sauvegardes compactes et reprise verifiee
"""Embedded in the notebook by scripts/update_colab_notebook.py."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import uuid


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = file_hash(source)
    if destination.is_file() and file_hash(destination) == expected:
        return expected
    temporary = destination.with_name(destination.name + ".part")
    shutil.copy2(source, temporary)
    if file_hash(temporary) != expected:
        raise RuntimeError(f"Verification failed: {destination}")
    temporary.replace(destination)
    return expected


def checkpoint_epoch(path):
    import torch
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    return int(checkpoint["iteration"])


class CheckpointStore:
    def __init__(self, run_dir, epoch_reader=checkpoint_epoch):
        self.root = Path(run_dir)
        self.experiment = self.root / "experiment"
        self.checkpoints = self.root / "checkpoints"
        self.pointer = self.checkpoints / "current.json"
        self.epoch_reader = epoch_reader
        self.last_signature = None

    def save_data(self, local):
        local = Path(local)
        self.experiment.mkdir(parents=True, exist_ok=True)
        files = {}
        for source in local.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(local)
            if (source.name.startswith(("G_", "D_", "events.out.tfevents"))
                    or source.suffix in {".index", ".part"} or source.name == "total_fea.npy"):
                continue
            files[relative.as_posix()] = copy_verified(source, self.experiment / relative)
        write_json_atomic(self.root / "data_manifest.json", {"files": files})

    def save_checkpoints(self, local):
        local = Path(local)
        candidates = [sorted(local.glob(pattern), key=lambda p: p.stat().st_mtime_ns, reverse=True)
                      for pattern in ("G_*.pth", "D_*.pth")]
        if not all(candidates):
            return False
        pair = [items[0] for items in candidates]
        signature = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in pair)
        if signature == self.last_signature:
            return True
        # Freeze both local files before uploading. RVC may be writing the next save.
        with tempfile.TemporaryDirectory(prefix="voiceclone-checkpoint-") as scratch:
            frozen = [Path(scratch) / p.name for p in pair]
            try:
                for source, target in zip(pair, frozen):
                    shutil.copy2(source, target)
                if signature != tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in pair):
                    return False
                epochs = [self.epoch_reader(p) for p in frozen]
                if epochs[0] != epochs[1]:
                    return False
            except Exception as error:
                print(f"Checkpoint still being written or unreadable; previous backup retained: {type(error).__name__}")
                return False
            generation = "generation_" + uuid.uuid4().hex
            target = self.checkpoints / generation
            hashes = {p.name: copy_verified(p, target / p.name) for p in frozen}
            manifest_path = self.root / "data_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
            for name in ("config.json", "filelist.txt"):
                if (local / name).is_file():
                    manifest["files"][name] = copy_verified(local / name, self.experiment / name)
            write_json_atomic(manifest_path, manifest)
            # The old complete pair remains valid until the new pair is verified.
            write_json_atomic(self.pointer, {"generation": generation, "epoch": epochs[0], "files": hashes})
            self.last_signature = signature
        # Retain one committed pair, never two permanent copies in experiment/.
        for folder in self.checkpoints.glob("generation_*"):
            if folder.name != generation and folder.is_dir() and not folder.is_symlink():
                shutil.rmtree(folder)
        for folder in (self.checkpoints, self.experiment):
            for pattern in ("G_*.pth", "D_*.pth"):
                for obsolete in folder.glob(pattern):
                    obsolete.unlink()
        print(f"Verified checkpoint backup: epoch {epochs[0]}", flush=True)
        return True

    def restore(self, local):
        local = Path(local)
        if not self.experiment.is_dir():
            raise RuntimeError("Training data backup is missing. Restore the local archive to Drive first.")
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(self.experiment, local)
        manifest = self.root / "data_manifest.json"
        if manifest.exists():
            for name, expected in json.loads(manifest.read_text())["files"].items():
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Invalid backup data path")
                if file_hash(local / relative) != expected:
                    raise RuntimeError(f"Corrupt training data: {name}")
        if self.pointer.exists():
            data = json.loads(self.pointer.read_text())
            generation = data["generation"]
            if not generation.startswith("generation_") or Path(generation).name != generation:
                raise RuntimeError("Invalid checkpoint generation")
            pair = []
            for name, expected in data["files"].items():
                if Path(name).name != name or not name.startswith(("G_", "D_")) or not name.endswith(".pth"):
                    raise RuntimeError("Invalid checkpoint filename")
                source = self.checkpoints / generation / name
                if file_hash(source) != expected:
                    raise RuntimeError(f"Corrupt checkpoint: {name}")
                pair.append(source)
        else:
            # v1.2.x stored newer checkpoints beside an older experiment copy.
            by_epoch = {}
            for folder in (self.experiment, self.checkpoints):
                for pattern in ("G_*.pth", "D_*.pth"):
                    for candidate in folder.glob(pattern):
                        try:
                            epoch = self.epoch_reader(candidate)
                            by_epoch.setdefault(epoch, {})[candidate.name[0]] = candidate
                        except Exception:
                            continue
            valid = [epoch for epoch, items in by_epoch.items() if set(items) == {"G", "D"}]
            if not valid:
                raise RuntimeError("No complete G/D checkpoint pair. Resume refused instead of restarting at zero.")
            pair = list(by_epoch[max(valid)].values())
        if len(pair) != 2 or {p.name[0] for p in pair} != {"G", "D"}:
            raise RuntimeError("Incomplete checkpoint pair")
        epochs = [self.epoch_reader(p) for p in pair]
        if epochs[0] != epochs[1]:
            raise RuntimeError("G and D checkpoints have different epochs")
        for pattern in ("G_*.pth", "D_*.pth"):
            for stale in local.glob(pattern):
                stale.unlink()
        for source in pair:
            copy_verified(source, local / source.name)
        print(f"Restored verified checkpoint pair: epoch {epochs[0]}")
        return epochs[0]
