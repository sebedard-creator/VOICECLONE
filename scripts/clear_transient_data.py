"""Delete only VOICECLONE-QC data that can be safely recreated."""

from __future__ import annotations

import shutil
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.resolve()
TARGET_NAMES = (
    "cache",
    "temp",
    "staging",
    "dataset_cleaned",
    "Nettoyage Seul",
    "outputs",
    "logs",
)


def is_reparse_point(path: Path) -> bool:
    return bool(path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def clear_folder(folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    resolved = folder.resolve()
    if folder.name not in TARGET_NAMES or not resolved.is_relative_to(ROOT):
        raise RuntimeError(f"Refusing to clear unsafe folder: {folder}")

    deleted = 0
    for child in folder.iterdir():
        if child.is_symlink() or is_reparse_point(child):
            child.unlink() if child.is_file() else child.rmdir()
        else:
            child_resolved = child.resolve()
            if not child_resolved.is_relative_to(resolved):
                raise RuntimeError(f"Refusing to follow path outside target: {child}")
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        deleted += 1
    return deleted


def main() -> None:
    print(f"VOICECLONE root: {ROOT}")
    total = 0
    for name in TARGET_NAMES:
        folder = ROOT / name
        removed = clear_folder(folder)
        total += removed
        print(f"{name}: {removed} item(s) removed")
    print(f"Complete: {total} top-level transient item(s) removed.")


if __name__ == "__main__":
    main()
