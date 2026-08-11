from __future__ import annotations

import os
import ctypes
import importlib.util
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []
_PRELOADED_TORCHCODEC_HANDLES = []


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []

    for raw_dir in os.environ.get("PATH", "").split(os.pathsep):
        if raw_dir:
            dirs.append(Path(raw_dir))

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if packages.exists():
            dirs.extend(path.parent for path in packages.glob("**/avcodec-*.dll"))

    return dirs


def _add_ffmpeg_dll_dirs() -> None:
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return

    seen: set[str] = set()
    for directory in _candidate_dirs():
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if any(resolved.glob("avcodec-*.dll")):
            try:
                _DLL_DIRECTORY_HANDLES.append(add_dll_directory(str(resolved)))
            except OSError:
                continue


_add_ffmpeg_dll_dirs()


def _preload_torchcodec() -> None:
    if os.environ.get("VOICECLONE_PRELOAD_TORCHCODEC") != "1":
        return
    try:
        import torch  # noqa: F401

        spec = importlib.util.find_spec("torchcodec")
        if spec is None or not spec.submodule_search_locations:
            return
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        for dll_path in sorted(package_dir.glob("libtorchcodec_core*.dll"), reverse=True):
            try:
                _PRELOADED_TORCHCODEC_HANDLES.append(ctypes.CDLL(str(dll_path)))
            except OSError:
                continue
    except Exception:
        return


_preload_torchcodec()
