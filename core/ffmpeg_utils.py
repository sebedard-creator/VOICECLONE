from __future__ import annotations

import os
import shutil
from pathlib import Path

from pydub import AudioSegment

from .config import Settings
from .errors import DependencyMissing


def ensure_ffmpeg_available(settings: Settings) -> Path:
    configured_path = getattr(settings.audio, "ffmpeg_path", "") or ""
    candidates = []
    if configured_path:
        candidates.append(Path(configured_path))

    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))

    candidates.extend(_discover_winget_ffmpeg())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            _activate_ffmpeg(candidate)
            return candidate

    raise DependencyMissing(
        "ffmpeg est requis pour lire et exporter l'audio. "
        "Installe FFmpeg ou renseigne audio.ffmpeg_path dans config/settings.json."
    )


def _discover_winget_ffmpeg() -> list[Path]:
    roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path(os.environ.get("ProgramFiles", "")) / "WinGet" / "Packages",
    ]
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            matches.extend(root.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
            matches.extend(root.glob("*FFmpeg*/**/bin/ffmpeg.exe"))
        except OSError:
            continue
    return sorted(set(matches), key=lambda path: ("shared" not in str(path).lower(), str(path)))


def _activate_ffmpeg(ffmpeg_path: Path) -> None:
    bin_dir = str(ffmpeg_path.parent)
    current_path = os.environ.get("PATH", "")
    path_parts = [part.lower() for part in current_path.split(os.pathsep) if part]
    if bin_dir.lower() not in path_parts:
        os.environ["PATH"] = bin_dir + os.pathsep + current_path

    AudioSegment.converter = str(ffmpeg_path)
