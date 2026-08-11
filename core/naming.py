from __future__ import annotations

import re
import unicodedata

from .errors import VoiceCloneError


def safe_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", ascii_name).strip(" ._-")
    cleaned = re.sub(r"\s+", "_", cleaned)
    if not cleaned:
        raise VoiceCloneError("Nom invalide. Utilise au moins une lettre ou un chiffre.")
    return cleaned

