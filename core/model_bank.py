from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Settings, ensure_runtime_layout


@dataclass
class VoiceRecord:
    id: str
    actor_name: str
    model_path: str
    index_path: str | None
    status: str


def load_voice_bank(settings: Settings) -> dict[str, Any]:
    paths = ensure_runtime_layout(settings)
    if not paths.voice_bank_json.exists():
        return {"voices": []}
    return json.loads(paths.voice_bank_json.read_text(encoding="utf-8"))


def save_voice_bank(settings: Settings, bank: dict[str, Any]) -> None:
    paths = ensure_runtime_layout(settings)
    paths.voice_bank_json.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_voice_models(settings: Settings) -> list[VoiceRecord]:
    paths = ensure_runtime_layout(settings)
    existing_bank = load_voice_bank(settings)
    existing_by_id = {voice.get("id"): voice for voice in existing_bank.get("voices", [])}

    records: list[VoiceRecord] = []
    for model_path in sorted(paths.models_bank.glob("*.pth")):
        record_id = model_path.stem
        existing = existing_by_id.get(record_id, {})
        index_path = _find_matching_index(paths.models_bank, model_path)
        records.append(
            VoiceRecord(
                id=record_id,
                actor_name=existing.get("actor_name") or record_id,
                model_path=str(model_path),
                index_path=str(index_path) if index_path else None,
                status="ready" if index_path else "missing_index",
            )
        )

    save_voice_bank(settings, {"voices": [asdict(record) for record in records]})
    return records


def find_index_for_model(settings: Settings, model_path: Path) -> Path | None:
    paths = ensure_runtime_layout(settings)
    return _find_matching_index(paths.models_bank, model_path)


def _find_matching_index(models_dir: Path, model_path: Path) -> Path | None:
    same_stem = models_dir / f"{model_path.stem}.index"
    if same_stem.exists():
        return same_stem

    candidates = sorted(models_dir.glob("*.index"))
    lowered_stem = model_path.stem.lower()
    for candidate in candidates:
        if lowered_stem in candidate.stem.lower() or candidate.stem.lower() in lowered_stem:
            return candidate
    return candidates[0] if len(candidates) == 1 else None

