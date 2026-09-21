"""Controlled RVC auditions: one setting changed at a time, matched loudness."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import VoiceCloneError
from .ffmpeg_utils import ensure_ffmpeg_available
from .naming import safe_name
from .rvc_inference import convert_voice


PRESETS = (
    ("A", 0.75, 0.33, "Reglage actuel"),
    ("B", 0.60, 0.33, "Index reduit a 0.60"),
    ("C", 0.45, 0.33, "Index reduit a 0.45"),
    ("D", 0.75, 0.25, "Protect reduit a 0.25"),
    ("E", 0.75, 0.15, "Protect reduit a 0.15"),
)


def measure_audio(ffmpeg: Path, path: Path) -> dict:
    completed = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-nostats", "-i", str(path), "-af",
         "loudnorm=I=-23:TP=-2:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, timeout=180,
    )
    if completed.returncode:
        raise VoiceCloneError(f"Mesure audio impossible pour {path.name}: {completed.stderr[-1500:]}")
    start = completed.stderr.rfind('{')
    end = completed.stderr.rfind('}')
    values = json.loads(completed.stderr[start:end + 1])
    loudness, peak = float(values['input_i']), float(values['input_tp'])
    if not math.isfinite(loudness) or not math.isfinite(peak):
        raise VoiceCloneError(f"Extrait silencieux ou niveau non mesurable: {path.name}")
    info = sf.info(path)
    return dict(lufs=loudness, true_peak_dbtp=peak, duration=info.duration,
                sample_rate=info.samplerate, channels=info.channels, frames=info.frames)


def listening_target(measurements: list[dict]) -> float:
    # Linear gain only; no limiter, compression, or clipping to bias the audition.
    return min([-23.0] + [m['lufs'] - 2.0 - m['true_peak_dbtp'] for m in measurements])


def match_loudness(source: Path, destination: Path, measured: dict, target: float) -> float:
    gain_db = target - measured['lufs']
    audio, rate = sf.read(source, dtype='float32', always_2d=True)
    audio *= 10 ** (gain_db / 20)
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) >= 1:
        raise VoiceCloneError("Niveau d'ecoute invalide; aucun ecretage applique.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, rate, subtype='PCM_24')
    return gain_db


def run_comparison(settings, model_path: Path, guide: Path, reference: Path | None = None,
                   transpose: int = 0, progress=print) -> Path:
    if not model_path.is_file() or not guide.is_file():
        raise VoiceCloneError("Modele ou guide de comparaison introuvable.")
    ffmpeg = ensure_ffmpeg_available(settings)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder = settings.root_path / 'outputs' / 'comparisons' / f'{safe_name(model_path.stem)}_{stamp}'
    folder.mkdir(parents=True)
    shutil.copy2(guide, folder / 'guide_original.wav')
    if reference:
        shutil.copy2(reference, folder / 'reference_catherine.wav')
    report = dict(model=str(model_path), guide=str(guide), transpose=transpose,
                  f0_method=settings.rvc.f0_method, seed=20260912, variants=[],
                  reference_note="Reference de timbre issue du dataset; texte different du guide.")
    manifest = folder / 'comparaison.json'
    def save():
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    for code, index_rate, protect, description in PRESETS:
        progress(f"Rendu {code}/E : index={index_rate:.2f}, protect={protect:.2f}")
        result = convert_voice(settings, model_path, folder / 'guide_original.wav',
                               transpose=transpose, index_rate=index_rate, protect=protect,
                               seed=report['seed'])
        destination = folder / f'{code}_original.wav'
        # The generated output is moved only within this project's outputs folder.
        if not result.resolve().is_relative_to((settings.root_path / 'outputs').resolve()):
            raise VoiceCloneError("Sortie hors dossier autorise.")
        result.replace(destination)
        measured = measure_audio(ffmpeg, destination)
        report['variants'].append(dict(code=code, index_rate=index_rate, protect=protect,
                                       description=description, original=destination.name, measured=measured))
        save()
    items = [dict(code='Guide', original='guide_original.wav', description='Guide utilise pour les cinq conversions')]
    if reference:
        items.append(dict(code='Reference', original='reference_catherine.wav', description=report['reference_note']))
    items.extend(report['variants'])
    for item in items:
        if 'measured' not in item:
            item['measured'] = measure_audio(ffmpeg, folder / item['original'])
    target = listening_target([item['measured'] for item in items])
    report['listening_target_lufs'] = target
    report['listening'] = items
    for item in items:
        item['listening_file'] = f"ecoute/{item['code']}.wav"
        item['gain_db'] = match_loudness(folder / item['original'], folder / item['listening_file'], item['measured'], target)
        item['matched_measurement'] = measure_audio(ffmpeg, folder / item['listening_file'])
    report['completed'] = True
    save()
    template = Path(__file__).resolve().parents[1] / 'templates' / 'rvc_comparison.html'
    page = template.read_text(encoding='utf-8').replace('__COMPARISON_DATA__', json.dumps(report, ensure_ascii=False).replace('<', '\\u003c'))
    (folder / 'ecouter.html').write_text(page, encoding='utf-8')
    progress(f"Comparaison terminee: {folder}")
    return folder
