from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)
    from pydub import AudioSegment

from .errors import VoiceCloneError


def normalize_peak(segment: AudioSegment, target_peak_dbfs: float) -> AudioSegment:
    if segment.max_dBFS == float("-inf"):
        return segment
    return segment.apply_gain(target_peak_dbfs - segment.max_dBFS)


def audiosegment_to_float32(segment: AudioSegment) -> np.ndarray:
    mono = segment.set_channels(1)
    samples = np.array(mono.get_array_of_samples()).astype(np.float32)
    scale = float(1 << (8 * mono.sample_width - 1))
    if scale <= 0:
        raise VoiceCloneError("Largeur d'echantillon audio invalide.")
    return np.clip(samples / scale, -1.0, 1.0)


def write_wav_24bit(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples.astype(np.float32), sample_rate, subtype="PCM_24")


def export_segment_wav_24bit(segment: AudioSegment, path: Path, sample_rate: int) -> None:
    normalized = segment.set_channels(1).set_frame_rate(sample_rate)
    write_wav_24bit(path, audiosegment_to_float32(normalized), sample_rate)


def convert_file_to_wav_24bit(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    target_peak_dbfs: float | None = None,
) -> Path:
    data, _ = librosa.load(str(input_path), sr=sample_rate, mono=True)
    if target_peak_dbfs is not None and data.size:
        peak = float(np.max(np.abs(data)))
        if peak > 0:
            target = 10 ** (target_peak_dbfs / 20.0)
            data = np.clip(data * (target / peak), -1.0, 1.0)
    write_wav_24bit(output_path, data, sample_rate)
    return output_path
