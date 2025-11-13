from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def load_mono_16k(path: str | Path) -> tuple[np.ndarray, int]:
    p = Path(path)
    assert p.exists(), f"Audio file not found: {p}"
    audio, sr = librosa.load(str(p), sr=16000, mono=True)
    assert audio.size > 0, "Loaded empty audio array"
    return audio.astype(np.float32), 16000


def peak_normalize(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    assert audio.ndim == 1, "Audio must be mono"
    max_val = np.max(np.abs(audio)) or 1.0
    scale = peak / max_val
    if scale < 1.0:
        logger.debug("Peak-normalizing by scale %.4f", scale)
        audio = audio * scale
    return audio


def save_wav(path: str | Path, audio: np.ndarray, sr: int = 16000) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
