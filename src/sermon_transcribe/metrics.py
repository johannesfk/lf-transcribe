from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from jiwer import wer as jiwer_wer

from .asr import ASRResult

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    wall_time_s: float
    peak_vram_gb: float
    wer: Optional[float] = None


def compute_metrics(asr: ASRResult, reference_text: Optional[str]) -> Metrics:
    wall = asr.wall_time_s
    peak = asr.peak_vram_gb
    if reference_text:
        hyp = " ".join(s.text for s in asr.segments)
        val = 100.0 * jiwer_wer(reference_text, hyp)
        logger.info("WER: %.2f%%", val)
        return Metrics(wall_time_s=wall, peak_vram_gb=peak, wer=val)
    return Metrics(wall_time_s=wall, peak_vram_gb=peak, wer=None)
