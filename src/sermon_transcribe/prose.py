from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from .asr import Segment

logger = logging.getLogger(__name__)


@dataclass
class Paragraph:
    start: float
    end: float
    text: str


def _split_sentences(text: str) -> List[str]:
    # Simple sentence splitter suitable for sermons; avoids heavy deps
    # Split on . ! ? followed by space/capital or end of text
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'\(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _starts_discourse(s: str, starters: List[str]) -> bool:
    return any(s.startswith(w + " ") for w in starters)


def paragraphs_from_segments(
    segs: List[Segment],
    paragraph_gap_s: float = 2.5,
    max_sentences: int = 6,
    discourse_starters: Optional[List[str]] = None,
) -> List[Paragraph]:
    discourse_starters = discourse_starters or ["Now", "So", "Let’s", "In closing"]
    paras: List[Paragraph] = []
    cur_sentences: List[str] = []
    cur_start: Optional[float] = None
    last_end: Optional[float] = None

    for s in segs:
        if cur_start is None:
            cur_start = s.start
        # New paragraph if large gap
        if last_end is not None and (s.start - last_end) > paragraph_gap_s:
            if cur_sentences:
                paras.append(Paragraph(start=cur_start, end=last_end, text=" ".join(cur_sentences).strip()))
                cur_sentences = []
                cur_start = s.start
        # Add sentences from this segment
        for sentence in _split_sentences(s.text):
            if _starts_discourse(sentence, discourse_starters) and cur_sentences:
                paras.append(Paragraph(start=cur_start, end=s.start, text=" ".join(cur_sentences).strip()))
                cur_sentences = [sentence]
                cur_start = cur_start if cur_start is not None else s.start
            else:
                cur_sentences.append(sentence)
            if len(cur_sentences) >= max_sentences:
                paras.append(Paragraph(start=cur_start, end=s.end, text=" ".join(cur_sentences).strip()))
                cur_sentences = []
                cur_start = None
        last_end = s.end
    if cur_sentences and last_end is not None:
        paras.append(Paragraph(start=cur_start or 0.0, end=last_end, text=" ".join(cur_sentences).strip()))
    return paras


def apply_polish_guarded(text: str, max_diff_ratio: float = 0.07) -> str:
    """Placeholder constrained polishing step.

    In this base implementation, we do a conservative pass: trim spaces and
    standardize common punctuation spaces. Real LLM-based polish can be plugged
    in here under the same guard.
    """
    pre = text
    out = re.sub(r"\s+", " ", pre).strip()
    # diff guard (very naive):
    if len(pre) == 0:
        return out
    ratio = abs(len(out) - len(pre)) / max(1, len(pre))
    if ratio > max_diff_ratio:
        logger.warning("Polish skipped due to diff ratio %.3f > %.3f", ratio, max_diff_ratio)
        return pre
    return out
