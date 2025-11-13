from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

from .asr import ASRResult, Segment

logger = logging.getLogger(__name__)


def run_alignment(audio: Any, asr: ASRResult) -> ASRResult:
    """Run WhisperX alignment to get accurate word timestamps.

    Words that cannot be aligned will have null timings; we leave the text intact.
    """
    # Remove any stub modules before importing whisperx to ensure real modules load
    import sys
    stub_modules = []
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("torchaudio", "torchaudio.pipelines", "pyannote", "pyannote.audio", 
                       "pyannote.audio.core", "pyannote.audio.core.io", "pyannote.core"):
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, '__file__') and getattr(mod, '__file__', None) == "<stub>":
                stub_modules.append(mod_name)
    
    for mod_name in stub_modules:
        del sys.modules[mod_name]

    import whisperx  # type: ignore

    if not asr.segments:
        return asr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Alignment is GPU-accelerated in this pipeline"

    align_model, metadata = whisperx.load_align_model(language_code=asr.language, device=device)

    # Reformat segments to WhisperX expected format
    seg_dicts: List[Dict[str, Any]] = [
        {"start": s.start, "end": s.end, "text": s.text} for s in asr.segments
    ]

    logger.info("Running alignment for %d segments...", len(seg_dicts))
    aligned = whisperx.align(seg_dicts, align_model, metadata, audio, device, return_char_alignments=False)

    out_segments: List[Segment] = []
    for aligned_seg in aligned.get("segments", []):
        words = []
        for wd in aligned_seg.get("words", []):
            words.append({
                "word": wd.get("word", wd.get("text", "")),
                "start": wd.get("start"),
                "end": wd.get("end"),
                "confidence": wd.get("score"),
            })
        out_segments.append(Segment(
            start=aligned_seg.get("start"),
            end=aligned_seg.get("end"),
            text=aligned_seg.get("text", ""),
            words=words
        ))

    logger.info("Alignment complete")
    return ASRResult(language=asr.language, segments=out_segments, wall_time_s=asr.wall_time_s, peak_vram_gb=asr.peak_vram_gb)
