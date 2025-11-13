from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch

from .config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: Optional[List[Dict[str, Any]]] = None


@dataclass
class ASRResult:
    language: str
    segments: List[Segment]
    wall_time_s: float
    peak_vram_gb: float


def _cuda_vram_gb() -> Tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free_b, total_b = torch.cuda.mem_get_info()
    return free_b / (1024 ** 3), total_b / (1024 ** 3)


def _peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _try_batch_size(transcriber, audio, batch_size: int, language: str) -> bool:
    try:
        _ = transcriber.transcribe(audio, batch_size=batch_size, language=language)
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("OOM at batch_size=%d; reducing", batch_size)
            torch.cuda.empty_cache()
            return False
        raise


def _auto_tune_batch(transcriber, audio, cfg: AppConfig, max_trials: int = 5) -> int:
    bs = cfg.asr.batch_size
    trials = 0
    while trials < max_trials and bs >= 1:
        ok = _try_batch_size(transcriber, audio[: min(16000 * 30, len(audio))], bs, cfg.asr.language)
        if ok:
            free_gb, total_gb = _cuda_vram_gb()
            logger.info("Batch %d ok; free VRAM: %.2f/%.2f GB", bs, free_gb, total_gb)
            # Ensure headroom; if too little, reduce
            if free_gb < cfg.asr.vram_target_gb and bs > 1:
                bs = max(1, bs // 2)
                trials += 1
            else:
                break
        else:
            bs = max(1, bs // 2)
            trials += 1
    assert bs >= 1, "Auto-tuned batch size must be >=1"
    return bs


def transcribe_with_whisperx(audio: Any, cfg: AppConfig, progress_callback: Optional[Callable[[float, float], None]] = None) -> ASRResult:
    """Transcribe using WhisperX with Faster-Whisper backend.

    Relies on WhisperX for VAD-based segmentation and batched decoding.
    Alignment is handled in a separate step.
    
    Args:
        audio: Audio array to transcribe
        cfg: Application configuration
        progress_callback: Optional callback(current_time, total_duration) for progress updates
    """
    # Workaround: avoid importing real torchaudio (which may require CUDA libs) and
    # provide a minimal stub with the attributes referenced by dependencies.
    try:  # pragma: no cover - environment dependent
        import sys
        import types
        if "torchaudio" not in sys.modules:
            from importlib.machinery import ModuleSpec  # type: ignore
            ta = types.ModuleType("torchaudio")

            class _AudioMetaData:  # minimal placeholder
                pass

            def _list_audio_backends():
                try:
                    import soundfile  # noqa: F401
                    return ["soundfile"]
                except Exception:
                    return []

            setattr(ta, "AudioMetaData", _AudioMetaData)
            setattr(ta, "list_audio_backends", _list_audio_backends)
            # Provide minimal module metadata so find_spec works
            try:
                ta.__spec__ = ModuleSpec(name="torchaudio", loader=None)  # type: ignore[attr-defined]
                ta.__file__ = "<stub>"
                ta.__path__ = []  # type: ignore[attr-defined]
            except Exception:
                pass
            sys.modules["torchaudio"] = ta
    except Exception:  # noqa: BLE001
        pass

    # Stub out heavy pyannote imports that WhisperX's VAD module pulls in at import-time.
    # We don't use the Pyannote VAD path by default, but importing it drags torchaudio CUDA libs.
    # Providing minimal stubs prevents unnecessary native library loading in environments
    # where torchaudio CUDA extension is unavailable.
    try:  # pragma: no cover
        import sys
        import types
        if "pyannote.audio" not in sys.modules:
            pa = types.ModuleType("pyannote")
            paa = types.ModuleType("pyannote.audio")
            pac = types.ModuleType("pyannote.audio.core")
            pacio = types.ModuleType("pyannote.audio.core.io")
            pacore = types.ModuleType("pyannote.core")
            pap = types.ModuleType("pyannote.audio.pipelines")
            papu = types.ModuleType("pyannote.audio.pipelines.utils")

            # Minimal API used only for typing at import-time
            setattr(paa, "Model", object)
            setattr(paa, "Pipeline", object)
            setattr(paa, "Inference", object)
            setattr(pacio, "AudioFile", object)
            setattr(pap, "VoiceActivityDetection", object)
            setattr(papu, "PipelineModel", object)
            setattr(pacore, "Annotation", object)
            setattr(pacore, "SlidingWindowFeature", object)
            setattr(pacore, "Segment", object)

            sys.modules["pyannote"] = pa
            sys.modules["pyannote.audio"] = paa
            sys.modules["pyannote.audio.core"] = pac
            sys.modules["pyannote.audio.core.io"] = pacio
            sys.modules["pyannote.core"] = pacore
            sys.modules["pyannote.audio.pipelines"] = pap
            sys.modules["pyannote.audio.pipelines.utils"] = papu
    except Exception:  # noqa: BLE001
        pass

    # Use Faster-Whisper directly to avoid VAD (Silero/Pyannote) CUDA/cuDNN dependencies
    from faster_whisper import WhisperModel  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "GPU (CUDA) is required for 8GB target performance"

    # Ensure Torch won't attempt to use cuDNN (some environments lack libcudnn)
    try:  # pragma: no cover
        torch.backends.cudnn.enabled = False  # type: ignore[attr-defined]
    except Exception:
        pass

    # Normalize model id for Faster-Whisper: map OpenAI repo ids to FW shorthand
    model_id = cfg.asr.model_id
    if cfg.asr.backend.lower() in {"ctranslate2", "faster_whisper", "faster-whisper"}:
        tail = model_id.split("/")[-1]
        if tail.startswith("whisper-"):
            tail = tail[len("whisper-"):]
        model_id = tail  # e.g., "large-v3"

    logger.info("Loading ASR model %s on %s (%s)", model_id, device, cfg.asr.compute_type)
    fw = WhisperModel(model_id, device=device, compute_type=cfg.asr.compute_type)

    # Calculate total duration for progress tracking
    sample_rate = 16000  # We normalize to 16kHz in audio.py
    total_duration = len(audio) / sample_rate
    
    logger.info("Processing audio with duration %s", time.strftime("%H:%M:%S", time.gmtime(total_duration)))
    
    # Transcribe in fixed chunks (no VAD) to avoid torch/cuDNN paths
    start = time.time()
    segments_iter, info = fw.transcribe(
        audio,
        language=cfg.asr.language,
        task=cfg.asr.task,
        beam_size=getattr(cfg.asr, "beam_size", 1) or 1,
        condition_on_previous_text=False,
        vad_filter=False,
        chunk_length=int(getattr(cfg.vad, "segment_len_s", 30) or 30),
    )

    segs: List[Segment] = []
    # Iterate through segments as they're generated
    for s in segments_iter:
        # faster-whisper yields objects with .start, .end, .text
        seg = Segment(start=float(s.start), end=float(s.end), text=str(s.text).strip(), words=None)
        segs.append(seg)
        # Report progress based on segment end time
        if progress_callback:
            # Clamp to total_duration to avoid going over 100%
            progress_callback(min(seg.end, total_duration), total_duration)
    
    # Ensure progress reaches 100%
    if progress_callback:
        progress_callback(total_duration, total_duration)
    
    wall = time.time() - start
    peak = _peak_vram_gb()
    lang = info.language or cfg.asr.language
    logger.info("Decoded %d segments in %.1fs (peak VRAM %.2f GB)", len(segs), wall, peak)
    return ASRResult(language=lang, segments=segs, wall_time_s=wall, peak_vram_gb=peak)
