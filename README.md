# Sermon Transcribe (8GB GPU)

An 8GB‑friendly, long‑form sermon transcription tool built on WhisperX + Faster‑Whisper with alignment, optional diarization, and robust prose formatting. Includes a safe optional polishing step.

## Features

- VAD segmentation and batched decoding for long‑form audio (1.5h+)
- Whisper Large‑V3 (default) with GPU‑aware batch auto‑tuning; Turbo/Distil options
- Accurate timestamps via forced alignment; optional diarization
- Sermon‑specific paragraphing; optional constrained polish step
- Exports: Markdown, DOCX, JSON (words + timestamps), optional SRT/VTT
- Bake‑off script to compare models/configs

## Requirements

- Linux with NVIDIA GPU (8GB VRAM target)
- Python 3.10+
- CUDA-compatible PyTorch
- uv as the project manager: https://docs.astral.sh/uv/

## Quickstart (uv)

```bash
# create and sync environment
uv sync

# run CLI help
uv run sermon-transcribe --help

# transcribe a sermon (basic)
uv run sermon-transcribe transcribe \
  --config configs/example.yaml \
  sermons/2025-10-12.wav

# compare two configs on a 10–15 min excerpt
uv run sermon-transcribe compare \
  --config-a configs/example.yaml \
  --config-b configs/example_fast.yaml \
  samples/excerpt.wav
```

## Configuration

See `configs/example.yaml` for a complete example with comments. Key sections:

- audio, vad, asr, alignment, diarization, prose (format + polish), dictionary, export, eval, perf

## Notes

- On first run, models are downloaded to your HF cache. Ensure disk space.
- For diarization, accept pyannote model licenses and set your HF token.
- If you hit OOM on 8GB, the pipeline auto‑reduces batch size; it can also switch to int8_float16 or int8 compute types.

## Development

```bash
uv run pytest -q
uv run sermon-transcribe --help
```

## License

MIT
