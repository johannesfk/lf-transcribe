from __future__ import annotations
import os
from faster_whisper import WhisperModel
import torch
from typing import List, Tuple
import gc

def process_audio(
        audio_file: str,
        model_size: str = "medium",
        batch_size: int = 16,
        use_cuda: bool = True,
        fallback_to_cpu: bool = True,
        lang: str = "en"
) -> None:
    """Process audio file with memory-optimized settings"""

    if use_cuda and torch.cuda.is_available():
        try:
            model = WhisperModel(
                model_size,
                device="cuda",
                compute_type="int8_float16",
                num_workers=2,
                cpu_threads=4
            )
        except RuntimeError as e:
            if fallback_to_cpu and "out of memory" in str(e):
                print("GPU out of memory, falling back to CPU...")
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
            else:
                raise e
    else:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # Ensure output directory exists
    os.makedirs("transcripts", exist_ok=True)

    # Transcribe with optimized settings
    segments, info = model.transcribe(
        audio_file,
        beam_size=4,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        language=lang
    )

    print(f"Detected language '{info.language}' with probability {info.language_probability}")
        
    # Process segments in batches
    segment_batch: List[Tuple[float, float, str]] = []
    output_file = f"transcripts/transcribe-{audio_file.split('.')[0]}.txt"
    
    for segment in segments:
        segment_batch.append((segment.start, segment.end, segment.text))
        
        # Process batch when full or at end
        if len(segment_batch) >= batch_size:
            # Write batch to file
            with open(output_file, "a") as f:
                for start, end, text in segment_batch:
                    print(f"[{start:.2f}s -> {end:.2f}s] {text}")
                    f.write(f"{text}\n")
            
            # Clear batch
            segment_batch = []
            
            # Periodic memory cleanup
            if use_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    # Write any remaining segments
    if segment_batch:
        with open(output_file, "a") as f:
            for start, end, text in segment_batch:
                print(f"[{start:.2f}s -> {end:.2f}s] {text}")
                f.write(f"{text}\n")

if __name__ == "__main__":
    audio_file = "251105-AUDACITY-Freedom.mp3"
    model_size = "large-v2"  # Try smaller model size if still having memory issues
    
    process_audio(
        audio_file=audio_file,
        model_size=model_size,
        batch_size=16,
        use_cuda=True,
        fallback_to_cpu=False
    )