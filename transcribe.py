from __future__ import annotations
import os
from faster_whisper import WhisperModel

audio_file = "left.wav"
model_size = "large-v3"

if __name__ == "__main__":
    try:
        print(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']!r}")

    except KeyError:
        print("Environment variables not set!")

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe(audio_file, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

if not os.path.exists("transcribes"):
    os.makedirs("transcribes")

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    # Write the segment text to a file called the same as the audio file name, but with a .txt extension
    # If the file doesn't exist, create it. If it does, append to it.
    # The file should be in a folder called "transcribes"
    # If the folder doesn't exist, create it.
    with open(f"transcribes/{audio_file.split('.')[0]}.txt", "a") as f:
        f.write(segment.text + "\n")