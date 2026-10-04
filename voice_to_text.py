import sys
import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import whisper

sys.stdout.reconfigure(encoding='utf-8')

SAMPLE_RATE = 16000
DURATION = 6

print("========================================")
print("1. English Speech-to-Text (Teacher Task)")
print("2. Urdu Speech-to-Text (اردو)")
print("========================================")
mode = input("Select (1 or 2): ").strip()

lang_code = "en" if mode == "1" else "ur"
lang_name = "English" if mode == "1" else "Urdu"

print("\nLoading Whisper model...")
model = whisper.load_model("tiny")  # 'base' ki jagah 'tiny' likhein
print("Model ready!")

print(f"\n>>> Speak NOW in {lang_name} for {DURATION} seconds...")
audio_data = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"  # Clear audio capture
)
sd.wait()

# Audio volume boost (agar mic ki awaaz halki ho to normalize kar dega)
max_val = np.max(np.abs(audio_data))
if max_val > 0:
    audio_data = audio_data / max_val

# Save audio
audio_int16 = (audio_data * 32767).astype(np.int16)
write("recording.wav", SAMPLE_RATE, audio_int16)

print("AI is transcribing your voice...")

# Direct Whisper run without confusion
if lang_code == "ur":
    result = model.transcribe(
        "recording.wav",
        language="ur",
        fp16=False,
        initial_prompt="یہ اردو ہے: میرا نام علی ہے، کمپیوٹر، پروجیکٹ۔"
    )
else:
    result = model.transcribe(
        "recording.wav",
        language="en",
        fp16=False
    )

recognized_text = result["text"].strip()

print("\n" + "="*40)
print("RECOGNIZED TEXT:")
print(recognized_text if recognized_text else "[Koi aawaz detect nahi hui]")
print("="*40)

# Save and open in Notepad
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(recognized_text)

os.system("notepad result.txt")
print("\n========== TIMESTAMPS RESULT ==========")
for segment in result["segments"]:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:.2f}s -> {end:.2f}s] : {text}")
print("========================================")