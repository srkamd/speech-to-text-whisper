import os
import time
import csv
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# Settings
DATASET_DIR = "my_dataset"
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
CSV_PATH = os.path.join(DATASET_DIR, "metadata.csv")
SAMPLE_RATE = 16000
DURATION = 5  # Har sentence ke liye 5 seconds

# Folders create karein agar nahi hain
os.makedirs(AUDIO_DIR, exist_ok=True)

# Training ke liye sentences list (Aap aur bhi add kar sakte hain)
SENTENCES = [
    "Hello my name is Ali and this is my voice dataset.",
    "I am learning Python and artificial intelligence.",
    "OpenAI Whisper is an advanced speech recognition model.",
    "Speech to text converts spoken voice into written words.",
    "Today we are training a custom speech model.",
    "The quick brown fox jumps over the lazy dog.",
    "Python is a powerful and easy programming language.",
    "Machine learning models require good quality audio data.",
    "This is my final practical speech to text project.",
    "Thank you for listening to my voice recording."
]

print("="*60)
print("🎙️ AUTOMATED VOICE DATASET CREATOR")
print(f"Total Sentences to record: {len(SENTENCES)}")
print(f"Files will be saved in: {AUDIO_DIR}")
print("="*60)

records = []

for i, sentence in enumerate(SENTENCES, start=1):
    audio_filename = f"audio_{i:03d}.wav"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    relative_audio_path = f"audio/{audio_filename}"

    while True:
        print("\n" + "-"*60)
        print(f"Sentence [{i}/{len(SENTENCES)}]:")
        print(f"\n👉 \"{sentence}\"\n")
        print("-"*60)
        
        input("Press [ENTER] to start recording...")

        # 3, 2, 1 Countdown
        for sec in range(3, 0, -1):
            print(f"Starting in {sec}...", end="\r")
            time.sleep(1)

        print("🔴 RECORDING NOW! Speak clearly into your mic...       ")
        
        # Record audio
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        # Volume Boost / Normalization (Awaaz agar dheemi ho to theek karega)
        max_vol = np.max(np.abs(audio))
        if max_vol > 0.01:
            audio = audio / max_vol
        else:
            print("⚠️ Warning: Bohat dheemi aawaz capture hui hai. Thoda zor se bolein!")

        # Save WAV file (16-bit PCM standard)
        audio_int16 = (audio * 32767).astype(np.int16)
        write(audio_path, SAMPLE_RATE, audio_int16)

        print(f"✅ Saved as {audio_filename}")
        
        choice = input("Press [ENTER] to accept, or type 'r' to re-record: ").strip().lower()
        if choice != 'r':
            records.append({"file_name": relative_audio_path, "sentence": sentence})
            break
        else:
            print("🔄 Re-recording this sentence...")

# CSV Metadata file save karein
print("\nSaving dataset metadata.csv...")
with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file_name", "sentence"])
    writer.writeheader()
    writer.writerows(records)

print("\n" + "="*60)
print("🎉 CONGRATULATIONS! DATASET COMPLETE!")
print(f"Total audio files saved: {len(records)} in '{AUDIO_DIR}/'")
print(f"CSV file created: '{CSV_PATH}'")
print("Aapka personal voice dataset training ke liye bilkul tayyar hai!")
print("="*60)