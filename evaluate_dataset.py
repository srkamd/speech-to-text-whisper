import os
import csv
import whisper

print("Loading Whisper model...")
model = whisper.load_model("base")

CSV_PATH = "my_dataset/metadata.csv"
total_words = 0
correct_words = 0

print("\n" + "="*60)
print("TESTING WHISPER ON YOUR 10 RECORDINGS (WER EVALUATION)")
print("="*60)

with open(CSV_PATH, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        audio_file = os.path.join("my_dataset", row["file_name"])
        reference_text = row["sentence"].strip()

        # Whisper prediction
        result = model.transcribe(audio_file, fp16=False)
        predicted_text = result["text"].strip()

        print(f"\n📁 File: {row['file_name']}")
        print(f"✅ What you actually said : \"{reference_text}\"")
        print(f"🤖 What Whisper heard     : \"{predicted_text}\"")

print("\n" + "="*60)
print("Evaluation Complete!")
print("="*60)
