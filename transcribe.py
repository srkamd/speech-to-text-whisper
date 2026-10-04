import whisper

print("Loading Whisper...")
model = whisper.load_model("base")
print("Whisper loaded!")

result = model.transcribe("my_voice.wav")

print("Recognized text:")
print(result["text"])