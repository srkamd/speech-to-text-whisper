import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

sample_rate = 16000
duration = 5

print("\n>>> MICROPHONE TEST STARTING...")
print("Boliye: 'Hello testing 1 2 3' zor se mic ke paas bol kar dekhein...")

# Realtek mic ke dono channels capture karein taake silence na aaye
audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=2,
    dtype="float32"
)
sd.wait()

# Awaaz ka volume check karein
max_volume = np.max(np.abs(audio))
print(f"\nCaptured Volume Level: {max_volume:.4f}")

if max_volume < 0.02:
    print("❌ DANGER: Aapka Mic ya to MUTE hai ya aawaz bilkul nahi aa rahi!")
    print("Windows Settings -> System -> Sound -> Microphone volume 100% karein.")
else:
    print("✅ SUCCESS: Aawaz bilkul theek capture hui hai!")

# Save to file
write("my_voice.wav", sample_rate, audio)
print("File 'my_voice.wav' save ho gayi hai. Isay play karke suniye.")