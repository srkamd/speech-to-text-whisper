import sounddevice as sd
from scipy.io.wavfile import write

sample_rate = 16000
duration = 5  # 5 seconds recording

print("Speak now...")
audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="int16"
)
sd.wait()

write("my_voice.wav", sample_rate, audio)
print("Recording saved as my_voice.wav")