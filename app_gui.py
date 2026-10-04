import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sounddevice as sd
from scipy.io.wavfile import write
import whisper

# Global variables
SAMPLE_RATE = 16000
DURATION = 5  # 5 seconds recording
is_processing = False

# Function to record audio and run Whisper in a background thread
def process_speech():
    global is_processing
    is_processing = True
    
    btn_record.config(state="disabled")
    lbl_status.config(text="🔴 Recording... Speak now!", fg="#d9534f")
    root.update()

    try:
        # 1. Record Audio
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16"
        )
        sd.wait()
        
        write("gui_recording.wav", SAMPLE_RATE, audio)
        
        # 2. Transcribe
        lbl_status.config(text="⏳ AI is transcribing with Whisper...", fg="#f0ad4e")
        root.update()

        model_name = combo_model.get()
        model = whisper.load_model(model_name)
        
        result = model.transcribe("gui_recording.wav", fp16=False)
        
        # 3. Display Results
        txt_output.delete("1.0", tk.END)
        txt_output.insert(tk.END, "=== FINAL TRANSCRIPTION ===\n")
        txt_output.insert(tk.END, result["text"].strip() + "\n\n")
        
        txt_output.insert(tk.END, "=== DETAILED TIMESTAMPS ===\n")
        for seg in result["segments"]:
            start = seg["start"]
            end = seg["end"]
            text = seg["text"]
            txt_output.insert(tk.END, f"[{start:.2f}s -> {end:.2f}s] {text}\n")
            
        lbl_status.config(text="✅ Transcription Complete!", fg="#5cb85c")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        lbl_status.config(text="❌ Error occurred", fg="red")
        
    finally:
        btn_record.config(state="normal")
        is_processing = False

# Thread wrapper so UI doesn't freeze
def start_thread():
    if not is_processing:
        t = threading.Thread(target=process_speech, daemon=True)
        t.start()

# Clear Text box
def clear_text():
    txt_output.delete("1.0", tk.END)
    lbl_status.config(text="Ready", fg="#333333")

# --- GUI WINDOW SETUP ---
root = tk.Tk()
root.title("Python Speech-to-Text with Whisper (AI)")
root.geometry("620x580")
root.resizable(False, False)
root.configure(bg="#f4f6f9")

# Header Title
lbl_title = tk.Label(
    root, 
    text="🎙️ Speech-to-Text AI Application", 
    font=("Segoe UI", 16, "bold"), 
    bg="#f4f6f9", 
    fg="#2c3e50"
)
lbl_title.pack(pady=15)

# Controls Frame
frame_controls = tk.Frame(root, bg="#f4f6f9")
frame_controls.pack(pady=5)

tk.Label(frame_controls, text="Select Whisper Model:", font=("Segoe UI", 10), bg="#f4f6f9").grid(row=0, column=0, padx=5, pady=5)
combo_model = ttk.Combobox(frame_controls, values=["base", "tiny"], state="readonly", width=10)
combo_model.set("base")
combo_model.grid(row=0, column=1, padx=5, pady=5)

# Status Label
lbl_status = tk.Label(
    root, 
    text="Ready", 
    font=("Segoe UI", 12, "bold"), 
    bg="#f4f6f9", 
    fg="#333333"
)
lbl_status.pack(pady=10)

# Buttons Frame
frame_buttons = tk.Frame(root, bg="#f4f6f9")
frame_buttons.pack(pady=5)

btn_record = tk.Button(
    frame_buttons, 
    text="🎙️ Start Recording (5s)", 
    command=start_thread,
    font=("Segoe UI", 11, "bold"), 
    bg="#0275d8", 
    fg="white", 
    padx=15, 
    pady=8,
    relief="flat",
    cursor="hand2"
)
btn_record.grid(row=0, column=0, padx=10)

btn_clear = tk.Button(
    frame_buttons, 
    text="Clear Output", 
    command=clear_text,
    font=("Segoe UI", 10), 
    bg="#e0e0e0", 
    padx=10, 
    pady=8,
    relief="flat",
    cursor="hand2"
)
btn_clear.grid(row=0, column=1, padx=10)

# Output Text Area
lbl_out = tk.Label(root, text="Recognized Output:", font=("Segoe UI", 10, "bold"), bg="#f4f6f9", fg="#2c3e50")
lbl_out.pack(anchor="w", padx=30, pady=(15, 5))

txt_output = tk.Text(root, wrap="word", font=("Consolas", 10), height=14, width=68, bg="white", relief="solid", bd=1)
txt_output.pack(padx=30, pady=5)

# Footer
lbl_footer = tk.Label(root, text="OpenAI Whisper • sounddevice • SciPy", font=("Segoe UI", 8), bg="#f4f6f9", fg="#888888")
lbl_footer.pack(side="bottom", pady=10)

# Start Application
root.mainloop()