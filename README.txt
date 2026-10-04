========================================================================
             Python Speech-to-Text with OpenAI Whisper
            Practical AI Project - Student Implementation
========================================================================

1. PROJECT OVERVIEW:
This project demonstrates an end-to-end AI Speech-to-Text application.
Spoken audio is captured through a microphone, digitized into a WAV audio
file, and transcribed into written text with precise timestamps using 
OpenAI's pretrained Whisper neural network.

2. ARCHITECTURE PIPELINE:
Student Speaks -> Microphone -> sounddevice (16kHz PCM) -> Audio WAV ->
Whisper Neural Network (Inference) -> Recognized Text + Timestamps -> GUI

3. PROJECT FILES:
- app_gui.py          : Desktop Graphical User Interface (Exercise 8).
- voice_to_text.py    : Complete command-line voice-to-text pipeline.
- record.py           : Captures microphone input and saves WAV file.
- transcribe.py       : Loads an audio file and transcribes via Whisper.
- create_dataset.py   : Automated dataset creator for voice fine-tuning.
- evaluate_dataset.py : Evaluates Word Error Rate (WER) on custom recordings.
- my_dataset/         : 10 recorded speech clips + metadata.csv.
- README.txt          : Documentation and submission summary.

4. HOW TO RUN:
To launch the GUI Application:
   python app_gui.py

To run the Command-line voice-to-text pipeline:
   python voice_to_text.py

5. CORE AI CONCEPTS COVERED:
- Pretrained Models: Using models already trained on 680,000+ hours of data.
- Inference vs Training:
    * Training: Updates model weights based on data and loss calculation.
    * Inference: Uses fixed weights of a trained model to make predictions.
- Model Sizes:
    * tiny   : 39M parameters (fastest, lightweight).
    * base   : 74M parameters (balanced, recommended starting point).
    * small  : 244M parameters (higher accuracy).
- Word Error Rate (WER):
    WER = (Substitutions + Deletions + Insertions) / Total Reference Words
- Timestamps: Time-aligned sentence segments returned by Whisper decoder.

========================================================================
