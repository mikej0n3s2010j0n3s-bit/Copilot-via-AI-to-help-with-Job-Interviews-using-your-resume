import subprocess
import queue
import sounddevice as sd
import numpy as np
import webrtcvad
import tkinter as tk
import threading
import time
from faster_whisper import WhisperModel

# ======================
# CONFIG
# ======================

MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LLAMA_PATH = "llama.cpp/build/bin/llama-cli"

THREADS = 4
CTX_SIZE = 512
MAX_TOKENS = 120

CHUNK_SECONDS = 5
SAMPLE_RATE = 16000

# ======================
# LOAD RESUME
# ======================

try:
    with open("resume.txt", "r") as f:
        RESUME_TEXT = f.read()
except:
    RESUME_TEXT = ""

# ======================
# LOAD WHISPER (INT8 CPU)
# ======================

whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# ======================
# AUDIO + SILENCE DETECTION
# ======================

vad = webrtcvad.Vad(2)
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def is_speech(audio_chunk):
    pcm = (audio_chunk * 32768).astype(np.int16).tobytes()
    return vad.is_speech(pcm, SAMPLE_RATE)

# ======================
# TINY CONTEXT MEMORY
# ======================

conversation_memory = []

def build_prompt(question):
    memory = "\n".join(conversation_memory[-3:])
    prompt = f"""
You are assisting in a live job interview.

Resume:
{RESUME_TEXT}

Recent context:
{memory}

Question:
{question}

Respond with exactly 3 short bullet points.
Max 30 words total.
"""
    return prompt

# ======================
# LLAMA RUNNER
# ======================

def run_llama(prompt):
    result = subprocess.run(
        [
            LLAMA_PATH,
            "-m", MODEL_PATH,
            "-t", str(THREADS),
            "--ctx-size", str(CTX_SIZE),
            "-n", str(MAX_TOKENS),
            "-p", prompt
        ],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

# ======================
# SMART QUESTION DETECTION
# ======================

def is_question(text):
    keywords = ["tell me", "explain", "describe", "how", "why", "what", "?"]
    return any(k in text.lower() for k in keywords)

# ======================
# MAIN PROCESS LOOP
# ======================

def process_audio():
    buffer = []
    silence_counter = 0

    while True:
        chunk = audio_queue.get()
        buffer.append(chunk)

        if len(buffer) >= SAMPLE_RATE * CHUNK_SECONDS / 1024:
            audio_data = np.concatenate(buffer, axis=0)
            buffer = []

            if is_speech(audio_data):
                silence_counter = 0
            else:
                silence_counter += 1

            # Wait until interviewer stops speaking
            if silence_counter < 2:
                continue

            segments, _ = whisper_model.transcribe(audio_data, beam_size=1)
            text = " ".join([seg.text for seg in segments]).strip()

            if not text:
                continue

            if not is_question(text):
                continue

            conversation_memory.append(text)

            prompt = build_prompt(text)

            answer = run_llama(prompt)

            conversation_memory.append(answer)

            update_gui(answer)

# ======================
# TKINTER FLOATING WINDOW
# ======================

def update_gui(text):
    output_box.config(state="normal")
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, text)
    output_box.config(state="disabled")

root = tk.Tk()
root.title("Interview AI")
root.attributes("-topmost", True)
root.geometry("400x250")

output_box = tk.Text(root, wrap="word", font=("Arial", 11))
output_box.pack(expand=True, fill="both")
output_box.config(state="disabled")

# ======================
# START AUDIO STREAM
# ======================

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    blocksize=1024
)

stream.start()

threading.Thread(target=process_audio, daemon=True).start()

root.mainloop()
