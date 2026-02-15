# 🎙️ Interview AI Assistant (TinyLlama + Whisper)

Lightweight real-time interview assistant designed to run on older hardware.

Optimized for:
- 2011 iMac (i5 Sandy Bridge)
- 16GB RAM
- CPU-only inference

No Ollama. No cloud. Fully local.

---

## 🚀 Features

- 🎤 Real-time audio listening
- 🧠 Whisper (int8 CPU) transcription
- ⚡ TinyLlama (1.1B GGUF) via llama.cpp
- 🛑 Silence detection (waits for interviewer to finish speaking)
- ❓ Smart question detection
- 📄 Resume ingestion
- 🧩 Tiny rolling memory (last 3 exchanges)
- 🪟 Floating Tkinter window
- 🔒 Fully offline

---

## 🏗️ Architecture

Audio → Silence Detection → Whisper → Question Detection → TinyLlama → Bullet Output → Floating UI

No server layer.  
No HTTP calls.  
Direct llama.cpp subprocess execution.

---

## 📂 Project Structure

