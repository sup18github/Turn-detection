# TinyTurn — Hinglish Audio Turn Detection System
## Product Requirements Document (PRD)

**Project Type:** Audio ML / Conversational AI Infrastructure
**Primary Goal:** Tiny, fast, and accurate turn detection for Indian Hinglish speech
**Target Interface:** Interactive Streamlit Application
**Model Direction:** Whisper Tiny / lightweight audio encoder + temporal model + pause/acoustic features
**Deployment Goal:** Real-time CPU-capable streaming inference (<500 ms latency)

---

### Core Objectives
1. Prevent **False Early-End** interruptions during speaker hesitations, fillers (*matlab*, *actually*, *haan*), and code-switching.
2. Maintain low **End Detection Latency** for genuine turn completions.
3. Compare rule-based silence baselines, acoustic MLPs, Whisper Tiny classifiers, and a proposed Hybrid Model (Whisper + GRU + Explicit Pause Feature Fusion).
4. Provide zero-leakage speaker/conversation evaluation and slice-based error analysis.
5. Deliver an interactive 4-section Streamlit playground for real-time visualization, waveform timelines, model insights, and benchmarking.
