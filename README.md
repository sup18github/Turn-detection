# TinyTurn — Hinglish Audio Turn Detection System

**TinyTurn** is a tiny, fast, and accurate audio-based turn detection model built specifically for voice AI infrastructure handling **Indian Hinglish speech**, filler words (*matlab*, *actually*, *haan*, *basically*), and mid-sentence code-switching pauses.

---

## Key Features
- **Low-Latency Streaming**: 100ms real-time chunking with 2-second rolling audio buffer (<35 ms inference time).
- **Hinglish & Filler Robustness**: Dual-branch hybrid architecture combining Whisper Tiny sequence representations + GRU temporal aggregator + explicit pause/VAD features.
- **Low False Early-End Rate**: Reduces premature interruptions from **24.5% (Silence Baseline)** down to **4.1% (Hybrid Model)**.
- **Interactive Streamlit Playground**: 4-section UI featuring live state indicators, probability gauges, interactive audio timeline plots, model parameter insights, and slice-filtered benchmark dashboards.
- **Zero-Leakage Dataset Pipeline**: Automated download, auditing, speech standardization (16 kHz mono float32), and strict speaker-isolated train/val/test splits.

---

## Model Benchmark Overview

| Model Architecture | F1 Score | Precision | Recall | False END % | Latency (ms) | Size (MB) | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pause Threshold (700ms) | 0.685 | 0.621 | 0.763 | 24.5% | 0.05 ms | <0.01 MB | 0.0001 |
| VAD + Pause MLP | 0.792 | 0.754 | 0.834 | 14.2% | 0.85 ms | 0.05 MB | 0.0008 |
| Acoustic MLP | 0.834 | 0.812 | 0.857 | 9.8% | 3.20 ms | 0.22 MB | 0.0032 |
| Whisper Tiny Encoder | 0.871 | 0.865 | 0.877 | 6.4% | 28.50 ms | 39.50 MB | 0.0285 |
| **Hybrid (Whisper+GRU+Pause)** | **0.915** | **0.908** | **0.922** | **4.1%** | **31.20 ms** | **40.10 MB** | **0.0312** |

---

## Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/your-org/tiny-turn-detector.git
cd tiny-turn-detector
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Pipeline & Audit
```bash
python src/data/download.py   # Generate/download dataset
python src/data/audit.py      # Run audit and export dataset_report.json
python src/data/split.py      # Generate 0-leakage speaker splits
```

### 3. Model Training
```bash
python src/training/train.py configs/hybrid.yaml
```

### 4. Evaluation & Benchmarking
```bash
python src/evaluation/evaluate.py
```

### 5. Launch Interactive Streamlit Playground
```bash
streamlit run demo/app.py
```

---

## System Architecture

```
                    AUDIO STREAM (100ms Chunks)
                                │
                        Rolling 2s Buffer
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
     Whisper Tiny Encoder                 Silero / Energy VAD
             │                                     │
     Temporal Model (GRU)                Pause & Acoustic Features
  (Context Representations)            (Silence ms, Speech Ratio, Energy)
             │                                     │
             └──────────────────┬──────────────────┘
                                ▼
                       Feature Fusion Layer
                                │
                          MLP Classifier
                                │
                             P(END)
                                │
                 Hysteresis & Smoothing Engine
                                │
                          State Machine
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
             CONTINUE                         END
```

---

## Repository Structure
```
tiny-turn-detector/
├── configs/          # YAML experiment configurations
├── data/             # Raw, processed, train/val/test jsonl manifests
├── src/
│   ├── data/         # Ingestion, audit, standardization, splits
│   ├── features/     # VAD, acoustic, and explicit pause extraction
│   ├── models/       # Rule baseline, Acoustic MLP, Whisper, Hybrid Model
│   ├── training/     # Loss functions, callbacks, training loops
│   ├── evaluation/   # Metrics, latency benchmarks, slice error analysis
│   └── inference/    # Real-time streaming, state machine, REST API server
├── demo/             # Interactive Streamlit playground app
├── results/          # Benchmark matrices, reports, ONNX checkpoints
└── docs/             # PRD, Architecture, Experiment logs
```

---

## 🔗 Live Application & Links

- **GitHub Repository (Public)**: [https://github.com/sup18github/Turn-detection](https://github.com/sup18github/Turn-detection)
- **Direct 1-Click Live Application**: [https://46513bc7b9d57c.lhr.life](https://46513bc7b9d57c.lhr.life) *(Direct HTTPS link for anyone over the internet)*
- **Local Application**: [http://localhost:8501](http://localhost:8501)

