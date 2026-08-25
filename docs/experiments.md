# TinyTurn Experiment Log (E0 – E6)

## Experiment Summary Matrix

| Exp ID | Model Architecture | Temporal Aggregator | Pause Features | Fine-tuning | F1 Score | False END % | Latency (ms) | Size (MB) |
|---|---|---|---|---|---:|---:|---:|---:|
| **E0** | Pause Threshold (700ms) | None | Yes (Rule) | — | 0.685 | 24.5% | 0.05 ms | <0.01 MB |
| **E1** | VAD + Pause MLP | None | Yes | Frozen | 0.792 | 14.2% | 0.85 ms | 0.05 MB |
| **E2** | Acoustic MLP | None | No | Frozen | 0.834 | 9.8% | 3.20 ms | 0.22 MB |
| **E3** | Whisper Tiny + MLP | Mean Pooling | No | Frozen | 0.871 | 6.4% | 28.50 ms | 39.50 MB |
| **E4** | Whisper Tiny + GRU | GRU (64 units) | No | Frozen | 0.892 | 5.2% | 30.10 ms | 39.80 MB |
| **E5** | **Hybrid (Whisper+GRU+Pause)** | **GRU (64 units)** | **Yes (6-dim)** | **Frozen** | **0.915** | **4.1%** | **31.20 ms** | **40.10 MB** |
| **E6** | Hybrid Model (ONNX INT8) | GRU (64 units) | Yes (6-dim) | Quantized | 0.908 | 4.3% | 14.50 ms | 10.40 MB |

---

## Detailed Experiment Logs

### Experiment E0: Fixed Silence Threshold Baseline
- **Hypothesis**: Simple silence thresholds (700ms) fail on Hinglish speech due to mid-sentence code-switching pauses.
- **Results**: F1 = 0.685, False Early-End Rate = 24.5%.
- **Conclusion**: Fixed silence thresholds cause severe premature interruptions during pauses.

### Experiment E1: VAD + Pause Features MLP
- **Hypothesis**: Machine learning on explicit timing features will improve turn decision quality.
- **Results**: F1 = 0.792, False Early-End Rate = 14.2%.
- **Conclusion**: Substantial improvement over rule-based silence, but lacks acoustic/linguistic context.

### Experiment E2: Acoustic MLP Classifier
- **Hypothesis**: MFCCs, RMS energy, spectral centroid/rolloff, and pitch contours provide cues for turn completion.
- **Results**: F1 = 0.834, False Early-End Rate = 9.8%.
- **Conclusion**: Acoustic features effectively capture trailing pitch drops and energy decay.

### Experiment E3: Whisper Tiny Encoder + Mean Pooling
- **Hypothesis**: Pretrained speech representations from Whisper Tiny capture linguistic context.
- **Results**: F1 = 0.871, False Early-End Rate = 6.4%.
- **Conclusion**: Deep speech representations dramatically improve turn completion recognition.

### Experiment E4 & E5: Hybrid Model (Whisper + GRU + Explicit Pause Fusion)
- **Hypothesis**: Combining temporal sequence modeling (GRU) with explicit conversational pause metrics prevents false ends on filler words (*matlab*, *actually*) and mid-sentence pauses.
- **Results**: **F1 = 0.915**, **False Early-End Rate = 4.1%**, **Median Latency = 31.2 ms**.
- **Conclusion**: Optimal architecture balancing high accuracy, low false interruption rate, and low streaming CPU latency.
