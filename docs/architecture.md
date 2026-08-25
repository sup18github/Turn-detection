# TinyTurn Technical Architecture Specification

## Overview

TinyTurn process 100ms audio chunks over a rolling 2-second context buffer to output real-time turn completion probabilities `P(END)`.

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

## Component Details

1. **Audio Standardization**: 16 kHz Mono Float32 PCM.
2. **Audio Encoder**: Whisper Tiny Encoder extracting 384-dimensional sequence representations across temporal frames.
3. **Temporal Aggregator**: GRU (64 hidden units) or 1D Conv capturing temporal contours.
4. **Explicit Pause Engine**: Extracts 6 conversational timing features (`current_silence_ms`, `last_speech_duration_ms`, `speech_ratio`, `VAD_probability`, `recent_pause_count`, `energy_slope`).
5. **Feature Fusion Layer**: Concatenates sequence representations (64-dim) with explicit pause features (6-dim) into a 70-dim vector before MLP classification.
6. **State Machine**: States `LISTENING` ➔ `PAUSING` ➔ `END_CANDIDATE` ➔ `END` with 3-frame hysteresis smoothing and minimum silence guard (250ms).
