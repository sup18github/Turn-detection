import time
import numpy as np
import torch
from src.features.pause import extract_pause_features
from src.features.acoustic import extract_acoustic_features
from src.features.vad import SimpleVAD
from src.inference.smoothing import ProbabilitySmoothing
from src.inference.state_machine import TurnStateMachine

class StreamingTurnDetector:
    """
    Real-Time Streaming Engine:
    Maintains a rolling 2-second audio buffer and processes 100ms audio chunks.
    """
    def __init__(
        self,
        model=None,
        sample_rate: int = 16000,
        buffer_seconds: float = 2.0,
        chunk_ms: int = 100
    ):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_seconds)
        self.chunk_size = int(sample_rate * (chunk_ms / 1000.0))
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)

        self.vad = SimpleVAD(sample_rate=sample_rate)
        self.smoothing = ProbabilitySmoothing()
        self.state_machine = TurnStateMachine()
        self.model = model

    def process_chunk(self, chunk: np.ndarray) -> dict:
        t0 = time.perf_counter()

        # Update rolling buffer
        chunk = chunk.astype(np.float32)
        if len(chunk) >= self.buffer_size:
            self.buffer = chunk[-self.buffer_size:]
        else:
            self.buffer = np.roll(self.buffer, -len(chunk))
            self.buffer[-len(chunk):] = chunk

        # Extract features
        pause_feats = extract_pause_features(self.buffer, self.sample_rate, self.vad)
        silence_ms = pause_feats[0] * 2000.0
        speech_ratio = pause_feats[2]
        is_speech = speech_ratio > 0.15

        # Inference
        if self.model is not None:
            acoust_feats = extract_acoustic_features(self.buffer, self.sample_rate)
            t_acoust = torch.tensor(acoust_feats, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                raw_prob = float(self.model(t_acoust).item())
        else:
            # Fallback heuristic using silence duration & speech ratio
            raw_prob = float(1.0 / (1.0 + np.exp(-(silence_ms - 600.0) / 150.0)))

        smooth_prob = self.smoothing.update(raw_prob)
        state = self.state_machine.process_frame(smooth_prob, is_speech, silence_ms)

        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0

        decision = "END" if state == "END" else "CONTINUE"

        return {
            "decision": decision,
            "state": state,
            "end_probability": round(smooth_prob, 4),
            "raw_probability": round(raw_prob, 4),
            "silence_ms": round(silence_ms, 1),
            "inference_ms": round(inference_ms, 2)
        }
