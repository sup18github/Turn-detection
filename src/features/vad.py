import numpy as np

class SimpleVAD:
    """
    Lightweight, fast Voice Activity Detector based on Short-Time Energy
    and frame-wise RMS.
    """
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 20, energy_threshold: float = 0.015):
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_duration_ms / 1000.0))
        self.energy_threshold = energy_threshold

    def get_speech_probabilities(self, waveform: np.ndarray) -> np.ndarray:
        """
        Computes frame-wise speech probability scores [0.0, 1.0].
        """
        num_frames = max(1, len(waveform) // self.frame_length)
        probs = []

        for i in range(num_frames):
            frame = waveform[i * self.frame_length : (i + 1) * self.frame_length]
            if len(frame) == 0:
                probs.append(0.0)
                continue

            rms = np.sqrt(np.mean(frame**2) + 1e-8)
            # Sigmoidal mapping of RMS relative to threshold
            prob = 1.0 / (1.0 + np.exp(-150.0 * (rms - self.energy_threshold)))
            probs.append(float(prob))

        return np.array(probs, dtype=np.float32)
