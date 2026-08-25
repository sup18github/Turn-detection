import numpy as np

class ProbabilitySmoothing:
    """
    Smoothing and hysteresis layer for streaming turn detection.
    Maintains a rolling window of recent frame P(END) predictions.
    """
    def __init__(self, window_size: int = 3, high_threshold: float = 0.75, low_threshold: float = 0.45):
        self.window_size = window_size
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.history = []

    def update(self, prob: float) -> float:
        self.history.append(prob)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        smooth_prob = float(np.mean(self.history))
        return smooth_prob

    def check_hysteresis(self, smooth_prob: float, current_candidate: bool) -> bool:
        if current_candidate:
            # Stay candidate unless smooth_prob drops below low_threshold
            return smooth_prob >= self.low_threshold
        else:
            # Become candidate only if smooth_prob exceeds high_threshold
            return smooth_prob >= self.high_threshold
