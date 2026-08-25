import numpy as np

def extract_context_window(
    waveform: np.ndarray,
    center_sample_idx: int,
    window_seconds: float = 2.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Extracts a fixed context window centered at or ending at a decision point.
    Pads with zeros if window extends beyond bounds.
    """
    window_length = int(window_seconds * sample_rate)
    start_idx = max(0, center_sample_idx - window_length)
    end_idx = center_sample_idx

    chunk = waveform[start_idx:end_idx]

    if len(chunk) < window_length:
        pad_len = window_length - len(chunk)
        chunk = np.pad(chunk, (pad_len, 0), mode='constant', constant_values=0)

    return chunk.astype(np.float32)
