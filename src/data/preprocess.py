import numpy as np

def standardize_audio(
    waveform: np.ndarray,
    orig_sr: int = 16000,
    target_sr: int = 16000,
    target_peak: float = 0.95
) -> np.ndarray:
    """
    Resamples, converts to mono, normalizes float32 audio.
    """
    # Mono conversion if 2D array
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)

    waveform = waveform.astype(np.float32)

    # Peak normalization
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * target_peak

    return waveform


def apply_audio_augmentation(
    waveform: np.ndarray,
    noise_factor: float = 0.003,
    speed_factor: float = 1.0,
    gain_db: float = 0.0
) -> np.ndarray:
    """
    Applies audio augmentations for training robustness.
    """
    aug_waveform = waveform.copy()

    # Gain variation
    if gain_db != 0.0:
        factor = 10.0 ** (gain_db / 20.0)
        aug_waveform = aug_waveform * factor

    # Add Gaussian noise
    if noise_factor > 0:
        noise = np.random.normal(0, noise_factor, len(aug_waveform))
        aug_waveform += noise

    # Clip to float32 range
    aug_waveform = np.clip(aug_waveform, -1.0, 1.0)
    return aug_waveform.astype(np.float32)
