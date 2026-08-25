import numpy as np
import librosa

def extract_acoustic_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160
) -> np.ndarray:
    """
    Extracts acoustic feature vectors:
    - 13 MFCCs (mean & std across frames)
    - RMS Energy (mean & max)
    - Spectral Centroid (mean)
    - Spectral Rolloff (mean)
    - Zero Crossing Rate (mean)
    - Energy Slope
    Returns a 1D feature vector of shape (~30,).
    """
    if len(waveform) < n_fft:
        waveform = np.pad(waveform, (0, n_fft - len(waveform)))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # RMS Energy
    rms = librosa.feature.rms(y=waveform, frame_length=n_fft, hop_length=hop_length)[0]
    rms_mean = np.mean(rms)
    rms_max = np.max(rms)

    # Spectral features
    centroid = np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0])
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0])

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=waveform, frame_length=n_fft, hop_length=hop_length)[0])

    # Energy Slope
    energy_slope = (rms[-1] - rms[0]) / (len(rms) + 1e-6)

    feature_vector = np.concatenate([
        mfcc_mean,
        mfcc_std,
        np.array([rms_mean, rms_max, centroid / 4000.0, rolloff / 8000.0, zcr, energy_slope], dtype=np.float32)
    ])

    return feature_vector.astype(np.float32)
