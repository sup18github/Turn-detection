import numpy as np
from src.features.vad import SimpleVAD

def extract_pause_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    vad: SimpleVAD = None
) -> np.ndarray:
    """
    Extracts explicit conversational pause and timing features:
    1. current_silence_ms (ms of trailing silence)
    2. last_speech_duration_ms (ms of preceding speech segment)
    3. speech_ratio (fraction of audio window with active speech)
    4. VAD_probability (mean VAD probability of last 200ms)
    5. recent_pause_count (number of micro-pauses in window)
    6. energy_slope (rate of energy decay into current pause)
    """
    if vad is None:
        vad = SimpleVAD(sample_rate=sample_rate)

    speech_probs = vad.get_speech_probabilities(waveform)
    frame_ms = (vad.frame_length / sample_rate) * 1000.0
    is_speech = speech_probs > 0.5

    # 1. Trailing silence duration
    silence_frames = 0
    for s in reversed(is_speech):
        if not s:
            silence_frames += 1
        else:
            break
    current_silence_ms = silence_frames * frame_ms

    # 2. Last speech segment duration
    speech_frames = 0
    in_last_speech = False
    for s in reversed(is_speech):
        if not s and not in_last_speech:
            continue
        elif s:
            in_last_speech = True
            speech_frames += 1
        elif not s and in_last_speech:
            break
    last_speech_duration_ms = speech_frames * frame_ms

    # 3. Speech ratio
    speech_ratio = np.mean(is_speech.astype(np.float32))

    # 4. VAD probability of trailing 200ms
    num_trailing = max(1, int(200 / frame_ms))
    vad_probability = np.mean(speech_probs[-num_trailing:])

    # 5. Recent pause count
    transitions = np.diff(is_speech.astype(int))
    recent_pause_count = int(np.sum(transitions == -1))

    # 6. Energy slope into pause
    num_frames = len(speech_probs)
    if num_frames >= 5:
        energy_slope = float(speech_probs[-1] - speech_probs[-5])
    else:
        energy_slope = 0.0

    features = np.array([
        current_silence_ms / 2000.0,       # normalized by max 2s
        last_speech_duration_ms / 2000.0,  # normalized by max 2s
        speech_ratio,
        vad_probability,
        float(recent_pause_count) / 5.0,
        energy_slope
    ], dtype=np.float32)

    return features
