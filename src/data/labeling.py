import numpy as np

def compute_turn_label(
    pause_duration_ms: float,
    future_speech_detected: bool,
    max_pause_threshold_ms: float = 2000.0
) -> int:
    """
    Computes turn completion label:
    - If future speech is detected within context -> CONTINUE (0)
    - If no future speech and pause extends beyond threshold -> END (1)
    """
    if future_speech_detected:
        return 0  # CONTINUE
    else:
        return 1  # END
