class TurnStateMachine:
    """
    State Machine for Streaming Turn Detection:
    States:
    - LISTENING: Speaker is actively talking
    - PAUSING: Silence/pause detected
    - END_CANDIDATE: High probability turn end under evaluation
    - END: Confirmed turn completion
    """
    def __init__(
        self,
        min_silence_guard_ms: float = 250.0,
        max_silence_timeout_ms: float = 2000.0,
        end_threshold: float = 0.75
    ):
        self.min_silence_guard_ms = min_silence_guard_ms
        self.max_silence_timeout_ms = max_silence_timeout_ms
        self.end_threshold = end_threshold
        self.state = "LISTENING"

    def process_frame(
        self,
        end_prob: float,
        is_speech: bool,
        silence_ms: float
    ) -> str:
        if is_speech:
            self.state = "LISTENING"
            return self.state

        if not is_speech:
            if silence_ms < self.min_silence_guard_ms:
                self.state = "PAUSING"
            elif silence_ms >= self.max_silence_timeout_ms:
                self.state = "END"
            elif end_prob >= self.end_threshold:
                if self.state == "END_CANDIDATE":
                    self.state = "END"
                else:
                    self.state = "END_CANDIDATE"
            else:
                self.state = "PAUSING"

        return self.state
