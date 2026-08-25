import os
import json
import numpy as np
import soundfile as sf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_hinglish_audio(
    duration_s: float,
    sample_rate: int = 16000,
    has_pause: bool = True,
    pause_start_s: float = 1.0,
    pause_duration_s: float = 0.5,
    is_turn_end: bool = False,
    noise_level: float = 0.005,
    filler_type: str = None
) -> np.ndarray:
    """
    Generates synthetic speech waveforms mimicking Hinglish conversational speech,
    fillers ("matlab", "actually", "haan"), mid-sentence pauses, and turn completions.
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    waveform = np.zeros_like(t)

    # Base fundamental frequency variations (human voice formant simulation)
    f0 = 140.0 + 30.0 * np.sin(2 * np.pi * 1.5 * t)
    harmonic1 = 0.4 * np.sin(2 * np.pi * f0 * t)
    harmonic2 = 0.2 * np.sin(2 * np.pi * (2 * f0) * t)
    harmonic3 = 0.1 * np.sin(2 * np.pi * (3 * f0) * t)
    speech_signal = harmonic1 + harmonic2 + harmonic3

    # Apply speech amplitude envelope
    speech_envelope = np.ones_like(t)
    
    if has_pause:
        p_start_idx = int(pause_start_s * sample_rate)
        p_end_idx = int((pause_start_s + pause_duration_s) * sample_rate)
        
        # Apply smooth decay before pause
        fade_len = int(0.05 * sample_rate)
        if p_start_idx > fade_len:
            speech_envelope[p_start_idx - fade_len:p_start_idx] = np.linspace(1.0, 0.0, fade_len)
        speech_envelope[p_start_idx:p_end_idx] = 0.0

        if not is_turn_end and p_end_idx < len(speech_envelope):
            # Speaker continues after pause
            remaining = len(speech_envelope) - p_end_idx
            ramp = min(fade_len, remaining)
            speech_envelope[p_end_idx:p_end_idx + ramp] = np.linspace(0.0, 1.0, ramp)
            speech_envelope[p_end_idx + ramp:] = 1.0
        elif is_turn_end:
            # Silence extends to the end
            speech_envelope[p_end_idx:] = 0.0

    waveform = speech_signal * speech_envelope

    # Filler word acoustic modification
    if filler_type:
        # Add low pitch hum / filler formant characteristic
        filler_env = (t >= (pause_start_s - 0.4)) & (t < pause_start_s)
        waveform[filler_env] += 0.3 * np.sin(2 * np.pi * 110.0 * t[filler_env])

    # Add realistic ambient noise
    noise = np.random.normal(0, noise_level, len(waveform))
    waveform += noise

    # Peak normalization
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.9

    return waveform.astype(np.float32)


def build_hinglish_dataset(num_samples: int = 400, sample_rate: int = 16000):
    """
    Generates a balanced dataset of Hinglish turn detection samples:
    - Normal completion (END = 1)
    - Mid-sentence pause (CONTINUE = 0)
    - Filler before pause (CONTINUE = 0)
    - Short answer (END = 1)
    - Long hesitation (CONTINUE = 0)
    - Self-correction (CONTINUE = 0)
    """
    metadata = []
    fillers = ["matlab", "actually", "haan", "like", "basically", "ek_second"]
    languages = ["hinglish", "hindi", "english"]

    print(f"Generating {num_samples} Hinglish conversational audio samples...")

    for i in range(num_samples):
        speaker_id = f"speaker_{i % 25:03d}"
        conv_id = f"conv_{i % 50:03d}"
        sample_id = f"hinglish_sample_{i:04d}"

        # Determine scenario
        scenario_type = i % 6
        duration_s = 2.0  # standard 2-second context window

        if scenario_type == 0:
            # Normal completion (END)
            is_turn_end = True
            has_pause = True
            pause_start_s = 1.0
            pause_duration_s = 1.0
            category = "normal_ending"
            filler = None
            text = "Mujhe ek cab book karni hai."
        elif scenario_type == 1:
            # Mid-sentence pause (CONTINUE)
            is_turn_end = False
            has_pause = True
            pause_start_s = 0.8
            pause_duration_s = 0.6
            category = "mid_sentence_pause"
            filler = None
            text = "Mujhe ek cab book karni hai... airport ke liye."
        elif scenario_type == 2:
            # Filler word pause (CONTINUE)
            is_turn_end = False
            has_pause = True
            pause_start_s = 1.1
            pause_duration_s = 0.5
            category = "filler"
            filler = fillers[i % len(fillers)]
            text = f"Mujhe {filler}... meeting reschedule karni hai."
        elif scenario_type == 3:
            # Short answer (END)
            is_turn_end = True
            has_pause = True
            pause_start_s = 0.4
            pause_duration_s = 1.6
            category = "short_answer"
            filler = None
            text = "Haan."
        elif scenario_type == 4:
            # Long thinking pause (CONTINUE)
            is_turn_end = False
            has_pause = True
            pause_start_s = 0.6
            pause_duration_s = 1.0
            category = "long_hesitation"
            filler = None
            text = "Mujhe... ek minute do."
        else:
            # Self correction (CONTINUE)
            is_turn_end = False
            has_pause = True
            pause_start_s = 0.9
            pause_duration_s = 0.5
            category = "self_correction"
            filler = "actually"
            text = "Mujhe Delhi jaana hai... nahi, actually Mumbai jaana hai."

        audio = generate_synthetic_hinglish_audio(
            duration_s=duration_s,
            sample_rate=sample_rate,
            has_pause=has_pause,
            pause_start_s=pause_start_s,
            pause_duration_s=pause_duration_s,
            is_turn_end=is_turn_end,
            noise_level=0.005 + 0.005 * (i % 3),
            filler_type=filler
        )

        audio_path = RAW_DIR / f"{sample_id}.wav"
        sf.write(str(audio_path), audio, sample_rate)

        # Label logic: 1 for END, 0 for CONTINUE
        label = 1 if is_turn_end else 0

        metadata.append({
            "sample_id": sample_id,
            "file_path": str(audio_path),
            "duration": duration_s,
            "sample_rate": sample_rate,
            "speaker_id": speaker_id,
            "conversation_id": conv_id,
            "category": category,
            "language": languages[i % len(languages)],
            "transcript": text,
            "filler": filler,
            "pause_start_s": pause_start_s,
            "pause_duration_s": pause_duration_s,
            "label": label,
            "label_name": "END" if label == 1 else "CONTINUE"
        })

    manifest_path = PROCESSED_DIR / "dataset_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset generated with {len(metadata)} samples. Manifest: {manifest_path}")
    return metadata


if __name__ == "__main__":
    build_hinglish_dataset()
