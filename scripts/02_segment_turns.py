"""
02_segment_turns.py

Turns full-utterance audio (from 01_prepare_datasets.py's raw manifest) into
short INCOMPLETE_TURN / TURN_COMPLETE clips by:

  1. Getting word-level timestamps (faster-whisper, word_timestamps=True).
  2. Finding pause candidates (inter-word gaps >= MIN_PAUSE_MS).
  3. Weak-labeling each candidate using the Hinglish filler/continuation
     lexicon + pause duration + F0 trajectory + trailing energy decay.
  4. Cutting a clip ending at the pause boundary (with preceding context)
     and writing a manifest row.

This produces WEAK labels, not ground truth. Sample and manually verify a
few hundred per class before trusting this as an eval set (see README §6).

Usage:
    python 02_segment_turns.py \
        --in data/raw_manifest.jsonl \
        --out manifests/mined.jsonl \
        --clips_dir clips/ \
        --context_s 2.5
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from lexicons.hinglish_lexicon import is_continuation, is_filler, is_sentence_final

TARGET_SR = 16000          # Whisper-Tiny native input rate
MIN_PAUSE_MS = 200         # below this, don't even consider it a candidate boundary
SHORT_PAUSE_MS = 500       # 200-500ms -> hesitation range
LONG_PAUSE_MS = 600        # >=600ms -> candidate terminal silence
TRAILING_WINDOW_S = 0.4    # window for F0 / energy features before the cut


def get_word_timestamps(audio_path, whisper_model):
    """Return list of {'word': str, 'start': float, 'end': float} via faster-whisper."""
    segments, _ = whisper_model.transcribe(
        audio_path, word_timestamps=True, language=None, vad_filter=True,
    )
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    return words


def f0_slope(y, sr):
    """Rough F0 trajectory slope (Hz/s) over a short window. +ve = rising pitch."""
    if len(y) < int(0.05 * sr):
        return 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr,
        )
    except Exception:
        return 0.0
    f0 = f0[~np.isnan(f0)]
    if len(f0) < 3:
        return 0.0
    t = np.linspace(0, len(y) / sr, len(f0))
    slope = np.polyfit(t, f0, 1)[0]
    return float(slope)


def energy_decay_db(y, sr):
    """RMS energy change (dB) from first half to second half of the window."""
    if len(y) < int(0.05 * sr):
        return 0.0
    mid = len(y) // 2
    rms_first = np.sqrt(np.mean(y[:mid] ** 2)) + 1e-8
    rms_second = np.sqrt(np.mean(y[mid:] ** 2)) + 1e-8
    return float(20 * np.log10(rms_second / rms_first))


def classify_pause(prev_word, next_word, pause_ms, is_last_word):
    """
    Weak-label heuristic. Returns (label, filler_flag, filler_word) or
    (None, None, None) if the candidate is too ambiguous to keep.
    """
    prev_is_filler = is_filler(prev_word)
    next_is_continuation = next_word is not None and is_continuation(next_word)
    prev_is_sentence_final = is_sentence_final(prev_word)

    # Strong negative: filler word right before the pause.
    if prev_is_filler and pause_ms >= MIN_PAUSE_MS:
        return "INCOMPLETE_TURN", True, prev_word

    # Strong negative: short/medium pause immediately followed by a
    # continuation word -> the clause keeps going.
    if SHORT_PAUSE_MS <= pause_ms < LONG_PAUSE_MS and next_is_continuation:
        return "INCOMPLETE_TURN", False, None

    # Strong positive: long pause at/near a sentence-final marker, or at the
    # true end of the utterance (no next word).
    if pause_ms >= LONG_PAUSE_MS and (prev_is_sentence_final or is_last_word):
        return "TURN_COMPLETE", False, None

    # Ambiguous mid-range case with no strong lexical cue -> let acoustic
    # features decide downstream; mark as undecided here.
    if pause_ms >= LONG_PAUSE_MS and is_last_word:
        return "TURN_COMPLETE", False, None

    return None, None, None


def process_row(row, whisper_model, clips_dir, context_s):
    audio_path = row.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        return []

    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"[warn] failed to load {audio_path}: {e}", file=sys.stderr)
        return []

    words = get_word_timestamps(audio_path, whisper_model)
    if len(words) < 2:
        return []

    out_rows = []
    for i, w in enumerate(words):
        is_last = i == len(words) - 1
        next_word = words[i + 1]["word"] if not is_last else None
        pause_ms = int((words[i + 1]["start"] - w["end"]) * 1000) if not is_last else int(
            (len(y) / sr - w["end"]) * 1000
        )
        if pause_ms < MIN_PAUSE_MS:
            continue

        label, filler_flag, filler_word = classify_pause(w["word"], next_word, pause_ms, is_last)
        if label is None:
            continue

        cut_sample = int(w["end"] * sr)
        start_sample = max(0, cut_sample - int(context_s * sr))
        clip = y[start_sample:cut_sample]
        if len(clip) < int(0.3 * sr):  # skip clips that are too short to be useful
            continue

        trail_start = max(0, len(clip) - int(TRAILING_WINDOW_S * sr))
        trail = clip[trail_start:]

        clip_id = f"{Path(audio_path).stem}_{i:04d}"
        out_path = Path(clips_dir) / f"{clip_id}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, clip, TARGET_SR)

        transcript_snippet = " ".join(x["word"] for x in words[: i + 1])

        out_rows.append({
            "audio_path": str(out_path),
            "duration": round(len(clip) / TARGET_SR, 3),
            "label": label,
            "filler_flag": filler_flag,
            "filler_word": filler_word,
            "pause_ms": pause_ms,
            "f0_slope_hz_per_s": round(f0_slope(trail, TARGET_SR), 2),
            "energy_decay_db": round(energy_decay_db(trail, TARGET_SR), 2),
            "transcript": transcript_snippet,
            "source_dataset": row.get("source_dataset"),
            "language_tag": row.get("language_tag"),
            "split": "train",
            "synthetic": False,
        })
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--context_s", type=float, default=2.5)
    ap.add_argument("--whisper_model", default="tiny")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(args.whisper_model, device=args.device, compute_type="int8")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.in_path, encoding="utf-8")]

    total = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for row in tqdm(rows, desc="segmenting"):
            for out_row in process_row(row, whisper_model, args.clips_dir, args.context_s):
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                total += 1

    print(f"Done. {total} labeled clips -> {args.out}")


if __name__ == "__main__":
    main()
