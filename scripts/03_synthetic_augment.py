"""
03_synthetic_augment.py

Synthesizes balanced INCOMPLETE_TURN / TURN_COMPLETE clips from clean,
single-sentence utterances (e.g. Common Voice `hi`, Svarah) by injecting
silence at different points:

  - INCOMPLETE_TURN: cut a word boundary somewhere in the first 30-70% of
    the utterance, inject 200-800ms of silence, TRUNCATE right after the
    injected silence. This mimics the exact moment a live agent has to
    decide "has the user stopped, or are they just thinking?"
  - TURN_COMPLETE: keep the whole utterance, append 200-800ms of trailing
    silence at the genuine end.
  - Optional hard negatives: splice a real filler-word audio snippet (from
    `--filler_bank`) into the mid-sentence cut point instead of pure
    silence.

Requires word-level timestamps to pick sensible cut points -- reuses
faster-whisper the same way 02_segment_turns.py does, so cuts land on word
boundaries rather than mid-phoneme.

Usage:
    python 03_synthetic_augment.py \
        --in data/cv_hi_manifest.jsonl \
        --out manifests/synthetic.jsonl \
        --clips_dir clips_synth/ \
        --filler_bank fillers/ \
        --n_per_utterance 2
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from tqdm import tqdm

TARGET_SR = 16000
SILENCE_MS_RANGE = (200, 800)
MID_CUT_RATIO_RANGE = (0.3, 0.7)


def load_and_resample(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0).numpy(), TARGET_SR


def to_audio_segment(y, sr=TARGET_SR):
    y_int16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
    return AudioSegment(
        y_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1,
    )


def from_audio_segment(seg):
    y = np.array(seg.get_array_of_samples()).astype(np.float32) / 32767.0
    return y, seg.frame_rate


def pick_mid_cut_sample(words, total_duration_s):
    """Pick a word-boundary cut point within MID_CUT_RATIO_RANGE of the utterance."""
    lo, hi = MID_CUT_RATIO_RANGE
    candidates = [
        w["end"] for w in words
        if lo * total_duration_s <= w["end"] <= hi * total_duration_s
    ]
    if not candidates:
        # fall back to a raw ratio cut if word timestamps didn't land in range
        return random.uniform(lo, hi) * total_duration_s
    return random.choice(candidates)


def load_filler_bank(filler_dir):
    if not filler_dir or not Path(filler_dir).exists():
        return []
    return sorted(Path(filler_dir).glob("*.wav"))


def build_incomplete_clip(y, sr, words, filler_bank, use_filler_prob=0.3):
    total_duration_s = len(y) / sr
    cut_s = pick_mid_cut_sample(words, total_duration_s)
    cut_sample = int(cut_s * sr)
    head = y[:cut_sample]

    seg = to_audio_segment(head, sr)
    silence_ms = random.randint(*SILENCE_MS_RANGE)
    filler_word = None

    if filler_bank and random.random() < use_filler_prob:
        filler_path = random.choice(filler_bank)
        filler_y, filler_sr = load_and_resample(str(filler_path))
        filler_seg = to_audio_segment(filler_y, filler_sr)
        seg = seg + filler_seg + AudioSegment.silent(duration=silence_ms // 2)
        filler_word = filler_path.stem
    else:
        seg = seg + AudioSegment.silent(duration=silence_ms)

    out_y, out_sr = from_audio_segment(seg)
    return out_y, out_sr, silence_ms, filler_word


def build_complete_clip(y, sr):
    seg = to_audio_segment(y, sr)
    silence_ms = random.randint(*SILENCE_MS_RANGE)
    seg = seg + AudioSegment.silent(duration=silence_ms)
    out_y, out_sr = from_audio_segment(seg)
    return out_y, out_sr, silence_ms


def get_word_timestamps(audio_path, whisper_model):
    segments, _ = whisper_model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        if seg.words:
            words.extend({"word": w.word.strip(), "start": w.start, "end": w.end} for w in seg.words)
    return words


def process_row(row, whisper_model, clips_dir, filler_bank, n_per_utterance):
    audio_path = row.get("audio_path")
    transcript = row.get("transcript", "")
    if not audio_path or not Path(audio_path).exists():
        return []

    y, sr = load_and_resample(audio_path)
    if len(y) < int(0.5 * sr):
        return []

    try:
        words = get_word_timestamps(audio_path, whisper_model)
    except Exception:
        words = []

    out_rows = []
    stem = Path(audio_path).stem

    for k in range(n_per_utterance):
        # --- negative (incomplete) ---
        neg_y, neg_sr, silence_ms, filler_word = build_incomplete_clip(y, sr, words, filler_bank)
        neg_path = Path(clips_dir) / f"{stem}_neg{k}.wav"
        neg_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(neg_path, neg_y, neg_sr)
        out_rows.append({
            "audio_path": str(neg_path),
            "duration": round(len(neg_y) / neg_sr, 3),
            "label": "INCOMPLETE_TURN",
            "filler_flag": filler_word is not None,
            "filler_word": filler_word,
            "pause_ms": silence_ms,
            "f0_slope_hz_per_s": None,   # compute downstream if needed
            "energy_decay_db": None,
            "transcript": transcript,
            "source_dataset": f"synthetic:{row.get('source_dataset', 'unknown')}",
            "language_tag": row.get("language_tag"),
            "split": "train",
            "synthetic": True,
        })

        # --- positive (complete) ---
        pos_y, pos_sr, silence_ms = build_complete_clip(y, sr)
        pos_path = Path(clips_dir) / f"{stem}_pos{k}.wav"
        sf.write(pos_path, pos_y, pos_sr)
        out_rows.append({
            "audio_path": str(pos_path),
            "duration": round(len(pos_y) / pos_sr, 3),
            "label": "TURN_COMPLETE",
            "filler_flag": False,
            "filler_word": None,
            "pause_ms": silence_ms,
            "f0_slope_hz_per_s": None,
            "energy_decay_db": None,
            "transcript": transcript,
            "source_dataset": f"synthetic:{row.get('source_dataset', 'unknown')}",
            "language_tag": row.get("language_tag"),
            "split": "train",
            "synthetic": True,
        })

    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--filler_bank", default=None,
                     help="Optional dir of short filler-word .wav clips (umm/arre/matlab/...) for hard negatives")
    ap.add_argument("--n_per_utterance", type=int, default=2)
    ap.add_argument("--whisper_model", default="tiny")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(args.whisper_model, device=args.device, compute_type="int8")

    filler_bank = load_filler_bank(args.filler_bank)
    if args.filler_bank and not filler_bank:
        print(f"[warn] --filler_bank given but no .wav files found in {args.filler_bank}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.in_path, encoding="utf-8")]

    total = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for row in tqdm(rows, desc="augmenting"):
            for out_row in process_row(row, whisper_model, args.clips_dir, filler_bank, args.n_per_utterance):
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                total += 1

    print(f"Done. {total} synthetic clips -> {args.out}")


if __name__ == "__main__":
    main()
