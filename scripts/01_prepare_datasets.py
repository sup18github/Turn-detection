"""
01_prepare_datasets.py

Discovers Hindi / Hinglish / Indian-English configs across the verified
source datasets and writes a normalized "raw manifest" (one row per source
utterance, full-length, with transcript) that 02_segment_turns.py consumes.

We deliberately do NOT hardcode config/split names for the multi-language
datasets (IndicVoices, Vaani) — dataset maintainers occasionally rename
configs between releases, so we query `get_dataset_config_names` at runtime
and filter by keyword. Always sanity-check the printed config list against
each dataset's card on first run.

Usage:
    python 01_prepare_datasets.py \
        --out data/raw_manifest.jsonl \
        --sources indicvoices vaani svarah common_voice \
        --max_per_source 5000 \
        --streaming

Datasets referenced (verify current license/gating status on the HF page
before large-scale download):
    ai4bharat/IndicVoices          (gated: none, CC BY 4.0)
    ARTPARK-IISc/Vaani             (gated: accept terms on HF first)
    ai4bharat/Svarah               (CC BY 4.0)
    ai4bharat/Lahaja               (gated: accept terms on HF first)
    mozilla-foundation/common_voice_11_0  (config "hi", CC0)
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm

LANGUAGE_KEYWORDS = {
    "hi", "hindi", "hin", "en-in", "en_in", "indian", "english", "eng",
}


def matching_configs(dataset_id: str, keywords=LANGUAGE_KEYWORDS):
    """Return configs whose name contains any of `keywords` (case-insensitive)."""
    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception as e:
        print(f"[warn] could not list configs for {dataset_id}: {e}", file=sys.stderr)
        return []
    matched = [c for c in configs if any(k in c.lower() for k in keywords)]
    print(f"[{dataset_id}] {len(configs)} total configs, {len(matched)} matched: {matched}")
    return matched


def normalize_row(example, source_dataset, language_tag, text_key="text"):
    """Map a HF dataset example into our common raw-manifest schema."""
    audio = example.get("audio") or example.get("audio_filepath")
    text = example.get(text_key) or example.get("sentence") or example.get("transcription") or ""
    if audio is None or not text:
        return None
    return {
        "audio_array": audio.get("array") if isinstance(audio, dict) else None,
        "audio_path": audio.get("path") if isinstance(audio, dict) else audio,
        "sampling_rate": audio.get("sampling_rate") if isinstance(audio, dict) else None,
        "transcript": text.strip(),
        "source_dataset": source_dataset,
        "language_tag": language_tag,
    }


def harvest(dataset_id, language_tag, out_f, max_rows, streaming, text_key="text",
            config=None, split="train"):
    configs = [config] if config else matching_configs(dataset_id)
    if not configs:
        print(f"[skip] no matching configs found for {dataset_id}", file=sys.stderr)
        return 0

    written = 0
    for cfg in configs:
        try:
            ds = load_dataset(dataset_id, cfg, split=split, streaming=streaming)
        except Exception as e:
            print(f"[warn] failed to load {dataset_id}/{cfg}/{split}: {e}", file=sys.stderr)
            continue

        iterator = ds if streaming else tqdm(ds, desc=f"{dataset_id}/{cfg}")
        for example in iterator:
            row = normalize_row(example, f"{dataset_id}:{cfg}", language_tag, text_key=text_key)
            if row is None:
                continue
            # Don't try to serialize raw waveform arrays into jsonl — write the
            # source path/id only; 02_segment_turns.py re-loads audio via
            # `datasets`/`soundfile` using this pointer.
            row.pop("audio_array", None)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written >= max_rows:
                return written
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="+",
                     default=["indicvoices", "vaani", "svarah", "lahaja", "common_voice"])
    ap.add_argument("--max_per_source", type=int, default=5000)
    ap.add_argument("--streaming", action="store_true", default=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    source_specs = {
        "indicvoices": dict(dataset_id="ai4bharat/IndicVoices", language_tag="hi-en",
                             text_key="text"),
        "vaani": dict(dataset_id="ARTPARK-IISc/Vaani", language_tag="hi-en",
                       text_key="transcript"),
        "svarah": dict(dataset_id="ai4bharat/Svarah", language_tag="en-IN",
                        text_key="text", config="default", split="test"),
        "lahaja": dict(dataset_id="ai4bharat/Lahaja", language_tag="hi",
                        text_key="text", config="default", split="test"),
        "common_voice": dict(dataset_id="mozilla-foundation/common_voice_11_0",
                              language_tag="hi", text_key="sentence",
                              config="hi", split="train"),
    }

    total = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for src in args.sources:
            spec = source_specs.get(src)
            if spec is None:
                print(f"[skip] unknown source key: {src}", file=sys.stderr)
                continue
            n = harvest(
                dataset_id=spec["dataset_id"],
                language_tag=spec["language_tag"],
                out_f=out_f,
                max_rows=args.max_per_source,
                streaming=args.streaming,
                text_key=spec["text_key"],
                config=spec.get("config"),
                split=spec.get("split", "train"),
            )
            print(f"[{src}] wrote {n} rows")
            total += n

    print(f"Done. {total} rows -> {args.out}")


if __name__ == "__main__":
    main()
