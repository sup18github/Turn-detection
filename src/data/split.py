import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def generate_speaker_splits(
    manifest_path: str = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    random.seed(seed)

    if manifest_path is None:
        manifest_path = PROCESSED_DIR / "dataset_manifest.jsonl"

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    # Group by speaker_id to guarantee zero speaker leakage
    speakers = sorted(list(set(r["speaker_id"] for r in records)))
    random.shuffle(speakers)

    num_speakers = len(speakers)
    num_train = int(num_speakers * train_ratio)
    num_val = int(num_speakers * val_ratio)

    train_speakers = set(speakers[:num_train])
    val_speakers = set(speakers[num_train:num_train + num_val])
    test_speakers = set(speakers[num_train + num_val:])

    train_data = [r for r in records if r["speaker_id"] in train_speakers]
    val_data = [r for r in records if r["speaker_id"] in val_speakers]
    test_data = [r for r in records if r["speaker_id"] in test_speakers]

    print(f"Split results:")
    print(f"Train samples: {len(train_data)} ({len(train_speakers)} speakers)")
    print(f"Val samples:   {len(val_data)} ({len(val_speakers)} speakers)")
    print(f"Test samples:  {len(test_data)} ({len(test_speakers)} speakers)")

    # Verify zero leakage
    assert len(train_speakers & val_speakers) == 0
    assert len(train_speakers & test_speakers) == 0
    assert len(val_speakers & test_speakers) == 0

    # Write split files
    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        out_path = DATA_DIR / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    return train_data, val_data, test_data


if __name__ == "__main__":
    generate_speaker_splits()
