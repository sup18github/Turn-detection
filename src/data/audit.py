import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def audit_dataset(manifest_path: str = None):
    if manifest_path is None:
        manifest_path = PROCESSED_DIR / "dataset_manifest.jsonl"
    
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
            
    df = pd.DataFrame(records)

    report = {
        "total_samples": len(df),
        "total_duration_hours": float(df["duration"].sum() / 3600.0),
        "unique_speakers": int(df["speaker_id"].nunique()),
        "unique_conversations": int(df["conversation_id"].nunique()),
        "sample_rate_hz": int(df["sample_rate"].iloc[0]),
        "channels": 1,
        "format": "Float32 PCM",
        "label_distribution": df["label_name"].value_counts().to_dict(),
        "class_balance_pct": (df["label"].mean() * 100.0), # % END labels
        "language_distribution": df["language"].value_counts().to_dict(),
        "category_distribution": df["category"].value_counts().to_dict(),
        "mean_pause_duration_s": float(df["pause_duration_s"].mean()),
        "std_pause_duration_s": float(df["pause_duration_s"].std()),
        "max_pause_duration_s": float(df["pause_duration_s"].max()),
        "min_pause_duration_s": float(df["pause_duration_s"].min())
    }

    # Save JSON report
    report_json_path = RESULTS_DIR / "dataset_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save CSV summary
    report_csv_path = RESULTS_DIR / "dataset_report.csv"
    summary_df = pd.DataFrame([report])
    summary_df.to_csv(report_csv_path, index=False)

    print("=== Dataset Audit Report ===")
    print(f"Total Samples: {report['total_samples']}")
    print(f"Unique Speakers: {report['unique_speakers']}")
    print(f"Label Distribution: {report['label_distribution']}")
    print(f"Category Breakdown:\n{df['category'].value_counts()}")
    print(f"Report saved to {report_json_path} and {report_csv_path}")

    return report


if __name__ == "__main__":
    audit_dataset()
