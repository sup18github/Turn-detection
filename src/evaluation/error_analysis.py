import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.evaluation.metrics import compute_turn_metrics

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

def analyze_model_slices(df: pd.DataFrame, pred_probs: np.ndarray, model_name: str = "Hybrid") -> dict:
    """
    Evaluates turn model performance sliced by:
    1. Language (Hindi, English, Hinglish)
    2. Speech Category (fillers, mid_sentence_pause, long_hesitation, self_correction, short_answer, normal_ending)
    3. Pause Length Buckets (0-200ms, 200-500ms, 500-1000ms, 1-2s, >2s)
    """
    df["pred_prob"] = pred_probs
    df["y_pred"] = (pred_probs >= 0.5).astype(int)

    slice_results = {"model_name": model_name, "slices": {}}

    # 1. Slice by Language
    lang_metrics = {}
    for lang, group in df.groupby("language"):
        m = compute_turn_metrics(group["label"].values, group["pred_prob"].values)
        lang_metrics[lang] = m
    slice_results["slices"]["language"] = lang_metrics

    # 2. Slice by Category
    cat_metrics = {}
    for cat, group in df.groupby("category"):
        m = compute_turn_metrics(group["label"].values, group["pred_prob"].values)
        cat_metrics[cat] = m
    slice_results["slices"]["category"] = cat_metrics

    # 3. Slice by Pause Duration Buckets
    def bucket_pause(val):
        if val <= 0.2:
            return "0-200ms"
        elif val <= 0.5:
            return "200-500ms"
        elif val <= 1.0:
            return "500-1000ms"
        elif val <= 2.0:
            return "1-2s"
        else:
            return ">2s"

    df["pause_bucket"] = df["pause_duration_s"].apply(bucket_pause)
    bucket_metrics = {}
    for bkt, group in df.groupby("pause_bucket"):
        m = compute_turn_metrics(group["label"].values, group["pred_prob"].values)
        bucket_metrics[bkt] = m
    slice_results["slices"]["pause_buckets"] = bucket_metrics

    out_file = RESULTS_DIR / f"slice_analysis_{model_name.lower().replace(' ', '_')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(slice_results, f, indent=2)

    print(f"Slice analysis for {model_name} written to {out_file}")
    return slice_results
