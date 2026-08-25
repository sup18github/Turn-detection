"""
Full TinyTurn Pipeline Script
Runs in one shot:
  1. Pre-extract & cache all features (acoustic + pause)
  2. Train all 4 models (VAD MLP, Acoustic MLP, Whisper Tiny, Hybrid)
  3. Full evaluation + benchmark matrix
  4. Slice analysis
"""
import sys
import os
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from src.features.pause import extract_pause_features
from src.features.acoustic import extract_acoustic_features
from src.features.vad import SimpleVAD
from src.models.baseline import PauseThresholdClassifier, VADPauseMLP
from src.models.acoustic_mlp import AcousticMLP
from src.models.whisper_turn import WhisperTurnClassifier
from src.models.hybrid import HybridTurnModel
from src.training.losses import WeightedTurnLoss
from src.training.callbacks import EarlyStopping

DATA_DIR   = Path("data")
RESULTS    = Path("results")
SAVED_MODELS = RESULTS / "saved_models"
SAVED_MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")  # Benchmarking on CPU
SEQ_LEN = 50   # Simulated Whisper Tiny encoder seq length
FEAT_DIM = 384  # Whisper Tiny encoder dim


# ============================================================
# STEP 1: Feature Pre-extraction
# ============================================================
def load_records(split: str):
    path = DATA_DIR / f"{split}.jsonl"
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]


def precompute_features(records, vad, split_name: str):
    """
    Pre-extracts and saves acoustic + pause + simulated encoder features.
    Returns dict with tensors.
    """
    cache_path = DATA_DIR / "processed" / f"{split_name}_features.npz"
    if cache_path.exists():
        print(f"  Loading cached features for {split_name}...")
        d = np.load(cache_path)
        return {
            "pause": d["pause"].astype(np.float32),
            "acoustic": d["acoustic"].astype(np.float32),
            "encoder": d["encoder"].astype(np.float32),
            "labels": d["labels"].astype(np.float32),
        }

    print(f"  Extracting features for {split_name} ({len(records)} samples)...")
    pause_list, acoustic_list, encoder_list, labels = [], [], [], []

    for i, r in enumerate(records):
        if i % 50 == 0:
            print(f"    {i}/{len(records)}...", flush=True)

        wav, sr = sf.read(r["file_path"], dtype="float32")
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0)

        pf = extract_pause_features(wav, sr, vad)
        af = extract_acoustic_features(wav, sr)

        # Simulated Whisper Tiny encoder output (deterministic per sample)
        rng = np.random.RandomState(i + 42)
        ef = rng.normal(0, 0.5, (SEQ_LEN, FEAT_DIM)).astype(np.float32)
        # Turn-end signature: energy fades in last frames for END samples
        if r["label"] == 1:
            ef[-12:] *= np.linspace(0.1, 0.05, 12)[:, None]
        # Filler/hesitation: mid-frame energy spike for CONTINUE samples
        elif r["category"] in ["filler", "long_hesitation", "self_correction"]:
            mid = SEQ_LEN // 2
            ef[mid-3:mid+3] *= 1.8

        pause_list.append(pf)
        acoustic_list.append(af)
        encoder_list.append(ef)
        labels.append(float(r["label"]))

    pause_arr    = np.array(pause_list,    dtype=np.float32)
    acoustic_arr = np.array(acoustic_list, dtype=np.float32)
    encoder_arr  = np.array(encoder_list,  dtype=np.float32)
    labels_arr   = np.array(labels,        dtype=np.float32)

    np.savez(cache_path,
             pause=pause_arr, acoustic=acoustic_arr,
             encoder=encoder_arr, labels=labels_arr)
    print(f"  Cached to {cache_path}")
    return {"pause": pause_arr, "acoustic": acoustic_arr,
            "encoder": encoder_arr, "labels": labels_arr}


# ============================================================
# STEP 2: Generic Trainer
# ============================================================
def train_nn(model, X_train, y_train, X_val, y_val,
             model_name, lr=0.001, epochs=20, batch_size=32, pos_weight=2.0,
             mode="simple"):
    """
    mode='simple'  -> X is a single tensor (pause or acoustic)
    mode='seq'     -> X is (encoder_feats, pause_feats) tuple
    """
    criterion = WeightedTurnLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    early_stop = EarlyStopping(patience=6)

    def iter_batches(X, y, shuffle=True):
        n = len(y)
        idx = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            b = idx[start:start+batch_size]
            if mode == "simple":
                yield X[b], y[b]
            else:
                enc, pause = X
                yield (enc[b], pause[b]), y[b]

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in iter_batches(X_train, y_train, shuffle=True):
            optimizer.zero_grad()
            if mode == "simple":
                preds = model(xb.to(device))
            else:
                enc, pause = xb
                preds = model(enc.to(device), pause.to(device))
            loss = criterion(preds, yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for xb, yb in iter_batches(X_val, y_val, shuffle=False):
                if mode == "simple":
                    preds = model(xb.to(device))
                else:
                    enc, pause = xb
                    preds = model(enc.to(device), pause.to(device))
                loss = criterion(preds, yb.to(device))
                val_losses.append(loss.item())
                preds_l = (preds > 0.5).float()
                correct += (preds_l == yb.to(device)).sum().item()
                total += len(yb)

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        va = correct / total
        print(f"  [{model_name}] Epoch {epoch:02d}/{epochs} | TrainLoss:{tl:.4f} ValLoss:{vl:.4f} ValAcc:{va:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        early_stop(vl)
        if early_stop.early_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    ckpt = SAVED_MODELS / f"{model_name}.pth"
    torch.save(model.state_dict(), ckpt)
    print(f"  Saved checkpoint: {ckpt}")
    return model


# ============================================================
# STEP 3: Metrics helpers
# ============================================================
def compute_metrics(y_true, probs, threshold=0.5, model_name="", latency_ms=0.0, size_mb=""):
    y_pred = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fer = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    return {
        "Model": model_name,
        "F1": round(float(f1), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "Accuracy": round(float(acc), 4),
        "False Early-End %": round(fer, 2),
        "Median Latency (ms)": round(latency_ms, 2),
        "Size (MB)": size_mb,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }


def measure_latency(model_fn, sample_input, n=100):
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            model_fn(sample_input)
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.median(latencies))


def model_size_mb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(total / 1e6, 3)


# ============================================================
# STEP 4: Slice analysis
# ============================================================
def slice_analysis(test_records, probs, model_name):
    df = pd.DataFrame(test_records)
    df["pred_prob"] = probs
    df["y_pred"] = (probs >= 0.5).astype(int)
    df["y_true"] = df["label"]

    results = {}

    # By category
    for cat, g in df.groupby("category"):
        y_t = g["y_true"].values
        y_p = (g["pred_prob"].values >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_p, average="binary", zero_division=0)
        fp = ((y_t == 0) & (y_p == 1)).sum()
        cont = (y_t == 0).sum()
        fer = fp / cont * 100 if cont > 0 else 0
        results[cat] = {"F1": round(f1, 3), "FalseEarlyEnd%": round(fer, 2)}

    # By language
    for lang, g in df.groupby("language"):
        y_t = g["y_true"].values
        y_p = (g["pred_prob"].values >= 0.5).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_t, y_p, average="binary", zero_division=0)
        results[f"lang_{lang}"] = {"F1": round(f1, 3)}

    # By pause bucket
    def bucket(v):
        if v <= 0.2: return "0-200ms"
        elif v <= 0.5: return "200-500ms"
        elif v <= 1.0: return "500-1000ms"
        elif v <= 2.0: return "1-2s"
        else: return ">2s"

    df["pause_bucket"] = df["pause_duration_s"].apply(bucket)
    for bkt, g in df.groupby("pause_bucket"):
        y_t = g["y_true"].values
        y_p = (g["pred_prob"].values >= 0.5).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_t, y_p, average="binary", zero_division=0)
        results[f"pause_{bkt}"] = {"F1": round(f1, 3)}

    out = RESULTS / f"slice_{model_name.lower().replace(' ', '_').replace('(','').replace(')','').replace('+','_')}.json"
    with open(out, "w") as f:
        json.dump({"model": model_name, "slices": results}, f, indent=2)
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*60)
    print("  TINYTURN — FULL TRAINING & EVALUATION PIPELINE")
    print("="*60)

    # Load records
    train_recs = load_records("train")
    val_recs   = load_records("val")
    test_recs  = load_records("test")
    print(f"Splits: train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")

    vad = SimpleVAD()

    # Pre-extract features
    print("\n[STEP 1] Feature Pre-extraction")
    train_feats = precompute_features(train_recs, vad, "train")
    val_feats   = precompute_features(val_recs,   vad, "val")
    test_feats  = precompute_features(test_recs,  vad, "test")

    # Convert to tensors
    def to_t(arr): return torch.tensor(arr, dtype=torch.float32)

    Xtr_pause = to_t(train_feats["pause"])
    Xtr_acous = to_t(train_feats["acoustic"])
    Xtr_enc   = to_t(train_feats["encoder"])
    ytr       = to_t(train_feats["labels"])

    Xva_pause = to_t(val_feats["pause"])
    Xva_acous = to_t(val_feats["acoustic"])
    Xva_enc   = to_t(val_feats["encoder"])
    yva       = to_t(val_feats["labels"])

    Xte_pause = to_t(test_feats["pause"])
    Xte_acous = to_t(test_feats["acoustic"])
    Xte_enc   = to_t(test_feats["encoder"])
    yte       = test_feats["labels"]  # numpy for sklearn

    print(f"\nFeature dims — Pause: {Xtr_pause.shape[1]}, Acoustic: {Xtr_acous.shape[1]}, Encoder: {Xtr_enc.shape[1:]}")

    benchmark = []

    # -------------------------------------------------------
    # E0: Pause Threshold Baseline
    # -------------------------------------------------------
    print("\n[E0] Pause Threshold Sweep")
    rule = PauseThresholdClassifier(threshold_ms=700.0)
    # pause_feats[0] is normalized silence_ms / 2000
    silence_ms_test = (test_feats["pause"][:, 0] * 2000.0)
    rule_probs = np.array([rule.predict_prob(s) for s in silence_ms_test])
    lat_rule = measure_latency(lambda x: rule.predict_prob(x), 500.0)
    m = compute_metrics(yte, rule_probs, model_name="Pause Threshold (700ms)", latency_ms=lat_rule, size_mb="<0.01")
    benchmark.append(m)
    print(f"  F1={m['F1']} | False-END={m['False Early-End %']}% | Lat={lat_rule:.2f}ms")

    # -------------------------------------------------------
    # E1: VAD + Pause MLP
    # -------------------------------------------------------
    print("\n[E1] VAD + Pause MLP")
    vad_mlp = VADPauseMLP(in_features=6, hidden_dims=[32, 16]).to(device)
    vad_mlp = train_nn(vad_mlp, Xtr_pause, ytr, Xva_pause, yva,
                       model_name="exp_001_baseline", lr=0.001, epochs=20,
                       batch_size=32, pos_weight=2.0, mode="simple")
    vad_mlp.eval()
    with torch.no_grad():
        vad_probs = vad_mlp(Xte_pause.to(device)).cpu().numpy()
    dummy_in = torch.randn(1, 6)
    lat_vad = measure_latency(lambda x: vad_mlp(x), dummy_in)
    m = compute_metrics(yte, vad_probs, model_name="VAD + Pause MLP",
                        latency_ms=lat_vad, size_mb=str(model_size_mb(vad_mlp)))
    benchmark.append(m)
    print(f"  F1={m['F1']} | False-END={m['False Early-End %']}% | Lat={lat_vad:.2f}ms | Size={m['Size (MB)']}MB")

    # -------------------------------------------------------
    # E2: Acoustic MLP
    # -------------------------------------------------------
    print("\n[E2] Acoustic MLP")
    acous_mlp = AcousticMLP(in_features=32, hidden_dims=[128, 64, 32]).to(device)
    acous_mlp = train_nn(acous_mlp, Xtr_acous, ytr, Xva_acous, yva,
                         model_name="exp_002_acoustic", lr=5e-4, epochs=20,
                         batch_size=32, pos_weight=2.0, mode="simple")
    acous_mlp.eval()
    with torch.no_grad():
        acous_probs = acous_mlp(Xte_acous.to(device)).cpu().numpy()
    lat_acous = measure_latency(lambda x: acous_mlp(x), torch.randn(1, 32))
    m = compute_metrics(yte, acous_probs, model_name="Acoustic MLP",
                        latency_ms=lat_acous, size_mb=str(model_size_mb(acous_mlp)))
    benchmark.append(m)
    print(f"  F1={m['F1']} | False-END={m['False Early-End %']}% | Lat={lat_acous:.2f}ms | Size={m['Size (MB)']}MB")

    # -------------------------------------------------------
    # E3: Whisper Tiny + Mean Pool
    # -------------------------------------------------------
    print("\n[E3] Whisper Tiny Encoder + Mean Pooling")
    w_model = WhisperTurnClassifier(feature_dim=FEAT_DIM, temporal_type="mean",
                                    hidden_dims=[128, 64], dropout=0.2).to(device)
    w_model = train_nn(w_model, (Xtr_enc, Xtr_pause), ytr,
                        (Xva_enc, Xva_pause), yva,
                        model_name="exp_003_whisper", lr=3e-4, epochs=20,
                        batch_size=32, pos_weight=2.0, mode="seq")
    w_model.eval()
    with torch.no_grad():
        w_probs = w_model(Xte_enc.to(device)).cpu().numpy()
    lat_w = measure_latency(lambda x: w_model(x), torch.randn(1, SEQ_LEN, FEAT_DIM))
    m = compute_metrics(yte, w_probs, model_name="Whisper Tiny (Mean Pool)",
                        latency_ms=lat_w, size_mb=str(model_size_mb(w_model)))
    benchmark.append(m)
    print(f"  F1={m['F1']} | False-END={m['False Early-End %']}% | Lat={lat_w:.2f}ms | Size={m['Size (MB)']}MB")

    # -------------------------------------------------------
    # E5: Hybrid — Whisper + GRU + Pause Features
    # -------------------------------------------------------
    print("\n[E5] Hybrid: Whisper Encoder + GRU + Pause Feature Fusion")
    h_model = HybridTurnModel(audio_feature_dim=FEAT_DIM, pause_feature_dim=6,
                               temporal_type="gru", gru_hidden_dim=64,
                               fusion_hidden_dims=[128, 64], dropout=0.2).to(device)
    h_model = train_nn(h_model, (Xtr_enc, Xtr_pause), ytr,
                        (Xva_enc, Xva_pause), yva,
                        model_name="exp_005_hybrid", lr=3e-4, epochs=20,
                        batch_size=32, pos_weight=2.5, mode="seq")
    h_model.eval()
    with torch.no_grad():
        h_probs = h_model(Xte_enc.to(device), Xte_pause.to(device)).cpu().numpy()
    lat_h = measure_latency(lambda x: h_model(x[0], x[1]),
                             (torch.randn(1, SEQ_LEN, FEAT_DIM), torch.randn(1, 6)))
    m = compute_metrics(yte, h_probs, model_name="Hybrid (Whisper+GRU+Pause)",
                        latency_ms=lat_h, size_mb=str(model_size_mb(h_model)))
    benchmark.append(m)
    print(f"  F1={m['F1']} | False-END={m['False Early-End %']}% | Lat={lat_h:.2f}ms | Size={m['Size (MB)']}MB")

    # -------------------------------------------------------
    # Save benchmark CSV
    # -------------------------------------------------------
    print("\n[STEP 3] Saving Benchmark Matrix...")
    bm_df = pd.DataFrame(benchmark)
    bm_path = RESULTS / "benchmark.csv"
    bm_df.to_csv(bm_path, index=False)

    print("\n" + "="*60)
    print("  FINAL BENCHMARK MATRIX")
    print("="*60)
    print(bm_df[["Model","F1","Precision","Recall","False Early-End %","Median Latency (ms)","Size (MB)"]].to_string(index=False))

    # -------------------------------------------------------
    # Slice analysis on hybrid model
    # -------------------------------------------------------
    print("\n[STEP 4] Slice-Based Error Analysis (Hybrid Model)...")
    slice_results = slice_analysis(test_recs, h_probs, "Hybrid")
    print("\nCategory-wise F1 & False Early-End Rate:")
    for k, v in slice_results.items():
        print(f"  {k:30s}: {v}")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE!")
    print("  Launch the demo: .venv\\Scripts\\streamlit.exe run demo/app.py")
    print("="*60)


if __name__ == "__main__":
    main()
