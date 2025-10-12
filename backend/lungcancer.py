r"""
Train a calibrated XGBoost lung-cancer model (verbose + fail-fast).

Encodings
- Numeric:  age, pack_years  -> StandardScaler (fit on train only)
- Binary:   gender (0=female,1=male), asbestos_exposure, secondhand_smoke_exposure,
            copd_diagnosis, family_history  -> 0/1
- Multi-level (fixed one-hot):
    radon_exposure: low / medium / high
    alcohol_consumption: none / moderate / heavy

Model
- XGBClassifier wrapped in CalibratedClassifierCV(method="isotonic", 5-fold)

Artifacts (saved next to this file)
- scaler.pkl
- model.pkl
- meta.json

Run:
  python lungcancer.py  # defaults to backend/lung_cancer_dataset.csv

(Optionally) override CSV via env:
  PowerShell:  $env:LUNG_CANCER_CSV="C:\path\lung_cancer_dataset.csv"
  CMD:         set LUNG_CANCER_CSV=C:\path\lung_cancer_dataset.csv
  mac/linux:   export LUNG_CANCER_CSV=/path/to/lung_cancer_dataset.csv
"""

import os
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)

# --- XGBoost import with friendly error ---
try:
    from xgboost import XGBClassifier
    import xgboost
except Exception as e:
    raise ImportError("xgboost is not installed. Install it with: pip install xgboost") from e

import sklearn

# ----- Paths -----
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.getenv(
    "LUNG_CANCER_CSV",
    os.path.join(BASE_DIR, "lung_cancer_dataset.csv"),
)
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
META_PATH = os.path.join(BASE_DIR, "meta.json")

# ----- Schema -----
TARGET = "lung_cancer"
NUMERIC_COLS = ["age", "pack_years"]

# true binaries (keep as 0/1)
BINARY_COLS = [
    "gender",  # 0=female, 1=male
    "asbestos_exposure",
    "secondhand_smoke_exposure",
    "copd_diagnosis",
    "family_history",
]

# fixed category levels for stable one-hot
RADON_LEVELS = ["low", "medium", "high"]
ALCOHOL_LEVELS = ["none", "moderate", "heavy"]

BINARY_MEANING = {
    "gender": "0=female, 1=male",
    "asbestos_exposure": "0=no, 1=yes",
    "secondhand_smoke_exposure": "0=no, 1=yes",
    "copd_diagnosis": "0=no, 1=yes",
    "family_history": "0=no, 1=yes",
}

# -------------------
# Parsers / Normalizers
# -------------------
def _norm_text(v) -> str:
    """Robust lowercase + whitespace normalize (handles NBSP)."""
    s = "" if v is None else str(v)
    s = s.replace("\u00a0", " ")  # NBSP -> space
    s = " ".join(s.split())       # collapse whitespace
    return s.lower()

def _parse_yesno(v) -> int:
    s = _norm_text(v)
    if s in {"1", "y", "yes", "true", "t"}:
        return 1
    if s in {"0", "n", "no", "false", "f"}:
        return 0
    try:
        return 1 if float(s) >= 0.5 else 0
    except:
        return 0

def _parse_gender(v) -> int:
    s = _norm_text(v)
    if s in {"male", "m", "1"}:
        return 1
    if s in {"female", "f", "0"}:
        return 0
    return _parse_yesno(v)

def _norm_radon(v) -> str:
    s = _norm_text(v)
    if s in {"low", "l"}:
        return "low"
    if s in {"medium", "med", "mid", "m"}:
        return "medium"
    if s in {"high", "h"}:
        return "high"
    # if dataset has "none"/missing, fold into "low" (3-bucket scheme)
    if s in {"none", "no", "0", "nil", "null", "n/a", "na", ""}:
        return "low"
    return "low"

def _norm_alcohol(v) -> str:
    s = _norm_text(v)
    if s in {"none", "no", "0", "nil", "null", "n/a", "na", ""}:
        return "none"
    if s in {"moderate", "mod", "medium", "light", "low"}:
        return "moderate"
    if s in {"heavy", "high", "yes", "1"}:
        return "heavy"
    return "none"

def _parse_target(v) -> int:
    s = _norm_text(v)
    POS = {
        "1", "y", "yes", "true", "t", "pos", "positive", "present",
        "cancer", "lung cancer", "malignant", "has_cancer", "has cancer",
    }
    NEG = {
        "0", "n", "no", "false", "f", "neg", "negative", "absent",
        "no cancer", "benign", "none", "healthy",
    }
    if s in POS:
        return 1
    if s in NEG:
        return 0
    try:
        return 1 if float(s) >= 0.5 else 0
    except:
        return 0

# -------------------
# Data load & encode
# -------------------
def load_dataframe():
    print(f"[1/6] Reading CSV from: {CSV_PATH}", flush=True)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at: {CSV_PATH}\n"
            "Hint: set LUNG_CANCER_CSV env to your file path."
        )

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded shape: {df.shape}", flush=True)

    # Drop common IDs (optional)
    for col in ["patient_id", "id", "uuid"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    # Show raw label preview
    try:
        vc = df[TARGET].value_counts(dropna=False)
        print(f"Raw TARGET value counts:\n{vc.to_string()}", flush=True)
    except Exception:
        pass

    # Parse target -> 0/1
    df[TARGET] = df[TARGET].apply(_parse_target).astype(int)

    # Show parsed counts (should have both 0 and 1)
    print("Parsed TARGET value counts:", df[TARGET].value_counts(dropna=False).to_dict(), flush=True)

    pi_raw = float(df[TARGET].mean())
    print(f"Parsed target prevalence = {pi_raw:.4f}", flush=True)
    if pi_raw <= 0.0 or pi_raw >= 1.0:
        uniq = df[TARGET].unique().tolist()
        raise ValueError(
            "Target ended up single-class after parsing. "
            f"Prevalence={pi_raw:.4f}. Extend _parse_target() to cover your labels. "
            f"Parsed uniques: {uniq}"
        )

    # Binaries
    print("[2/6] Parsing binary columns…", flush=True)
    for c in BINARY_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing expected binary column: {c}")
    df["gender"] = df["gender"].apply(_parse_gender).astype(int)
    for c in [c for c in BINARY_COLS if c != "gender"]:
        df[c] = df[c].apply(_parse_yesno).astype(int)

    # Numerics
    print("[3/6] Casting numerics…", flush=True)
    for c in NUMERIC_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing expected numeric column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    # Multi-level → normalized → one-hot
    print("[4/6] One-hot for radon & alcohol…", flush=True)
    if "radon_exposure" not in df.columns:
        raise ValueError("Missing expected column: radon_exposure")
    if "alcohol_consumption" not in df.columns:
        raise ValueError("Missing expected column: alcohol_consumption")

    # show raw distincts to confirm mapping
    print(
        "Distinct radon_exposure (raw):",
        list(pd.Series(df["radon_exposure"]).astype(str).str.lower().unique())[:10],
        flush=True,
    )
    print(
        "Distinct alcohol_consumption (raw):",
        list(pd.Series(df["alcohol_consumption"]).astype(str).str.lower().unique())[:10],
        flush=True,
    )

    df["radon_norm"] = df["radon_exposure"].map(_norm_radon)
    df["alcohol_norm"] = df["alcohol_consumption"].map(_norm_alcohol)

    radon_oh = pd.get_dummies(df["radon_norm"], prefix="radon", dtype=int).reindex(
        columns=[f"radon_{lvl}" for lvl in RADON_LEVELS], fill_value=0
    )
    alcohol_oh = pd.get_dummies(df["alcohol_norm"], prefix="alcohol", dtype=int).reindex(
        columns=[f"alcohol_{lvl}" for lvl in ALCOHOL_LEVELS], fill_value=0
    )

    one_hot_cols = list(radon_oh.columns) + list(alcohol_oh.columns)
    feature_order = NUMERIC_COLS + BINARY_COLS + one_hot_cols

    X = pd.concat([df[NUMERIC_COLS], df[BINARY_COLS], radon_oh, alcohol_oh], axis=1)
    y = df[TARGET].astype(int)

    print(f"Final X shape: {X.shape} | y mean: {y.mean():.4f}", flush=True)
    print("Feature order:", feature_order, flush=True)

    return X, y, feature_order, one_hot_cols

# -------------------
# Split & scale
# -------------------
def split_and_scale(X, y):
    print("[5/6] Train/test split + scale…", flush=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    Xtr.loc[:, NUMERIC_COLS] = scaler.fit_transform(Xtr[NUMERIC_COLS].astype(float))
    Xte.loc[:, NUMERIC_COLS] = scaler.transform(Xte[NUMERIC_COLS].astype(float))
    return Xtr, Xte, ytr, yte, scaler

# -------------------
# Train
# -------------------
def train_calibrated_xgb(Xtr, ytr):
    print("[6/6] Train XGBoost + isotonic calibration…", flush=True)
    pos = int(ytr.sum())
    neg = int(len(ytr) - pos)
    spw = (neg / max(pos, 1)) if pos else 1.0

    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=spw,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = CalibratedClassifierCV(estimator=xgb, method="isotonic", cv=skf)
    clf.fit(Xtr, ytr)
    return clf

# -------------------
# Evaluate
# -------------------
def evaluate(model, Xte, yte):
    proba = model.predict_proba(Xte)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise RuntimeError(
            "Model predict_proba returned a single column. "
            "This usually means the target had only one class after parsing."
        )
    p = proba[:, 1]
    roc = roc_auc_score(yte, p)
    pr = average_precision_score(yte, p)
    brier = brier_score_loss(yte, p)
    prec, rec, thr = precision_recall_curve(yte, p)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.argmax(f1s[:-1])) if len(thr) else 0
    best_thr = float(thr[best_idx]) if len(thr) else 0.5
    best_f1 = float(f1s[best_idx]) if len(f1s) else 0.0
    print(
        f"Calibrated XGBoost: ROC-AUC={roc:.3f} PR-AUC={pr:.3f} "
        f"Brier={brier:.3f} BestF1={best_f1:.3f} @ thr={best_thr:.3f}",
        flush=True,
    )
    return roc, pr, brier, best_f1, best_thr

# -------------------
# Save artifacts
# -------------------
def save_artifacts(scaler, model, feature_order, one_hot_cols, pi_train: float):
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "pi_train": float(pi_train),
        "feature_order": feature_order,
        "numeric_cols": NUMERIC_COLS,
        "binary_cols": BINARY_COLS,
        "binary_meaning": BINARY_MEANING,
        "one_hot_cols": one_hot_cols,
        "radon_levels": RADON_LEVELS,
        "alcohol_levels": ALCOHOL_LEVELS,
        "target": TARGET,
        "calibration_method": "isotonic",
        "model_family": "XGBoost",
        "versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "csv_path_used": CSV_PATH,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved: {SCALER_PATH}", flush=True)
    print(f"✅ Saved: {MODEL_PATH}", flush=True)
    print(f"✅ Saved: {META_PATH}", flush=True)

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    print(">>> START lungcancer.py", flush=True)
    print(f"Using Python at: {os.sys.executable}", flush=True)
    print(f"BASE_DIR: {BASE_DIR}", flush=True)

    X, y, feature_order, one_hot_cols = load_dataframe()
    pi_train = float(y.mean())
    print(f"Training prevalence (pi_train): {pi_train:.4f}", flush=True)

    Xtr, Xte, ytr, yte, scaler = split_and_scale(X, y)
    model = train_calibrated_xgb(Xtr, ytr)
    evaluate(model, Xte, yte)
    save_artifacts(scaler, model, feature_order, one_hot_cols, pi_train)

    print(">>> DONE", flush=True)
