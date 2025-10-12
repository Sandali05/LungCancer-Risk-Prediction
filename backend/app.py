# app.py
"""
FastAPI server that mirrors training encodings:
- gender: Male/Female → 1/0
- binaries: yes/no/true/false/1/0
- radon_exposure: Low/Medium/High → one-hot (radon_low, radon_medium, radon_high)
- alcohol_consumption: None/Moderate/Heavy → one-hot (alcohol_none, alcohol_moderate, alcohol_heavy)
- age & pack_years standardized by saved scaler

Run:
  uvicorn app:app --reload --port 8000
"""
from typing import Optional, Any, Dict
import os, json
import joblib
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(__file__)
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
META_PATH   = os.path.join(BASE_DIR, "meta.json")

missing = [p for p in [SCALER_PATH, MODEL_PATH] if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Missing artifacts: {', '.join(os.path.basename(p) for p in missing)}")

scaler = joblib.load(SCALER_PATH)
model  = joblib.load(MODEL_PATH)

meta: Dict[str, Any] = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        meta = json.load(f)

FEATURE_ORDER = meta.get("feature_order") or [
    # fallback if meta missing (shouldn’t happen once you retrain)
    "age","pack_years","gender","asbestos_exposure","secondhand_smoke_exposure",
    "copd_diagnosis","family_history",
    "radon_low","radon_medium","radon_high",
    "alcohol_none","alcohol_moderate","alcohol_heavy",
]
NUMERIC_COLS   = meta.get("numeric_cols", ["age","pack_years"])
BINARY_COLS    = meta.get("binary_cols", ["gender","asbestos_exposure","secondhand_smoke_exposure","copd_diagnosis","family_history"])
ONE_HOT_COLS   = meta.get("one_hot_cols", ["radon_low","radon_medium","radon_high","alcohol_none","alcohol_moderate","alcohol_heavy"])
RADON_LEVELS   = meta.get("radon_levels", ["low","medium","high"])
ALCOHOL_LEVELS = meta.get("alcohol_levels", ["none","moderate","heavy"])
PI_TRAIN = float(meta.get("pi_train")) if "pi_train" in meta else None

# Optional override
_env_pi_train = os.getenv("PI_TRAIN", "")
if _env_pi_train:
    try: PI_TRAIN = float(_env_pi_train)
    except: pass

PI_DEPLOY = os.getenv("PI_DEPLOY", "")
try: PI_DEPLOY = float(PI_DEPLOY) if PI_DEPLOY else None
except: PI_DEPLOY = None

def _clip01(x: float, eps: float = 1e-12) -> float:
    return max(min(float(x), 1.0 - eps), eps)

def _to_percent(p: Optional[float]) -> Optional[float]:
    if p is None: return None
    p = max(min(p, 0.9999), 0.0)
    return round(p * 100.0, 2)

def prior_adjust(p: float, pi_train: float, pi_deploy: float) -> float:
    p = _clip01(p)
    if not (0.0 < pi_train < 1.0 and 0.0 < pi_deploy < 1.0): return p
    odds = p / (1.0 - p)
    base = (pi_deploy / (1.0 - pi_deploy)) / (pi_train / (1.0 - pi_train))
    return _clip01((odds * base) / (1.0 + (odds * base)))

# --------- parsers mirrored from training ----------
def parse_yesno(v: Any) -> int:
    if v is None: return 0
    if isinstance(v, bool): return 1 if v else 0
    s = str(v).strip().lower()
    if s in {"1","y","Yes","true","t"}: return 1
    if s in {"0","n","No","false","f"}: return 0
    try:
        return 1 if float(s) >= 0.5 else 0
    except:
        return 0

def parse_gender(v: Any) -> int:
    s = str(v).strip().lower()
    if s in {"Male","m","1"}:   return 1
    if s in {"Female","f","0"}: return 0
    return parse_yesno(v)

def norm_radon(v: Any) -> str:
    s = str(v).strip().lower()
    if s in {"Low","l"}: return "Low"
    if s in {"Medium","med","Mid","m"}: return "Medium"
    if s in {"High","h"}: return "High"
    if s in {"None","no","0","nil","null","n/a","na",""}: return "Low"  # 3-bucket scheme
    return "Low"

def norm_alcohol(v: Any) -> str:
    s = str(v).strip().lower()
    if s in {"None","no","0","nil","null","n/a","na",""}: return "None"
    if s in {"Moderate","mod","Medium","light","low"}:    return "Moderate"
    if s in {"Heavy","High","yes","1"}:                   return "Heavy"
    return "none"

def parse_float(val: Any, default: float = 0.0) -> float:
    try: return float(val)
    except: return default

# --------------- API -----------------
app = FastAPI(title="Lung Cancer Risk API (Calibrated XGBoost)", version="2.3")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class PatientInput(BaseModel):
    age: Any
    pack_years: Any
    gender: Any
    radon_exposure: Any              # expect "low|medium|high" (case-insensitive)
    asbestos_exposure: Any
    secondhand_smoke_exposure: Any
    copd_diagnosis: Any
    alcohol_consumption: Any         # expect "none|moderate|heavy" (case-insensitive)
    family_history: Any

@app.post("/predict")
def predict_risk(
    p: PatientInput,
    pi_deploy: Optional[float] = Query(
        default=None, description="Override deployment prevalence (0..1), e.g., 0.002 for 0.2%"
    ),
):
    # 1) parse & normalize incoming values
    age = parse_float(p.age, 0.0)
    pack_years = parse_float(p.pack_years, 0.0)

    gender  = parse_gender(p.gender)
    asbestos = parse_yesno(p.asbestos_exposure)
    second  = parse_yesno(p.secondhand_smoke_exposure)
    copd    = parse_yesno(p.copd_diagnosis)
    family  = parse_yesno(p.family_history)

    radon_level   = norm_radon(p.radon_exposure)         # low/medium/high
    alcohol_level = norm_alcohol(p.alcohol_consumption)  # none/moderate/heavy

    # 2) build one-hot vectors for radon & alcohol with the exact trained columns
    radon_cols   = [f"radon_{lvl}" for lvl in RADON_LEVELS]
    alcohol_cols = [f"alcohol_{lvl}" for lvl in ALCOHOL_LEVELS]

    radon_oh = {c: 0 for c in radon_cols}
    alc_oh   = {c: 0 for c in alcohol_cols}
    radon_key = f"radon_{radon_level}" if f"radon_{radon_level}" in radon_oh else "radon_low"
    alco_key  = f"alcohol_{alcohol_level}" if f"alcohol_{alcohol_level}" in alc_oh else "alcohol_none"
    radon_oh[radon_key] = 1
    alc_oh[alco_key] = 1

    # 3) standardize numeric features using saved scaler
    numeric_df = pd.DataFrame([[age, pack_years]], columns=NUMERIC_COLS)
    numeric_scaled = scaler.transform(numeric_df)
    age_s, pack_s = float(numeric_scaled[0, NUMERIC_COLS.index("age")]), float(numeric_scaled[0, NUMERIC_COLS.index("pack_years")])

    # 4) assemble the full feature vector in training order
    features = {
        "age": age_s,
        "pack_years": pack_s,
        "gender": int(gender),
        "asbestos_exposure": int(asbestos),
        "secondhand_smoke_exposure": int(second),
        "copd_diagnosis": int(copd),
        "family_history": int(family),
        **radon_oh,
        **alc_oh,
    }

    # ensure all expected columns exist (and only those)
    for k in FEATURE_ORDER:
        if k not in features:
            features[k] = 0
    x_df = pd.DataFrame([features])[FEATURE_ORDER]

    # 5) predict raw prob (calibrated to training prior)
    p_raw = _clip01(float(model.predict_proba(x_df)[0, 1]))

    # 6) optional prevalence adjustment
    use_pi_deploy = pi_deploy if (pi_deploy is not None) else PI_DEPLOY
    used_adjustment = (PI_TRAIN is not None) and (use_pi_deploy is not None) and (0.0 < use_pi_deploy < 1.0)
    p_adj = prior_adjust(p_raw, PI_TRAIN, use_pi_deploy) if used_adjustment else None
    p_main = p_adj if used_adjustment else p_raw

    model_name = getattr(getattr(model, "estimator", model), "__class__", type(model)).__name__
    return {
        "model": model_name,
        "risk_percentage": _to_percent(p_main),
        "raw_risk_percentage": _to_percent(p_raw),
        "adjusted_risk_percentage": _to_percent(p_adj) if p_adj is not None else None,
        "adjusted_for_prevalence": used_adjustment,
        "pi_train": PI_TRAIN,
        "pi_deploy": use_pi_deploy,
        "inputs_used": {
            "age": age, "pack_years": pack_years, "gender": gender,
            "asbestos_exposure": asbestos, "secondhand_smoke_exposure": second,
            "copd_diagnosis": copd, "family_history": family,
            "radon_level": radon_level, "alcohol_level": alcohol_level
        },
    }

@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /predict with PatientInput JSON"}

@app.get("/model-info")
def model_info():
    model_name = getattr(getattr(model, "estimator", model), "__class__", type(model)).__name__
    return {
        "feature_order": FEATURE_ORDER,
        "numeric_cols": NUMERIC_COLS,
        "binary_cols": BINARY_COLS,
        "one_hot_cols": ONE_HOT_COLS,
        "radon_levels": RADON_LEVELS,
        "alcohol_levels": ALCOHOL_LEVELS,
        "binary_meaning": meta.get("binary_meaning"),
        "pi_train": PI_TRAIN,
        "pi_deploy": PI_DEPLOY,
        "notes": "Server parses strings; standardizes age & pack_years; builds one-hot for radon/alcohol to match training.",
        "model_class": model_name,
        "calibration_method": meta.get("calibration_method", "isotonic"),
        "model_family": meta.get("model_family", "XGBoost"),
    }
