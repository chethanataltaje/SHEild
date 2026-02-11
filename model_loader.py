# model_loader.py
import joblib
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
import os

MODEL_FILE = "catboost_sbert_hand.cbm"
META_FILE  = "catboost_meta.joblib"
SBERT_DIR  = "sbert_model"

def load_models():
    """
    Loads SBERT, CatBoost model and metadata.
    Returns: (sbert, catboost_model, global_thr, reclaimed_thr)
    """
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Missing model file: {MODEL_FILE}")
    if not os.path.exists(META_FILE):
        raise FileNotFoundError(f"Missing metadata file: {META_FILE}")
    if not os.path.isdir(SBERT_DIR):
        raise FileNotFoundError(f"Missing SBERT folder: {SBERT_DIR}/")

    sbert = SentenceTransformer(SBERT_DIR)
    model = CatBoostClassifier()
    model.load_model(MODEL_FILE)

    meta = joblib.load(META_FILE)
    global_thr = float(meta.get("best_thr", 0.5))
    reclaimed_thr = float(meta.get("best_thr_reclaimed", max(global_thr, 0.6)))

    return sbert, model, global_thr, reclaimed_thr

