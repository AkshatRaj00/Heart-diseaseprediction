import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

class ClinicalInferenceEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self._load()

    def _load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            print("[ENGINE] Verified UCI Random Forest and StandardScaler loaded.")
        else:
            raise FileNotFoundError("Clinical artifacts missing. Run train_and_save_model.py first.")

    def predict(self, raw_features: np.ndarray):
        df = pd.DataFrame(raw_features.reshape(1, -1), columns=FEATURE_NAMES)
        scaled = self.scaler.transform(df)
        pred = int(self.model.predict(scaled)[0])
        probs = self.model.predict_proba(scaled)[0]
        confidence = float(probs[pred])
        risk_prob = float(probs[1])
        return pred, confidence, risk_prob, scaled

inference_engine = ClinicalInferenceEngine()
