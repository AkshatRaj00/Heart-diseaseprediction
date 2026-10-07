import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

MODIFIABLE_SPECS = {
    "oldpeak":  {"idx": 9, "step": -0.2, "floor": 0.0},
    "trestbps": {"idx": 3, "step": -5.0, "floor": 115.0},
    "chol":     {"idx": 4, "step": -10.0, "floor": 170.0},
    "thalach":  {"idx": 7, "step": 5.0,  "floor": 160.0}
}

def compute_counterfactual_recourse(model, scaler, raw_vector: np.ndarray, target_risk=0.35):
    df_orig = pd.DataFrame(raw_vector.reshape(1, -1), columns=FEATURE_NAMES)
    scaled_orig = scaler.transform(df_orig)
    prob_orig = float(model.predict_proba(scaled_orig)[0][1])

    if prob_orig <= target_risk:
        return {
            "recourse_needed": False,
            "current_risk": round(prob_orig * 100, 2),
            "projected_risk": round(prob_orig * 100, 2),
            "interventions": []
        }

    df_candidate = df_orig.copy()
    current_prob = prob_orig
    priority = ["oldpeak", "trestbps", "chol", "thalach"]

    for key in priority:
        spec = MODIFIABLE_SPECS[key]
        idx = spec["idx"]
        val = df_candidate.iloc[0, idx]

        while True:
            next_val = val + spec["step"]
            if spec["step"] < 0 and next_val < spec["floor"]:
                next_val = spec["floor"]
            elif spec["step"] > 0 and next_val > spec["floor"]:
                next_val = spec["floor"]

            if next_val == val:
                break

            df_candidate.iloc[0, idx] = next_val
            scaled_cand = scaler.transform(df_candidate)
            new_prob = float(model.predict_proba(scaled_cand)[0][1])

            if new_prob < current_prob:
                current_prob = new_prob
                val = next_val
                if current_prob <= target_risk:
                    break
            else:
                df_candidate.iloc[0, idx] = val
                break

        if current_prob <= target_risk:
            break

    labels = {
        "trestbps": "trestbps (Resting BP mm Hg)",
        "chol": "chol (Serum Cholesterol mg/dL)",
        "thalach": "thalach (Max Heart Rate bpm)",
        "oldpeak": "oldpeak (ST Depression)"
    }

    interventions = []
    for key in priority:
        idx = MODIFIABLE_SPECS[key]["idx"]
        c_orig = float(df_orig.iloc[0, idx])
        c_opt = float(df_candidate.iloc[0, idx])
        delta = round(c_opt - c_orig, 2)
        if abs(delta) >= 0.1:
            interventions.append({
                "biomarker": labels[key],
                "current_value": round(c_orig, 2),
                "target_value": round(c_opt, 2),
                "required_delta": delta
            })

    return {
        "recourse_needed": True,
        "current_risk": round(prob_orig * 100, 2),
        "projected_risk": round(current_prob * 100, 2),
        "interventions": interventions
    }
