import shap
import numpy as np

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

class ClinicalExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def compute_shap_values(self, scaled_vector: np.ndarray):
        shap_vals = self.explainer.shap_values(scaled_vector)
        if isinstance(shap_vals, list):
            class_shap = shap_vals[1][0]
        elif len(shap_vals.shape) == 3:
            class_shap = shap_vals[0, :, 1]
        else:
            class_shap = shap_vals[0]

        contributions = [
            {"feature": f, "impact": round(float(v), 4)}
            for f, v in zip(FEATURES, class_shap)
        ]
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions
