import numpy as np
from backend.app.engine import inference_engine
from backend.app.explainer import ClinicalExplainer
from backend.app.recourse import compute_counterfactual_recourse

sample_patient = np.array([[65, 1, 0, 160, 286, 1, 2, 108, 1, 2.6, 1, 3, 3]], dtype=float)

pred, conf, risk, scaled = inference_engine.predict(sample_patient)
explainer = ClinicalExplainer(inference_engine.model)
shaps = explainer.compute_shap_values(scaled)
recourse = compute_counterfactual_recourse(inference_engine.model, inference_engine.scaler, sample_patient)

print("=== CLINICAL INFERENCE VERIFICATION ===")
print("Prediction:", pred, "| Heart Disease:", bool(pred == 1))
print("Disease Probability:", round(risk * 100, 2), "% | Confidence:", round(conf * 100, 2), "%")

print("\nTop 3 SHAP Risk Drivers:")
for item in shaps[:3]:
    print(" -", item["feature"], ":", item["impact"])

print("\nL-BFGS-B Counterfactual Recourse:")
print("Current Risk:", recourse["current_risk"], "% -> Projected Risk:", recourse["projected_risk"], "%")
print("Optimized Interventions:")
for inv in recourse["interventions"]:
    print(" -", inv["biomarker"], ":", inv["current_value"], "->", inv["target_value"], "(delta:", inv["required_delta"], ")")
