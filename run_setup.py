import os

files = {}

files["backend/app/schemas.py"] = '''from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PatientInput(BaseModel):
    age: float = Field(..., ge=18.0, le=120.0)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=80.0, le=240.0)
    chol: float = Field(..., ge=100.0, le=600.0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=60.0, le=240.0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0.0, le=10.0)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=1, le=3)

class ShapContribution(BaseModel):
    feature: str
    impact: float

class RecourseIntervention(BaseModel):
    biomarker: str
    current_value: float
    target_value: float
    required_delta: float

class RecourseResult(BaseModel):
    recourse_needed: bool
    current_risk: float
    projected_risk: float
    interventions: List[RecourseIntervention]

class AgentSynthesis(BaseModel):
    triage_assessment: str
    key_pathological_drivers: List[str]
    prioritized_interventions: List[str]
    clinical_urgency_level: str

class PredictionResponse(BaseModel):
    prediction: int
    has_heart_disease: bool
    risk_score_percentage: float
    confidence: float
    risk_tier: str
    clinical_notes: str
    shap_contributions: List[ShapContribution]
    counterfactual_recourse: RecourseResult
    agent_synthesis: AgentSynthesis

class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any]

class ChatResponse(BaseModel):
    reply: str
'''

files["backend/app/engine.py"] = '''import os
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
'''

files["backend/app/explainer.py"] = '''import shap
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
'''

files["backend/app/recourse.py"] = '''import numpy as np
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
'''

files["backend/app/agent.py"] = '''import os
import json
from typing import Dict, Any, List

class AutonomousClinicalAgent:
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.provider = None

        if self.groq_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.groq_key)
                self.provider = "groq"
                self.model_name = "llama-3.3-70b-versatile"
            except ImportError:
                pass
        elif self.openai_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_key)
                self.provider = "openai"
                self.model_name = "gpt-4o-mini"
            except ImportError:
                pass

    def generate_clinical_synthesis(
        self, patient_data: Dict[str, Any], risk_score: float, 
        risk_tier: str, shap_items: List[Dict[str, Any]], recourse: Dict[str, Any]
    ) -> Dict[str, Any]:
        system_prompt = """You are Dr. CardioSense AI, Senior Clinical Cardiothoracic AI Specialist.
Analyze patient biomarkers, verified tree predictions (ROC-AUC 0.945), local SHAP game-theoretic weights, and counterfactual recourse deltas.
Output strict JSON:
{
  "triage_assessment": "Comprehensive 2-3 paragraph clinical assessment",
  "key_pathological_drivers": ["Explanation of top SHAP features"],
  "prioritized_interventions": ["Actionable steps corresponding to recourse deltas"],
  "clinical_urgency_level": "ROUTINE" | "URGENT_TRIAGE" | "CRITICAL_EMERGENCY"
}"""

        user_prompt = f"""Clinical Profile: {json.dumps(patient_data)}
Risk Index: {risk_score}% ({risk_tier} Tier, Sensitivity Cutoff: 35.0%)
Top SHAP Vectors: {json.dumps(shap_items[:4])}
Counterfactual Target: {recourse.get('current_risk')}% -> {recourse.get('projected_risk')}%
Interventions: {json.dumps(recourse.get('interventions', []))}"""

        if self.provider in ["groq", "openai"]:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                return json.loads(resp.choices[0].message.content)
            except Exception:
                pass

        drivers = [
            f"{item['feature'].upper()}: marginal impact of {item['impact']:+.4f} on coronary obstruction probability" 
            for item in shap_items[:3]
        ]
        interventions = [
            f"Target {inv['biomarker']}: modify from {inv['current_value']} to {inv['target_value']} (delta {inv['required_delta']})" 
            for inv in recourse.get("interventions", [])
        ]
        if not interventions:
            interventions = [
                "Maintain baseline aerobic conditioning (150 min/wk moderate).",
                "Annual baseline ECG and ApoB/LDL cholesterol tracking."
            ]

        urgency = "CRITICAL_EMERGENCY" if risk_score > 70 else "URGENT_TRIAGE" if risk_score > 35 else "ROUTINE"
        assessment = (
            f"Patient presents with an evaluated coronary disease index of {risk_score}% ({risk_tier} tier). "
            f"Primary physiological strain is driven by {', '.join([s['feature'] for s in shap_items if s['impact'] > 0][:2]) or 'baseline parameters'}. "
            f"Mathematical counterfactual recourse indicates achieving the proposed biomarker adjustments projects risk reduction to {recourse.get('projected_risk', risk_score)}%."
        )

        return {
            "triage_assessment": assessment,
            "key_pathological_drivers": drivers,
            "prioritized_interventions": interventions,
            "clinical_urgency_level": urgency
        }

    def chat_reply(self, message: str, context: Dict[str, Any]) -> str:
        system_prompt = f"""You are CardioSense Copilot, a senior attending cardiologist AI.
Current Clinical Telemetry Context: {json.dumps(context)}
Provide exact, clinically sound medical answers directly citing the evaluated telemetry."""

        if self.provider in ["groq", "openai"]:
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    temperature=0.3
                )
                return res.choices[0].message.content
            except Exception as err:
                return f"Agent operational note: {err}"
        return f"Based on the evaluated risk of {context.get('risk_score', 'N/A')}%, the primary focus must be controlling {context.get('top_driver', 'systolic blood pressure and ischemic ST depression')}. Consult a board-certified cardiologist for diagnostic myocardial perfusion imaging."

clinical_agent = AutonomousClinicalAgent()
'''

files["backend/app/main.py"] = '''import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    PatientInput, PredictionResponse, ShapContribution, 
    RecourseResult, AgentSynthesis, ChatRequest, ChatResponse
)
from .engine import inference_engine
from .explainer import ClinicalExplainer
from .recourse import compute_counterfactual_recourse
from .agent import clinical_agent

app = FastAPI(
    title="CardioSense Premier Clinical AI Engine",
    version="5.0.0",
    description="UCI Cleveland Validated Cardiac Intelligence Service with Game-Theoretic SHAP and Recourse Optimization."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

explainer = None

@app.on_event("startup")
def startup_event():
    global explainer
    if inference_engine.model is not None:
        explainer = ClinicalExplainer(inference_engine.model)
        print("[STARTUP] ClinicalExplainer (TreeExplainer) ready.")

@app.get("/health")
def health():
    return {
        "status": "operational",
        "agent_provider": clinical_agent.provider or "evidence-based-engine",
        "model_loaded": inference_engine.model is not None,
        "scaler_loaded": inference_engine.scaler is not None,
        "explainer_loaded": explainer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_cardiac_risk(patient: PatientInput):
    raw_vector = np.array([[
        patient.age, patient.sex, patient.cp, patient.trestbps, patient.chol,
        patient.fbs, patient.restecg, patient.thalach, patient.exang,
        patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]], dtype=float)

    try:
        pred, conf, risk, scaled = inference_engine.predict(raw_vector)
        shaps = explainer.compute_shap_values(scaled)
        recourse = compute_counterfactual_recourse(
            inference_engine.model, inference_engine.scaler, raw_vector
        )

        tier = "High" if risk >= 0.70 else "Moderate" if risk >= 0.35 else "Low"
        notes = (
            "Screening threshold (35%) exceeded. Obstructive coronary artery disease suspected."
            if risk >= 0.35 else
            "Patient sits below 35% sensitivity screening threshold. No critical ischemic markers identified."
        )

        agent_report = clinical_agent.generate_clinical_synthesis(
            patient_data=patient.model_dump(),
            risk_score=round(risk * 100, 2),
            risk_tier=tier,
            shap_items=shaps,
            recourse=recourse
        )

        return PredictionResponse(
            prediction=pred,
            has_heart_disease=bool(pred == 1),
            risk_score_percentage=round(risk * 100, 2),
            confidence=round(conf * 100, 2),
            risk_tier=tier,
            clinical_notes=notes,
            shap_contributions=[ShapContribution(**item) for item in shaps],
            counterfactual_recourse=RecourseResult(**recourse),
            agent_synthesis=AgentSynthesis(**agent_report)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    reply = clinical_agent.chat_reply(req.message, req.context)
    return ChatResponse(reply=reply)
'''

os.makedirs("backend/app", exist_ok=True)
for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[SUCCESS] Wrote {path}")

print("[COMPLETE] All backend modules generated cleanly!")
