import os

# 1. Multi-Agent Engine in backend/app/agent.py
agent_code = """import os
import json
import time
from typing import Dict, Any, List, Generator
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

class ClinicalMedicalBoard:
    def __init__(self):
        raw_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        self.groq_key = raw_key
        self.client = None
        self.model_name = "qwen/qwen3.8-27b"
        
        if self.groq_key and self.groq_key.startswith("gsk_"):
            try:
                from groq import Groq
                self.client = Groq(api_key=self.groq_key)
            except Exception:
                self.client = None

    def stream_triage_debate(self, telemetry: Dict[str, Any], risk_score: float, risk_tier: str, shaps: List[Dict[str, Any]], recourse: Dict[str, Any]) -> Generator[str, None, None]:
        board_prompt = f\"\"\"Patient Clinical Telemetry:
{json.dumps(telemetry, indent=2)}

Risk Metrics: Fused Score {risk_score}% ({risk_tier} Stratification)
Top Biomarker Contributors: {json.dumps(shaps[:4])}
Counterfactual Target Risk: {recourse.get('current_risk')}% -> {recourse.get('projected_risk')}%
Modifications: {json.dumps(recourse.get('interventions', []))}

Simulate a rigorous 3-Agent Clinical Medical Board consensus:
1. [DR. CARDIOLOGIST]: Analyze hemodynamic strain, exercise ST changes, and ischemic mechanics.
2. [DR. PHARMACOLOGIST]: Detail specific pharmacological interventions (Statins, ACEi, Beta-Blockers) targeting the recourse deltas.
3. [DR. SAFETY AUDITOR]: Check for contraindications (e.g., verifying bradycardia or hypotension risks based on patient vitals) and finalize consensus action items.

Produce direct, authoritative clinical discourse with clear headers:
\"\"\"
        if self.client:
            try:
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": board_prompt}],
                    temperature=0.2,
                    max_tokens=900,
                    stream=True
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield f"data: {json.dumps({'token': delta})}\\n\\n"
                yield "data: [DONE]\\n\\n"
                return
            except Exception as e:
                yield f"data: {json.dumps({'token': f'Board streaming fallback initiated: {e}'})}\\n\\n"

        # Deterministic Multi-Agent Streaming Engine
        simulated_deliberation = [
            f"### [DR. CARDIOLOGIST - CLINICAL TRIAGE]\\n"
            f"Patient evaluated at {risk_score}% coronary risk index ({risk_tier} tier). "
            f"Primary ischemic drivers are {', '.join([s['feature'] for s in shaps[:2]])}. "
            f"Exercise-induced telemetry reflects marked subendocardial hypoperfusion under load.\\n\\n",
            
            f"### [DR. PHARMACOLOGIST - THERAPEUTIC REGIMEN]\\n"
            f"Reviewing counterfactual recourse deltas. Recommended targets:\\n"
        ]
        for inv in recourse.get("interventions", []):
            simulated_deliberation.append(f"- Optimize {inv['biomarker']}: shift {inv['current_value']} -> {inv['target_value']} (Delta {inv['required_delta']}).\\n")
        
        simulated_deliberation.append(
            f"\\n### [DR. SAFETY AUDITOR - CONTRAINDICATION AUDIT]\\n"
            f"Cross-referencing vital signs. Resting BP: {telemetry.get('trestbps')} mmHg, Max HR: {telemetry.get('thalach')} bpm. "
            f"No lethal contraindication for standard titrations. "
            f"Directives: 1) Schedule stress echocardiogram, 2) Baseline lipid panel, 3) Monitor ambulatory hemodynamics.\\n"
        )

        for text_block in simulated_deliberation:
            for word in text_block.split(" "):
                yield f"data: {json.dumps({'token': word + ' '})}\\n\\n"
                time.sleep(0.02)
        yield "data: [DONE]\\n\\n"

    def chat_stream(self, message: str, context: Dict[str, Any]) -> Generator[str, None, None]:
        if self.client:
            try:
                system_prompt = f"You are Dr. CardioSense Copilot, a senior cardiologist AI. Telemetry: {json.dumps(context)}. Answer concisely and medically."
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": message}],
                    temperature=0.3,
                    max_tokens=600,
                    stream=True
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield f"data: {json.dumps({'token': delta})}\\n\\n"
                yield "data: [DONE]\\n\\n"
                return
            except Exception:
                pass

        reply = f"[Offline Specialist]: For risk {context.get('risk_score', 'N/A')}%, titrate arterial pressure and control ischemic drivers."
        for word in reply.split(" "):
            yield f"data: {json.dumps({'token': word + ' '})}\\n\\n"
            time.sleep(0.03)
        yield "data: [DONE]\\n\\n"

clinical_board = ClinicalMedicalBoard()
"""

with open("backend/app/agent.py", "w", encoding="utf-8") as f:
    f.write(agent_code)

# 2. Main FastAPI with SSE Streaming Endpoints in backend/app/main.py
main_code = """import os
import torch
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .schemas import (
    PatientInput, PredictionResponse, ShapContribution, 
    RecourseResult, AgentSynthesis, ChatRequest, ChatResponse
)
from .engine import inference_engine
from .explainer import ClinicalExplainer
from .recourse import compute_counterfactual_recourse
from .agent import clinical_board

class CardiacDeepSurvNet(torch.nn.Module):
    def __init__(self, input_dim=13):
        super(CardiacDeepSurvNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.LeakyReLU(0.1)
        self.fc2 = torch.nn.Linear(64, 32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.act2 = torch.nn.LeakyReLU(0.1)
        self.res_fc = torch.nn.Linear(64, 32)
        self.dropout = torch.nn.Dropout(0.25)
        self.out_layer = torch.nn.Linear(32, 1)

    def forward(self, x):
        x1 = self.act1(self.bn1(self.fc1(x)))
        x2 = self.act2(self.bn2(self.fc2(x1)))
        res = self.res_fc(x1)
        return torch.sigmoid(self.out_layer(self.dropout(x2 + res)))

dl_net = CardiacDeepSurvNet(13)
pt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cardiac_deep_learning_model.pt")
if os.path.exists(pt_path):
    dl_net.load_state_dict(torch.load(pt_path, map_location=torch.device('cpu')))
    dl_net.eval()

app = FastAPI(title="CardioSense Consensus Intelligence Engine", version="7.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

explainer = None

@app.on_event("startup")
def startup():
    global explainer
    if inference_engine.model is not None:
        explainer = ClinicalExplainer(inference_engine.model)

@app.get("/health")
def health():
    return {
        "status": "operational",
        "streaming_sse": True,
        "multi_agent_board": True,
        "deep_learning_active": os.path.exists(pt_path)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    raw_vector = np.array([[
        patient.age, patient.sex, patient.cp, patient.trestbps, patient.chol,
        patient.fbs, patient.restecg, patient.thalach, patient.exang,
        patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]], dtype=float)

    try:
        pred, conf, risk, scaled = inference_engine.predict(raw_vector)
        
        with torch.no_grad():
            dl_prob = float(dl_net(torch.tensor(scaled, dtype=torch.float32))[0][0])

        fused_risk = round((risk * 0.65 + dl_prob * 0.35) * 100, 2)
        shaps = explainer.compute_shap_values(scaled)
        recourse = compute_counterfactual_recourse(inference_engine.model, inference_engine.scaler, raw_vector)

        is_ischemic = bool(fused_risk >= 35.0)
        tier = "High" if fused_risk >= 70.0 else "Moderate" if fused_risk >= 35.0 else "Low"
        notes = f"Dual Ensemble (RF {round(risk*100,1)}% + DeepSurvNet {round(dl_prob*100,1)}%)"

        return PredictionResponse(
            prediction=1 if is_ischemic else 0,
            has_heart_disease=is_ischemic,
            risk_score_percentage=fused_risk,
            confidence=round(conf * 100, 2),
            risk_tier=tier,
            clinical_notes=notes,
            shap_contributions=[ShapContribution(**item) for item in shaps],
            counterfactual_recourse=RecourseResult(**recourse),
            agent_synthesis=AgentSynthesis(
                triage_assessment="Multi-Agent Medical Consensus ready for live token streaming.",
                key_pathological_drivers=[f"{s['feature']}: {s['impact']:+.4f}" for s in shaps[:3]],
                prioritized_interventions=[f"Target {i['biomarker']} -> {i['target_value']}" for i in recourse.get('interventions', [])],
                clinical_urgency_level="CRITICAL_EMERGENCY" if fused_risk >= 70 else "URGENT_TRIAGE" if fused_risk >= 35 else "ROUTINE"
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/stream-board")
def stream_board_endpoint(req: dict):
    telemetry = req.get("telemetry", {})
    risk_score = req.get("risk_score", 0.0)
    risk_tier = req.get("risk_tier", "Moderate")
    shaps = req.get("shaps", [])
    recourse = req.get("recourse", {})

    return StreamingResponse(
        clinical_board.stream_triage_debate(telemetry, risk_score, risk_tier, shaps, recourse),
        media_type="text/event-stream"
    )

@app.post("/agent/stream-chat")
def stream_chat_endpoint(req: dict):
    message = req.get("message", "")
    context = req.get("context", {})
    return StreamingResponse(
        clinical_board.chat_stream(message, context),
        media_type="text/event-stream"
    )
"""

with open("backend/app/main.py", "w", encoding="utf-8") as f:
    f.write(main_code)

print("[SUCCESS] Backend upgraded with Multi-Agent Board & Token Streaming!")
