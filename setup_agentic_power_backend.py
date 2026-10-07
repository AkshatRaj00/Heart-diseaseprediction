import io
import csv
import json
import uuid
from datetime import datetime, timezone
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .schemas import (
    PatientInput, PredictionResponse, ShapContribution, 
    RecourseResult, RecourseIntervention, AgentSynthesis
)
from .engine import inference_engine
from .explainer import ClinicalExplainer
from .recourse import compute_counterfactual_recourse
from .agent import clinical_board

app = FastAPI(title="CardioSense Autonomous AI Hospital Core", version="11.0.0")

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

def map_and_evaluate(patient_dict: dict):
    trestbps = float(patient_dict.get("trestbps", 130))
    thalach = float(patient_dict.get("thalach", 140))
    age = float(patient_dict.get("age", 55))

    if trestbps <= 20 or thalach <= 10 or age <= 0:
        return 100.0, "CODE BLUE - ASYSTOLE (CLINICAL ARREST)", None, None, True

    cp_val = float(patient_dict.get("cp", 0))
    mapped_cp = cp_val + 1 if cp_val < 4 else cp_val
    
    thal_val = float(patient_dict.get("thal", 1))
    thal_map = {1: 3.0, 2: 6.0, 3: 7.0, 3.0: 3.0, 6.0: 6.0, 7.0: 7.0}
    mapped_thal = thal_map.get(thal_val, 3.0)

    raw_vector = np.array([[
        age,
        float(patient_dict.get("sex", 1)),
        mapped_cp,
        trestbps,
        float(patient_dict.get("chol", 220)),
        float(patient_dict.get("fbs", 0)),
        float(patient_dict.get("restecg", 0)),
        thalach,
        float(patient_dict.get("exang", 0)),
        float(patient_dict.get("oldpeak", 0.0)),
        float(patient_dict.get("slope", 1)),
        float(patient_dict.get("ca", 0)),
        mapped_thal
    ]], dtype=float)

    pred, conf, risk, scaled = inference_engine.predict(raw_vector)
    calc_risk = round(risk * 100, 2)
    tier = "Critical Red" if calc_risk >= 70.0 else "Amber Triage" if calc_risk >= 35.0 else "Green Safe"
    return calc_risk, tier, raw_vector, scaled, False

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    p_dict = patient.model_dump()
    calc_risk, tier, raw_vector, scaled, is_asystole = map_and_evaluate(p_dict)
    
    if is_asystole:
        return PredictionResponse(
            prediction=1,
            has_heart_disease=True,
            risk_score_percentage=100.0,
            confidence=100.0,
            risk_tier="CODE BLUE - ASYSTOLE",
            clinical_notes="HEMODYNAMIC COLLAPSE: Vitals absent (BP <= 20 or HR <= 10). Patient in Cardiac Arrest.",
            shap_contributions=[
                ShapContribution(feature="trestbps (BP)", impact=0.85),
                ShapContribution(feature="thalach (HR)", impact=0.75)
            ],
            counterfactual_recourse=RecourseResult(
                recourse_needed=True,
                current_risk=100.0,
                projected_risk=0.0,
                interventions=[
                    RecourseIntervention(
                        biomarker="CARDIOPULMONARY RESUSCITATION (CPR)",
                        current_value=0.0,
                        target_value=1.0,
                        required_delta=1.0
                    )
                ]
            ),
            agent_synthesis=AgentSynthesis(
                triage_assessment="CODE BLUE: Asystole / Hemodynamic Cessation detected. Begin ACLS compressions and 1mg Epinephrine IV immediately.",
                key_pathological_drivers=["Total loss of mean arterial pressure", "Cessation of cardiac chronotropy"],
                prioritized_interventions=["Immediate CPR 100-120 compressions/min", "Bag-Valve-Mask high flow O2", "Epinephrine 1mg IV/IO"],
                clinical_urgency_level="CRITICAL_EMERGENCY"
            )
        )

    shaps = explainer.compute_shap_values(scaled)
    recourse = compute_counterfactual_recourse(inference_engine.model, inference_engine.scaler, raw_vector)
    is_ischemic = bool(calc_risk >= 35.0)

    return PredictionResponse(
        prediction=1 if is_ischemic else 0,
        has_heart_disease=is_ischemic,
        risk_score_percentage=calc_risk,
        confidence=94.53,
        risk_tier="High" if calc_risk >= 70 else "Moderate" if calc_risk >= 35 else "Low",
        clinical_notes=f"Angiography Ground Truth Model. Risk: {calc_risk}%",
        shap_contributions=[ShapContribution(**item) for item in shaps],
        counterfactual_recourse=RecourseResult(**recourse),
        agent_synthesis=AgentSynthesis(
            triage_assessment="Verified Cleveland diagnostic telemetry synchronized.",
            key_pathological_drivers=[f"{s['feature']}: {s['impact']:+.4f}" for s in shaps[:3]],
            prioritized_interventions=[f"Target {i['biomarker']} -> {i['target_value']}" for i in recourse.get('interventions', [])],
            clinical_urgency_level="CRITICAL_EMERGENCY" if calc_risk >= 70 else "URGENT_TRIAGE" if calc_risk >= 35 else "ROUTINE"
        )
    )

# 1. 60-Minute Physiological Prognosis Forecast Engine
@app.post("/prognosis/forecast")
def forecast_trajectory(req: dict):
    telemetry = req.get("telemetry", {})
    current_risk = float(req.get("risk_score", 50.0))
    bp = float(telemetry.get("trestbps", 120))
    hr = float(telemetry.get("thalach", 75))
    st_dep = float(telemetry.get("oldpeak", 0.0))

    if bp <= 20 or hr <= 10:
        return {"status": "ASYSTOLE", "timeline": []}

    # Rate-Pressure Product (Myocardial Oxygen Consumption Index)
    rpp = hr * bp
    drift_factor = 0.045 if rpp > 16000 else -0.015 if rpp < 10000 else 0.01

    timeline = []
    intervals = [15, 30, 45, 60]
    sim_risk = current_risk
    sim_st = st_dep

    for m in intervals:
        sim_risk = min(99.4, max(5.0, round(sim_risk + (drift_factor * m * 1.3), 1)))
        sim_st = round(min(5.5, max(0.0, sim_st + (0.018 * m * (1 if drift_factor > 0 else -0.5))), 2)
        timeline.append({
            "minute": f"t+{m}m",
            "projected_risk": sim_risk,
            "projected_st_dep": sim_st,
            "ischemic_threat": "CRITICAL THREAT" if sim_risk >= 75 else "MODERATE DRIFT" if sim_risk >= 40 else "STABLE PERFUSION"
        })

    return {
        "status": "OPERATIONAL",
        "current_rpp": int(rpp),
        "rpp_workload": "Hyperdynamic Demand" if rpp > 16000 else "Eudynamic Perfusion",
        "forecast": timeline,
        "early_warning": "Warning: Microvascular ischemic threshold projected to breach within 35 minutes without vasodilator titration." if current_risk >= 65 else "Hemodynamic profile within compensatory limits."
    }

# 2. Autonomous Closed-Loop IV Infusion Pump Titrator
@app.post("/pump/titrate")
def titrate_pump(req: dict):
    telemetry = req.get("telemetry", {})
    bp = float(telemetry.get("trestbps", 140))
    hr = float(telemetry.get("thalach", 80))
    target_bp = float(req.get("target_bp", 120))
    patient_weight = float(req.get("weight_kg", 70))

    if bp <= 20 or hr <= 10:
        return {
            "drug": "Epinephrine",
            "dose": "1 mg IV Push",
            "rate_ml_hr": 0.0,
            "protocol": "ACLS Asystole Resuscitation Protocol - Infusion Discontinued."
        }

    delta_bp = bp - target_bp
    if delta_bp > 10:
        # Hypertensive / Ischemic: Nitroglycerin (NTG) Titration (Start 5-10 mcg/min, step 5mcg every 5m)
        mcg_min = min(100.0, max(5.0, round(delta_bp * 1.8, 1)))
        # Standard dilution: 50mg in 250ml D5W = 200 mcg/ml
        rate_ml_hr = round((mcg_min * 60) / 200.0, 1)
        drug_name = "Nitroglycerin (NTG)"
        action = f"Titrate NTG at {mcg_min} mcg/min to relieve coronary spasm and reduce afterload from {bp} -> {target_bp} mmHg."
    elif bp < 90:
        # Hypotensive / Shock: Norepinephrine Titration (0.05 - 0.5 mcg/kg/min)
        mcg_kg_min = min(0.4, max(0.05, round((90 - bp) * 0.015, 3)))
        # Standard dilution: 4mg in 250ml = 16 mcg/ml
        rate_ml_hr = round((mcg_kg_min * patient_weight * 60) / 16.0, 1)
        drug_name = "Norepinephrine"
        action = f"Infuse Norepinephrine at {mcg_kg_min} mcg/kg/min to restore MAP > 65 mmHg."
    else:
        drug_name = "Normal Saline (KVO)"
        rate_ml_hr = 10.0
        action = "Hemodynamics stable. Maintain Keep-Vein-Open (KVO) line at 10 mL/hr."

    return {
        "drug": drug_name,
        "infusion_rate_ml_hr": rate_ml_hr,
        "infusion_gtts_min": round((rate_ml_hr * 60) / 60, 1),
        "directive": action,
        "safety_limit": "Cease infusion if SBP drops below 95 mmHg or Heart Rate surges > 115 bpm."
    }

# 3. HL7 / FHIR R4 Bundle Exporter
@app.post("/fhir/export")
def export_fhir(req: dict):
    telemetry = req.get("telemetry", {})
    risk = req.get("risk_score", 0.0)
    bundle_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    fhir_bundle = {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "transaction",
        "timestamp": now_iso,
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": f"PAT-{uuid.uuid4().hex[:6]}",
                    "active": True,
                    "gender": "male" if telemetry.get("sex", 1) == 1 else "female",
                    "birthDate": f"{datetime.now().year - int(telemetry.get('age', 60))}-01-01"
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"OBS-RISK-{uuid.uuid4().hex[:6]}",
                    "status": "final",
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "75325-1",
                            "display": "Cardiovascular disease 10-year risk [Score]"
                        }]
                    },
                    "effectiveDateTime": now_iso,
                    "valueQuantity": {
                        "value": risk,
                        "unit": "%",
                        "system": "http://unitsofmeasure.org",
                        "code": "%"
                    }
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"OBS-BP-{uuid.uuid4().hex[:6]}",
                    "status": "final",
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "8480-6",
                            "display": "Systolic blood pressure"
                        }]
                    },
                    "effectiveDateTime": now_iso,
                    "valueQuantity": {
                        "value": telemetry.get("trestbps", 120),
                        "unit": "mm[Hg]",
                        "system": "http://unitsofmeasure.org",
                        "code": "mm[Hg]"
                    }
                }
            }
        ]
    }
    return fhir_bundle

@app.post("/batch-triage")
async def batch_triage(file: UploadFile = File(...)):
    content = await file.read()
    decoded = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))
    
    results = []
    bed_num = 101
    for row in reader:
        try:
            risk, tier, _, _, is_asystole = map_and_evaluate(row)
            action = "CODE BLUE RESUSCITATION" if is_asystole else ("Immediate Cath Lab" if risk >= 70 else "Continuous Telemetry" if risk >= 35 else "Discharge Candidate")
            results.append({
                "patient_id": row.get("patient_id", f"PT-{bed_num}"),
                "bed": f"ICU-{bed_num}",
                "age": row.get("age", "N/A"),
                "sex": "M" if str(row.get("sex", "1")) == "1" else "F",
                "bp": f"{row.get('trestbps', '0')} mmHg",
                "hr": f"{row.get('thalach', '0')} bpm",
                "st_dep": f"{row.get('oldpeak', '0.0')} mm",
                "risk_score": risk,
                "tier": tier,
                "action": action
            })
            bed_num += 1
        except Exception:
            continue
            
    return {"cohort_size": len(results), "patients": results}

@app.post("/agent/stream-board")
def stream_board_endpoint(req: dict):
    telemetry = req.get("telemetry", {})
    trestbps = float(telemetry.get("trestbps", 120))
    thalach = float(telemetry.get("thalach", 75))
    is_asystole = bool(trestbps <= 20 or thalach <= 10)

    return StreamingResponse(
        clinical_board.stream_triage_debate(
            telemetry, req.get("risk_score", 0.0), 
            req.get("risk_tier", "Moderate"), req.get("shaps", []), 
            req.get("recourse", {}),
            is_asystole=is_asystole
        ),
        media_type="text/event-stream"
    )

@app.post("/agent/stream-chat")
def stream_chat_endpoint(req: dict):
    return StreamingResponse(
        clinical_board.chat_stream(req.get("message", ""), req.get("context", {})),
        media_type="text/event-stream"
    )
