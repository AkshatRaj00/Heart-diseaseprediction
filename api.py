import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="CardioSense AI - Heart Disease Risk Engine",
    description="Production-grade clinical decision support API using ensemble tree classification.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = None
scaler = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"[ERROR] Failed to load model artifacts: {e}")

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

class PatientRecord(BaseModel):
    age: float = Field(..., ge=18, le=120, description="Age in years", example=55)
    sex: int = Field(..., ge=0, le=1, description="0 = Female, 1 = Male", example=1)
    cp: int = Field(..., ge=0, le=3, description="Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)", example=2)
    trestbps: float = Field(..., ge=80, le=240, description="Resting Blood Pressure (mm Hg)", example=130)
    chol: float = Field(..., ge=100, le=600, description="Serum Cholesterol (mg/dl)", example=250)
    fbs: int = Field(..., ge=0, le=1, description="Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)", example=0)
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG Results (0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy)", example=1)
    thalach: float = Field(..., ge=60, le=240, description="Maximum Heart Rate Achieved", example=155)
    exang: int = Field(..., ge=0, le=1, description="Exercise Induced Angina (1: Yes, 0: No)", example=0)
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise", example=1.2)
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)", example=1)
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels colored by flourosopy (0-3)", example=0)
    thal: int = Field(..., ge=1, le=3, description="Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect)", example=2)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/features")
def get_features():
    return {"features": FEATURE_NAMES}

@app.post("/predict")
def predict_cardiac_risk(patient: PatientRecord):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model runtime artifacts not loaded.")
    
    raw_vector = np.array([[
        patient.age, patient.sex, patient.cp, patient.trestbps, patient.chol,
        patient.fbs, patient.restecg, patient.thalach, patient.exang,
        patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]], dtype=float)

    try:
        scaled_vector = scaler.transform(raw_vector)
        pred_label = int(model.predict(scaled_vector)[0])
        prob_matrix = model.predict_proba(scaled_vector)[0]
        confidence = float(prob_matrix[pred_label])
        disease_probability = float(prob_matrix[1]) if len(prob_matrix) > 1 else float(pred_label)

        if disease_probability < 0.35:
            risk_tier = "Low"
            recommendation = "Normal cardiac profile indicated. Maintain balanced routine and scheduled wellness checks."
        elif disease_probability < 0.70:
            risk_tier = "Moderate"
            recommendation = "Elevated risk markers observed. Diagnostic lipid workup and clinical consultation recommended."
        else:
            risk_tier = "High"
            recommendation = "Significant cardiac indicators detected. Comprehensive cardiovascular examination strongly advised."

        return {
            "prediction": pred_label,
            "has_heart_disease": bool(pred_label == 1),
            "risk_score_percentage": round(disease_probability * 100, 2),
            "confidence": round(confidence * 100, 2),
            "risk_tier": risk_tier,
            "clinical_notes": recommendation
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(err)}")

STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "CardioSense API online. Visit /docs for Swagger UI documentation."}
