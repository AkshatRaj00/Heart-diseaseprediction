from pydantic import BaseModel, Field
from typing import List, Optional

class PatientInput(BaseModel):
    age: float = Field(..., ge=0, description="Age in years")
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=0, description="Resting BP mm Hg")
    chol: float = Field(..., ge=0, description="Serum Cholesterol mg/dl")
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=0, description="Max HR bpm")
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0.0)
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

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    context: dict
    history: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    reply: str
