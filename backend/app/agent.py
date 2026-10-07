import os
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

    def stream_triage_debate(self, telemetry: Dict[str, Any], risk_score: float, risk_tier: str, shaps: List[Dict[str, Any]], recourse: Dict[str, Any], is_asystole: bool = False) -> Generator[str, None, None]:
        if is_asystole:
            asystole_text = (
                "### [CODE BLUE - IMMEDIATE EMERGENCY RESUSCITATION]\n"
                "CRITICAL ALERT: Vitals reflect hemodynamic cessation (BP=0, HR=0). Patient in ASYSTOLE / CLINICAL ARREST.\n\n"
                "### [ACLS RESUSCITATION DIRECTIVE]\n"
                "- Initiate immediate closed-chest cardiac compressions (100-120 bpm, 2-2.4 inches depth).\n"
                "- Airway management: High-flow oxygenation, Bag-Valve-Mask (BVM), endotracheal intubation.\n"
                "- Pharmacotherapy: Epinephrine 1 mg IV/IO every 3-5 minutes immediately.\n"
                "- Verify asystole in two leads; check for reversible causes (5 Hs & 5 Ts: Hypovolemia, Hypoxia, Hydrogen ion acidosis, Hypo/Hyperkalemia, Hypothermia; Tension pneumothorax, Tamponade, Toxins, Thrombosis pulmonary/coronary).\n"
                "- DEFIBRILLATION CONTRAINDICATED for Asystole (Non-shockable rhythm).\n"
            )
            for word in asystole_text.split(" "):
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                time.sleep(0.015)
            yield "data: [DONE]\n\n"
            return

        board_prompt = f"""Patient Clinical Telemetry:
{json.dumps(telemetry, indent=2)}

Risk Metrics: Fused Score {risk_score}% ({risk_tier})
Top Contributors: {json.dumps(shaps[:4])}
Recourse: {json.dumps(recourse)}

Generate strict 4-Section Clinical Board Deliberation:
1. [DR. CARDIOLOGIST]: Hemodynamic strain and coronary perfusion analysis.
2. [DR. PHARMACOLOGIST - CLINICAL RX]: Tailor EXACT drug names, exact dosages, and scheduling (Statins, Antiplatelets, Beta-blockers/ACEi) based on BP={telemetry.get('trestbps')} and HR={telemetry.get('thalach')}. Explicitly state any contraindicated drugs.
3. [CARDIAC NUTRITIONIST & DIETITIAN]: Exact personalized diet chart (Sodium limit, recommended specific foods, and strictly prohibited items).
4. [DR. SAFETY AUDITOR]: Final safety clearance and red flag emergency symptoms.
"""
        if self.client:
            try:
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": board_prompt}],
                    temperature=0.2,
                    max_tokens=1000,
                    stream=True
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield f"data: {json.dumps({'token': delta})}\n\n"
                yield "data: [DONE]\n\n"
                return
            except Exception:
                pass

        # Deterministic Clinical Guidelines Engine (AHA/ACC Ground-Truth Rules)
        hr = float(telemetry.get("thalach", 75))
        bp = float(telemetry.get("trestbps", 120))
        st_dep = float(telemetry.get("oldpeak", 0))

        # Precision Pharmacotherapy Logic
        rx_lines = []
        if risk_score >= 70 or st_dep >= 1.5:
            rx_lines.append("- Ecosprin (Aspirin) 75 mg PO once daily post-lunch (Antiplatelet).")
            rx_lines.append("- Clopidogrel 75 mg PO once daily (Dual Antiplatelet Therapy for severe ischemia).")
            rx_lines.append("- Atorvastatin 80 mg PO once daily at bedtime (High-intensity lipid stabilization).")
        elif risk_score >= 35:
            rx_lines.append("- Ecosprin (Aspirin) 75 mg PO once daily post-lunch.")
            rx_lines.append("- Atorvastatin 40 mg PO once daily at bedtime.")
        else:
            rx_lines.append("- Primary prevention: No antiplatelet indicated without active vascular event.")
            rx_lines.append("- Rosuvastatin 10 mg PO once daily if LDL > 100 mg/dL.")

        if bp > 130:
            rx_lines.append(f"- Telmisartan 40 mg PO once daily in morning (ARBs for BP={bp} mmHg reduction).")

        if hr > 80 and bp > 110:
            rx_lines.append(f"- Metoprolol Succinate ER 25 mg PO once daily (Beta-blocker to control rate={hr} bpm).")
        elif hr < 60:
            rx_lines.append(f"- [CONTRAINDICATION WARNING]: Beta-blockers STRICTLY WITHHELD due to bradycardia (HR={hr} bpm).")

        if st_dep >= 1.0:
            rx_lines.append("- Sorbitrate (Isosorbide Dinitrate) 5 mg SL PRN for acute breakthrough angina episodes.")

        # Precision Medical Nutrition Therapy Logic
        diet_lines = []
        if bp > 130:
            diet_lines.append("- Strict DASH Protocol: Dietary Sodium < 1,500 mg/day (cut all table salt, pickles, papad).")
        else:
            diet_lines.append("- Standard Sodium Protocol: Dietary Sodium < 2,300 mg/day.")

        if float(telemetry.get("chol", 200)) > 200 or risk_score >= 35:
            diet_lines.append("- Cholesterol Control: Zero trans-fats, eliminate palm oil, vanaspati, and deep-fried items.")
            diet_lines.append("- Prescribed Functional Foods: 25g raw crushed flaxseeds daily (ALA Omega-3), 2 cloves raw garlic.")
            diet_lines.append("- Soluble Fiber Target: Steel-cut oats (minimum 35g fiber/day to absorb intestinal bile salts).")
        else:
            diet_lines.append("- Heart-Healthy Maintenance: Mediterranean-style diet, 5 servings colorful vegetables daily.")

        diet_lines.append("- Hydration: Fluid regulation 2.0 - 2.5 Liters/day (adjust if renal clearance is impaired).")

        stream_body = (
            f"### [DR. CARDIOLOGIST - CLINICAL TRIAGE]\n"
            f"Patient evaluated at {risk_score}% ischemic risk ({risk_tier}). "
            f"Resting BP is {bp} mmHg with achieved heart rate of {hr} bpm. "
            f"ST depression measures {st_dep} mm, indicating substantial subendocardial oxygen supply-demand mismatch.\n\n"
            f"### [DR. PHARMACOLOGIST - PERSONALIZED CLINICAL RX]\n"
            + "\n".join(rx_lines) + "\n\n"
            f"### [CARDIAC DIETITIAN - MEDICAL NUTRITION PROTOCOL]\n"
            + "\n".join(diet_lines) + "\n\n"
            f"### [DR. SAFETY AUDITOR - DISCHARGE CLEARANCE]\n"
            f"- RED FLAGS: If patient experiences squeezing retrosternal chest pain radiating to left arm/jaw, diaphoresis (cold sweating), or syncope, immediately administer chewable Aspirin 300mg and mobilize to Cath Lab.\n"
            f"- Follow-up: Repeat 12-lead ECG and serum troponin-I at 6 hours.\n"
        )

        for word in stream_body.split(" "):
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            time.sleep(0.015)
        yield "data: [DONE]\n\n"

    def chat_stream(self, message: str, context: Dict[str, Any]) -> Generator[str, None, None]:
        if self.client:
            try:
                system_prompt = f"You are Dr. CardioSense Copilot, an elite Senior Cardiologist AI. Telemetry: {json.dumps(context)}. Answer concisely and medically."
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
                        yield f"data: {json.dumps({'token': delta})}\n\n"
                yield "data: [DONE]\n\n"
                return
            except Exception:
                pass

        reply = f"[Clinical Specialist Copilot]: For BP={context.get('telemetry',{}).get('trestbps')} and Risk={context.get('risk_score')}%, strictly follow the dual-antiplatelet and low-sodium protocol as generated in Tab 6."
        for word in reply.split(" "):
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            time.sleep(0.02)
        yield "data: [DONE]\n\n"

clinical_board = ClinicalMedicalBoard()
