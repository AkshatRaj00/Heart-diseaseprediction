# ❤️ CardioAI Predictor

<p align="center">
  <strong>AI-Powered Heart Disease Risk Assessment</strong><br/>
  Machine Learning • Explainable AI • Streamlit
</p>

<p align="center">

<a href="https://heart-diseaseprediction-zhl7t64qx8w9h5zna3vrce.streamlit.app/">
<img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-Streamlit-FF4B4B?style=for-the-badge"/>
</a>

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/ML-Random%20Forest-00A86B?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Explainability-SHAP-8A2BE2?style=for-the-badge"/>

</p>

---

## 🎬 Live Demo

<p align="center">

<a href="https://heart-diseaseprediction-zhl7t64qx8w9h5zna3vrce.streamlit.app/">
<img src="https://img.shields.io/badge/▶%20OPEN%20CARDIOAI%20LIVE-EF4444?style=for-the-badge"/>
</a>

</p>

---

## 🖥️ Real App Preview

<p align="center">
<img src="https://github.com/user-attachments/assets/8c6f4d38-6810-4d82-a9ee-fe03012dacd1" width="90%"/>
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/eb4ba3aa-5aba-4b62-8ae6-a7bb1352055b" width="90%"/>
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/5e8e78f7-06d7-49cf-9965-d97e7776a576" width="90%"/>
</p>

---

## 🧠 How It Works

```mermaid
flowchart TD
    A([👤 User]) --> B[📝 Enter Health Parameters]
    B --> C[⚙️ Data Preprocessing]
    C --> D[📊 Feature Scaling]
    D --> E[🌲 Random Forest Model]

    E --> F{Risk Analysis}

    F --> G[🟢 Lower Estimated Risk]
    F --> H[🔴 Higher Estimated Risk]

    E --> I[🔍 SHAP Explainability]
    I --> J[📈 Feature Impact]

    G --> K[📊 Risk Dashboard]
    H --> K
    J --> K

    K --> L([💡 Result])
```

---

## 🔥 ML Pipeline

```text
Patient Data
     │
     ▼
┌───────────────┐
│ Preprocessing │
└───────┬───────┘
        ▼
┌───────────────┐
│ Feature Scale │
└───────┬───────┘
        ▼
┌────────────────┐
│ Random Forest  │
│    Classifier  │
└───────┬────────┘
        │
   ┌────┴────┐
   ▼         ▼
Risk Score  Prediction
   │         │
   └────┬────┘
        ▼
┌────────────────┐
│ SHAP Analysis  │
└───────┬────────┘
        ▼
📊 Explainable Dashboard
```

---

## ⚡ Features

| Feature | Description |
|---|---|
| 🧠 ML Prediction | Random Forest based risk estimation |
| 📊 Risk Dashboard | Probability & confidence visualization |
| 🔍 SHAP | Explains feature contribution |
| 🔐 Privacy First | No persistent user data storage |
| ⚡ Streamlit | Interactive web interface |
| 📈 Visual Analytics | Clear prediction insights |

---

## 🛠️ Tech Stack

```text
🐍 Python
│
├── Streamlit
├── Scikit-Learn
├── Pandas
├── NumPy
├── Matplotlib
├── Plotly
└── SHAP
```

---

## 📁 Project Structure

```text
Heart-diseaseprediction/
│
├── app.py
├── train_and_save_model.py
│
├── heart_disease_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── model_metadata.pkl
├── feature_importance.csv
└── X_train_scaled_sample.pkl
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/AkshatRaj00/Heart-diseaseprediction.git

cd Heart-diseaseprediction

pip install -r requirements.txt

streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 📊 Model

The project uses a **Random Forest Classifier** trained on clinical health parameters to produce an estimated heart-disease risk probability. The application also uses **SHAP-based explainability** to show how input features influence the model output.

---

## ⚠️ Medical Disclaimer

> This project is for **educational and informational purposes only**.  
> It is **not a medical diagnostic system** and should not replace professional medical advice, diagnosis, or treatment.

---

<p align="center">

### ❤️ CardioAI Predictor

<strong>Turning health data into explainable ML insights.</strong>

<br/><br/>

Built with 🧠 Machine Learning + ❤️ Technology

</p>
