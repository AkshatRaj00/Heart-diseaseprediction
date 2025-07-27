# app.py - Professional Heart Disease Predictor with Fixed SHAP

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# SHAP library for model explainability
import shap
# streamlit_shap for displaying SHAP plots in Streamlit
import streamlit_shap

# --- Page Configuration ---
st.set_page_config(
    page_title="OnePersonAI - Heart Disease Predictor",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Professional CSS Styling ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 15px 35px rgba(30, 60, 114, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    .main-header h1 {
        font-size: 3.2em;
        font-weight: 800;
        margin-bottom: 10px;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1.3em;
        font-weight: 400;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    /* Card Styles */
    .info-box, .success-box, .warning-box, .danger-box {
        padding: 25px;
        border-radius: 16px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left: 5px solid #ff9800;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 15px 30px;
        font-size: 1.1em;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Primary Button */
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.3);
    }
    
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #ff5252 0%, #d63031 100%);
        box-shadow: 0 12px 30px rgba(255, 107, 107, 0.4);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9ff 0%, #e8eaf6 100%);
        border-right: 3px solid #667eea;
    }
    
    .css-1lcbmhc {
        font-weight: 700;
        color: #1e3c72;
        font-size: 1.1em;
    }
    
    /* Metric Styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        border: 2px solid #e9ecef;
        transition: transform 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.0em;
        font-weight: 600;
        color: #495057;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2em;
        font-weight: 800;
        color: #1e3c72;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f1f3f4 0%, #e8eaf6 100%);
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Animations */
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Company Footer */
    .company-footer {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin-top: 40px;
        box-shadow: 0 10px 30px rgba(30, 60, 114, 0.2);
    }
    
    .company-logo {
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 10px;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Disclaimer Box */
    .disclaimer-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        border: 2px solid #e17055;
        border-radius: 16px;
        padding: 25px;
        margin: 25px 0;
        box-shadow: 0 8px 20px rgba(225, 112, 85, 0.2);
    }
    
    .disclaimer-box h4 {
        color: #d63031;
        font-weight: 700;
        margin-bottom: 15px;
    }
    
    .disclaimer-box p {
        color: #2d3436;
        font-weight: 500;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# --- Professional Header ---
st.markdown("""
<div class="main-header fade-in-up">
    <h1>🫀 CardioAI Predictor</h1>
    <p>Advanced AI-Powered Heart Disease Risk Assessment</p>
    <small>Powered by OnePersonAI Technologies</small>
</div>
""", unsafe_allow_html=True)

# --- Important Disclaimer ---
st.markdown("""
<div class="disclaimer-box fade-in-up">
    <h4>⚠️ IMPORTANT MEDICAL DISCLAIMER</h4>
    <p><strong>This AI tool is for educational and informational purposes only.</strong> It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical concerns. Never ignore professional medical advice based on information from this tool.</p>
    <p><strong>Emergency:</strong> If experiencing chest pain, difficulty breathing, or other serious symptoms, seek immediate medical attention or call emergency services.</p>
</div>
""", unsafe_allow_html=True)

# --- About Section ---
st.markdown("""
<div class="info-box fade-in-up">
    <h4>🏥 About CardioAI Predictor</h4>
    <p>Our advanced machine learning system analyzes 13 key medical parameters to assess heart disease risk. This tool uses sophisticated algorithms trained on clinical data to provide risk assessment insights.</p>
    <p><strong>Features:</strong> Real-time risk analysis, SHAP explainability, personalized recommendations, and comprehensive risk factor identification.</p>
</div>
""", unsafe_allow_html=True)

# --- Model Loading with Enhanced Error Handling ---
@st.cache_resource
def load_models():
    """Loads models or creates a backup model with proper feature handling"""
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        X_train_scaled_sample = joblib.load('X_train_scaled_sample.pkl')
        return model, scaler, X_train_scaled_sample, True
    except FileNotFoundError:
        # Create backup model with exact 13 features
        np.random.seed(42)
        
        # Feature names matching our input
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        n_samples = 500
        X_sample = np.random.randn(n_samples, len(feature_names))
        
        # Create realistic target based on multiple features
        y_sample = ((X_sample[:, 0] > 0) & 
                   (X_sample[:, 4] > 0) & 
                   (X_sample[:, 7] < 0)).astype(int)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X_sample)
        model.fit(X_scaled, y_sample)
        
        # Create background sample for SHAP
        X_train_scaled_sample = X_scaled[np.random.choice(X_scaled.shape[0], min(100, X_scaled.shape[0]), replace=False)]
        
        return model, scaler, X_train_scaled_sample, False

# Load models
model, scaler, X_train_scaled_sample, is_pretrained = load_models()

# Model status
if is_pretrained:
    st.success("✅ Production model loaded successfully!")
else:
    st.warning("⚠️ Demo mode active. Upload trained model files for full functionality.")

# --- Enhanced Sidebar ---
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
    <h2 style="color: white; text-align: center; margin: 0;">👤 Patient Data</h2>
    <p style="color: rgba(255,255,255,0.8); text-align: center; margin: 5px 0 0 0;">Enter your medical information</p>
</div>
""", unsafe_allow_html=True)

def user_input_features():
    """Enhanced user input collection with better organization"""
    
    # Basic Information
    st.sidebar.markdown("### 📋 Basic Information")
    age = st.sidebar.slider('Age (years)', 29, 77, 48, help="Your current age")
    sex = st.sidebar.selectbox('Gender', ('Male', 'Female'), help="Biological sex")
    
    # Cardiovascular Symptoms
    st.sidebar.markdown("### 💗 Cardiovascular Symptoms")
    cp = st.sidebar.selectbox(
        'Chest Pain Type', 
        ('Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'),
        help="Type of chest pain experienced"
    )
    
    # Vital Signs & Lab Results
    st.sidebar.markdown("### 🩺 Vital Signs & Lab Results")
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 129, 
                                help="Blood pressure at rest")
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 240,
                            help="Total cholesterol level")
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ('No', 'Yes'),
                              help="Is fasting blood sugar above 120?")
    
    # Cardiac Tests
    st.sidebar.markdown("### 🫀 Cardiac Test Results")
    restecg = st.sidebar.selectbox(
        'Resting ECG Results', 
        ('Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'),
        help="Electrocardiogram results at rest"
    )
    thalach = st.sidebar.slider('Maximum Heart Rate (bpm)', 71, 202, 150,
                               help="Highest heart rate achieved")
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('No', 'Yes'),
                                help="Chest pain during exercise")
    
    # Advanced Cardiac Parameters
    st.sidebar.markdown("### 🔬 Advanced Parameters")
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.2, 1.0, step=0.1,
                               help="ST depression induced by exercise")
    slope = st.sidebar.selectbox('ST Segment Slope', ('Upsloping', 'Flat', 'Downsloping'),
                                help="Slope of peak exercise ST segment")
    ca = st.sidebar.slider('Major Vessels (0-3)', 0, 3, 0,
                          help="Number of major vessels colored by fluoroscopy")
    thal = st.sidebar.selectbox('Thalassemia', ('Normal', 'Fixed Defect', 'Reversible Defect'),
                               help="Thalassemia test result")

    # Mapping dictionaries
    sex_map = {'Male': 1, 'Female': 0}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'No': 0, 'Yes': 1}
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    exang_map = {'No': 0, 'Yes': 1}
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}

    # Create feature data
    data = {
        'age': age, 'sex': sex_map[sex], 'cp': cp_map[cp], 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs_map[fbs], 'restecg': restecg_map[restecg],
        'thalach': thalach, 'exang': exang_map[exang], 'oldpeak': oldpeak,
        'slope': slope_map[slope], 'ca': ca, 'thal': thal_map[thal]
    }
    
    # Raw data for display
    raw_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg,
        'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }
    
    features = pd.DataFrame(data, index=[0])
    return features, raw_data

# Get user input
input_df, raw_data = user_input_features()

# --- Enhanced Input Summary ---
st.markdown("## 📊 Patient Profile Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **👤 Demographics:**
    - **Age:** {} years
    - **Gender:** {}
    - **Chest Pain:** {}
    """.format(raw_data['age'], raw_data['sex'], raw_data['cp']))

with col2:
    st.markdown("""
    **🩺 Vital Signs:**
    - **BP:** {} mm Hg
    - **Cholesterol:** {} mg/dl
    - **Max HR:** {} bpm
    """.format(raw_data['trestbps'], raw_data['chol'], raw_data['thalach']))

with col3:
    st.markdown("""
    **🔬 Test Results:**
    - **ECG:** {}
    - **Exercise Angina:** {}
    - **ST Depression:** {}
    """.format(raw_data['restecg'], raw_data['exang'], raw_data['oldpeak']))

# --- Enhanced Risk Assessment ---
def calculate_comprehensive_risk_score(data):
    """Enhanced risk calculation with more factors"""
    risk_score = 0
    risk_factors = []
    protective_factors = []
    
    # Age risk
    if data['age'] > 65:
        risk_score += 3
        risk_factors.append(f"Advanced age ({data['age']} years) - High risk")
    elif data['age'] > 55:
        risk_score += 2
        risk_factors.append(f"Elevated age ({data['age']} years) - Moderate risk")
    elif data['age'] < 40:
        protective_factors.append(f"Young age ({data['age']} years) - Protective")
    
    # Gender risk
    if data['sex'] == 'Male':
        risk_score += 1
        risk_factors.append("Male gender (statistically higher risk)")
    else:
        protective_factors.append("Female gender (lower pre-menopausal risk)")
    
    # Chest pain assessment
    if data['cp'] == 'Asymptomatic':
        risk_score += 3
        risk_factors.append("Asymptomatic presentation (concerning)")
    elif data['cp'] == 'Typical Angina':
        risk_score += 2
        risk_factors.append("Typical angina pattern")
    
    # Blood pressure
    if data['trestbps'] > 160:
        risk_score += 3
        risk_factors.append(f"Severe hypertension ({data['trestbps']} mm Hg)")
    elif data['trestbps'] > 140:
        risk_score += 2
        risk_factors.append(f"High blood pressure ({data['trestbps']} mm Hg)")
    elif data['trestbps'] < 120:
        protective_factors.append(f"Optimal blood pressure ({data['trestbps']} mm Hg)")
    
    # Cholesterol
    if data['chol'] > 300:
        risk_score += 3
        risk_factors.append(f"Very high cholesterol ({data['chol']} mg/dl)")
    elif data['chol'] > 240:
        risk_score += 2
        risk_factors.append(f"High cholesterol ({data['chol']} mg/dl)")
    elif data['chol'] < 200:
        protective_factors.append(f"Optimal cholesterol ({data['chol']} mg/dl)")
    
    # Heart rate response
    expected_max_hr = 220 - data['age']
    if data['thalach'] < expected_max_hr * 0.6:
        risk_score += 2
        risk_factors.append(f"Poor heart rate response ({data['thalach']} bpm)")
    elif data['thalach'] > expected_max_hr * 0.85:
        protective_factors.append(f"Good heart rate response ({data['thalach']} bpm)")
    
    # Exercise angina
    if data['exang'] == 'Yes':
        risk_score += 2
        risk_factors.append("Exercise-induced angina (significant)")
    
    # ST depression
    if data['oldpeak'] > 3.0:
        risk_score += 3
        risk_factors.append(f"Severe ST depression ({data['oldpeak']})")
    elif data['oldpeak'] > 1.5:
        risk_score += 2
        risk_factors.append(f"Moderate ST depression ({data['oldpeak']})")
    
    return risk_score, risk_factors, protective_factors

# Calculate enhanced risk
risk_score, risk_factors, protective_factors = calculate_comprehensive_risk_score(raw_data)

# Display risk assessment
st.markdown("## ⚠️ Comprehensive Risk Assessment")

# Risk level determination
if risk_score >= 8:
    risk_level = "🚨 VERY HIGH RISK"
    risk_color = "danger-box"
    risk_message = "Multiple significant risk factors detected. Immediate cardiology consultation strongly recommended."
elif risk_score >= 5:
    risk_level = "⚠️ HIGH RISK"
    risk_color = "warning-box"
    risk_message = "Several risk factors present. Medical evaluation advised soon."
elif risk_score >= 3:
    risk_level = "🔍 MODERATE RISK"
    risk_color = "info-box"
    risk_message = "Some risk factors identified. Regular monitoring recommended."
else:
    risk_level = "✅ LOW RISK"
    risk_color = "success-box"
    risk_message = "Minimal traditional risk factors. Continue healthy lifestyle."

st.markdown(f"""
<div class="{risk_color}">
    <h3>{risk_level}</h3>
    <h4>Risk Score: {risk_score}/15</h4>
    <p>{risk_message}</p>
</div>
""", unsafe_allow_html=True)

# Display risk and protective factors
if risk_factors or protective_factors:
    col1, col2 = st.columns(2)
    
    with col1:
        if risk_factors:
            st.markdown("### 🚨 Risk Factors")
            for factor in risk_factors:
                st.write(f"• {factor}")
    
    with col2:
        if protective_factors:
            st.markdown("### ✅ Protective Factors")
            for factor in protective_factors:
                st.write(f"• {factor}")

# --- Main Prediction Section ---
st.markdown("---")
st.markdown("## 🎯 AI-Powered Risk Prediction")

# Enhanced prediction button
predict_button = st.button('🔬 Analyze Heart Disease Risk', type="primary", use_container_width=True)

if predict_button:
    try:
        with st.spinner('🤖 Processing your medical data with advanced AI...'):
            # Scale input
            scaled_input = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(scaled_input)
            prediction_proba = model.predict_proba(scaled_input)
            
            # Enhanced results display
            st.markdown("### 🎯 AI Prediction Results")
            
            # Main result with enhanced styling
            if prediction[0] == 0:
                st.markdown("""
                <div class="success-box">
                    <h2>🎉 LOW RISK PREDICTION</h2>
                    <h3>AI Assessment: Lower Probability of Heart Disease</h3>
                    <p>The machine learning model indicates a lower likelihood of heart disease based on your current parameters. However, continue monitoring your health and maintaining healthy habits.</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown("""
                <div class="danger-box">
                    <h2>🚨 HIGH RISK PREDICTION</h2>
                    <h3>AI Assessment: Higher Probability of Heart Disease</h3>
                    <p>The model indicates elevated risk factors. <strong>Please consult a cardiologist promptly</strong> for comprehensive evaluation and appropriate medical care.</p>
                </div>
                """, unsafe_allow_html=True)
                st.snow()
            
            # Enhanced probability analysis
            st.markdown("### 📊 Detailed Probability Analysis")
            
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            
            no_disease_prob = prediction_proba[0][0] * 100
            disease_prob = prediction_proba[0][1] * 100
            confidence = max(prediction_proba[0]) * 100
            
            with prob_col1:
                st.metric(
                    label="🟢 No Heart Disease",
                    value=f"{no_disease_prob:.1f}%",
                    delta="Favorable" if no_disease_prob > 50 else "Concerning"
                )
            
            with prob_col2:
                st.metric(
                    label="🔴 Heart Disease Risk",
                    value=f"{disease_prob:.1f}%",
                    delta="High Alert" if disease_prob > 70 else "Monitor" if disease_prob > 50 else "Low"
                )
            
            with prob_col3:
                if confidence > 85:
                    confidence_level = "Very High"
                elif confidence > 70:
                    confidence_level = "High"
                elif confidence > 60:
                    confidence_level = "Moderate"
                else:
                    confidence_level = "Low"
                    
                st.metric(
                    label="🎯 Model Confidence",
                    value=f"{confidence:.1f}%",
                    delta=confidence_level
                )
            
            # Enhanced visualization
            st.markdown("### 📈 Risk Visualization Dashboard")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Probability bar chart
            categories = ['No Disease', 'Heart Disease']
            probabilities = [no_disease_prob, disease_prob]
            colors = ['#4CAF50', '#F44336']
            
            bars = ax1.bar(categories, probabilities, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{prob:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax1.set_ylabel('Probability (%)', fontweight='bold')
            ax1.set_title('Heart Disease Risk Prediction', fontweight='bold', fontsize=14)
            ax1.set_ylim(0, 110)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # Pie chart
            ax2.pie([no_disease_prob, disease_prob], labels=categories, colors=colors,
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax2.set_title('Risk Distribution', fontweight='bold', fontsize=14)
            
            # Risk score gauge
            risk_categories = ['Low\n(0-2)', 'Moderate\n(3-4)', 'High\n(5-7)', 'Very High\n(8+)']
            risk_values = [max(0, min(2, risk_score)), max(0, min(2, risk_score-2)), 
                          max(0, min(3, risk_score-4)), max(0, risk_score-7)]
            risk_colors = ['#4CAF50', '#FF9800', '#FF5722', '#D32F2F']
            
            bars3 = ax3.bar(risk_categories, [2, 2, 3, 8], color=risk_colors, alpha=0.3, edgecolor='white')
            bars3_actual = ax3.bar(risk_categories, risk_values, color=risk_colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            ax3.set_ylabel('Risk Score', fontweight='bold')
            ax3.set_title(f'Your Risk Score: {risk_score}/15', fontweight='bold', fontsize=14)
            ax3.set_ylim(0, 8)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            
            # Confidence meter
            confidence_levels = ['Low', 'Moderate', 'High', 'Very High']
            conf_thresholds = [60, 70, 85, 100]
            conf_colors = ['#FF5722', '#FF9800', '#4CAF50', '#2E7D32']
            
            for i, (level, threshold, color) in enumerate(zip(confidence_levels, conf_thresholds, conf_colors)):
                alpha = 0.8 if confidence >= (conf_thresholds[i-1] if i > 0 else 0) else 0.3
                ax4.barh(level, threshold, color=color, alpha=alpha, edgecolor='white', linewidth=2)
            
            ax4.axvline(x=confidence, color='black', linestyle='--', linewidth=3, label=f'Your Score: {confidence:.1f}%')
            ax4.set_xlabel('Confidence (%)', fontweight='bold')
            ax4.set_title('Model Confidence Level', fontweight='bold', fontsize=14)
            ax4.set_xlim(0, 100)
            ax4.legend()
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Fixed SHAP Explainability Section
            st.markdown("---")
            st.markdown("### 💡 AI Decision Explanation (SHAP Analysis)")
            st.markdown("""
                This analysis shows how each of your medical parameters influenced the AI's prediction.
                **Red bars** increase heart disease risk, **Blue bars** decrease risk.
                The magnitude shows the strength of each factor's influence.
            """)

            try:
                # Ensure we have the correct feature names
                feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                
                # Verify dimensions match
                if scaled_input.shape[1] != len(feature_names):
                    st.error(f"Feature dimension mismatch: Expected {len(feature_names)}, got {scaled_input.shape[1]}")
                    st.stop()
                
                if X_train_scaled_sample.shape[1] != len(feature_names):
                    st.error(f"Background data dimension mismatch: Expected {len(feature_names)}, got {X_train_scaled_sample.shape[1]}")
                    st.stop()
                
                # Create SHAP explainer with proper model wrapper
                def model_predict_proba_wrapper(X):
                    """Wrapper to ensure consistent output format"""
                    proba = model.predict_proba(X)
                    return proba
                
                # Use KernelExplainer with the wrapper
                explainer = shap.KernelExplainer(model_predict_proba_wrapper, X_train_scaled_sample)
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(scaled_input)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    # Binary classification - use class 1 (heart disease) SHAP values
                    shap_values_to_plot = shap_values[1][0]  # First sample, class 1
                    expected_value = explainer.expected_value[1]
                elif isinstance(shap_values, np.ndarray):
                    # Single output format
                    shap_values_to_plot = shap_values[0]
                    expected_value = explainer.expected_value
                else:
                    st.error("Unexpected SHAP values format")
                    st.stop()
                
                # Verify dimensions before creating explanation
                if len(shap_values_to_plot) != len(feature_names):
                    st.error(f"SHAP values dimension mismatch: Expected {len(feature_names)}, got {len(shap_values_to_plot)}")
                    st.stop()
                
                if len(input_df.iloc[0].values) != len(feature_names):
                    st.error(f"Input data dimension mismatch: Expected {len(feature_names)}, got {len(input_df.iloc[0].values)}")
                    st.stop()
                
                # Create properly formatted explanation object
                explanation = shap.Explanation(
                    values=shap_values_to_plot,
                    base_values=expected_value,
                    data=input_df.iloc[0].values,
                    feature_names=feature_names
                )
                
                # Display SHAP force plot
                try:
                    streamlit_shap.st_shap(
                        shap.plots.force(explanation), 
                        height=250, 
                        width=1200
                    )
                except Exception as shap_plot_error:
                    st.warning("SHAP force plot unavailable. Showing alternative explanation.")
                    
                    # Alternative: Create manual SHAP interpretation
                    st.markdown("#### Feature Impact Analysis")
                    
                    # Create DataFrame for manual display
                    shap_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Your Value': input_df.iloc[0].values,
                        'SHAP Impact': shap_values_to_plot,
                        'Raw Input': [f"{raw_data[name]}" for name in feature_names]
                    })
                    
                    # Sort by absolute SHAP impact
                    shap_df['Abs_Impact'] = np.abs(shap_df['SHAP Impact'])
                    shap_df = shap_df.sort_values('Abs_Impact', ascending=False)
                    
                    # Display top features
                    st.markdown("**Top 5 Most Influential Features:**")
                    for idx, row in shap_df.head().iterrows():
                        impact_direction = "increases" if row['SHAP Impact'] > 0 else "decreases"
                        color = "🔴" if row['SHAP Impact'] > 0 else "🔵"
                        st.write(f"{color} **{row['Feature']}** ({row['Raw Input']}): {impact_direction} risk by {abs(row['SHAP Impact']):.3f}")
                
            except Exception as shap_error:
                st.error(f"SHAP analysis unavailable: {str(shap_error)}")
                st.info("This might be due to model compatibility or data format issues. The prediction results above remain valid.")
            
            # Enhanced Recommendations
            st.markdown("---")
            st.markdown("### 💡 Personalized Health Recommendations")
            
            if prediction[0] == 0:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Maintain Your Heart-Healthy Lifestyle</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h5>🏃‍♂️ Physical Activity</h5>
                            <ul>
                                <li>Continue regular exercise (150 min/week)</li>
                                <li>Include both cardio and strength training</li>
                                <li>Daily walking for at least 30 minutes</li>
                            </ul>
                        </div>
                        <div>
                            <h5>🥗 Nutrition</h5>
                            <ul>
                                <li>Mediterranean or DASH diet</li>
                                <li>Limit sodium to <2300mg/day</li>
                                <li>Increase fruits, vegetables, whole grains</li>
                            </ul>
                        </div>
                        <div>
                            <h5>🩺 Health Monitoring</h5>
                            <ul>
                                <li>Annual cardiac check-up</li>
                                <li>Monitor BP, cholesterol yearly</li>
                                <li>Maintain healthy weight (BMI 18.5-24.9)</li>
                            </ul>
                        </div>
                        <div>
                            <h5>🧘‍♀️ Lifestyle</h5>
                            <ul>
                                <li>Stress management techniques</li>
                                <li>Quality sleep (7-9 hours)</li>
                                <li>Avoid smoking, limit alcohol</li>
                            </ul>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="danger-box">
                    <h4>🚨 URGENT - Immediate Action Plan</h4>
                    <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <h5>🏥 Immediate Medical Steps (Within 1-2 weeks)</h5>
                        <ul>
                            <li><strong>Schedule cardiology consultation immediately</strong></li>
                            <li>Request comprehensive cardiac evaluation</li>
                            <li>Get ECG, echocardiogram, stress test</li>
                            <li>Blood work: lipid panel, glucose, inflammatory markers</li>
                        </ul>
                        
                        <h5>💊 Potential Medical Interventions (Doctor's discretion)</h5>
                        <ul>
                            <li>Blood pressure medication if hypertensive</li>
                            <li>Statin therapy for high cholesterol</li>
                            <li>Aspirin therapy (if no contraindications)</li>
                            <li>Diabetes management if applicable</li>
                        </ul>
                        
                        <h5>🚨 Emergency Warning Signs - Call 911 Immediately</h5>
                        <ul>
                            <li>Severe chest pain or pressure</li>
                            <li>Pain radiating to arm, jaw, or back</li>
                            <li>Severe shortness of breath</li>
                            <li>Nausea with chest discomfort</li>
                            <li>Cold sweats, dizziness, fainting</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk Factor Modification Guide
            st.markdown("### 📋 Specific Risk Factor Modifications")
            
            modification_col1, modification_col2 = st.columns(2)
            
            with modification_col1:
                st.markdown("""
                **🎯 Primary Prevention Targets:**
                - **Blood Pressure:** <130/80 mm Hg
                - **LDL Cholesterol:** <100 mg/dl (high risk: <70)
                - **HDL Cholesterol:** >40 mg/dl (men), >50 mg/dl (women)
                - **Triglycerides:** <150 mg/dl
                - **Blood Glucose:** <126 mg/dl fasting
                - **BMI:** 18.5-24.9 kg/m²
                """)
            
            with modification_col2:
                st.markdown("""
                **⚠️ Critical Warning Signs to Monitor:**
                - New or worsening chest pain
                - Unexplained shortness of breath
                - Irregular heartbeat or palpitations
                - Excessive fatigue with activity
                - Swelling in legs, ankles, feet
                - Dizziness or fainting spells
                """)
    
    except Exception as e:
        st.error(f"❌ Analysis error occurred: {str(e)}")
        st.info("Please verify your input data and try again. If the problem persists, contact support.")
        with st.expander("Technical Details"):
            st.exception(e)

# --- Raw Data View ---
with st.expander("📋 View Technical Data"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Processed Features:**")
        st.dataframe(input_df, use_container_width=True)
    with col2:
        st.markdown("**Raw Input Values:**")
        st.json(raw_data)

# --- Company Footer ---
st.markdown("""
<div class="company-footer fade-in-up">
    <div class="company-logo">OnePersonAI</div>
    <h4>🏥 Advanced Healthcare AI Solutions</h4>
    <p><strong>CardioAI Predictor v2.0</strong> - Powered by cutting-edge machine learning algorithms</p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0;">
        <div>
            <h5>🔬 Technology</h5>
            <p>Random Forest ML<br>SHAP Explainability<br>Advanced Risk Scoring</p>
        </div>
        <div>
            <h5>🎯 Accuracy</h5>
            <p>Trained on Clinical Data<br>Validated Algorithms<br>Continuous Learning</p>
        </div>
        <div>
            <h5>🛡️ Privacy</h5>
            <p>Data Not Stored<br>Local Processing<br>HIPAA Conscious</p>
        </div>
    </div>
    <hr style="border-color: rgba(255,255,255,0.3); margin: 20px 0;">
    <p><strong>© 2024 OnePersonAI Technologies</strong><br>
    <em>Advancing healthcare through artificial intelligence</em></p>
    <small>For technical support: support@onepersonai.com | Medical emergencies: Call 911</small>
</div>
""", unsafe_allow_html=True)

# --- Enhanced Sidebar Information ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Medical Parameter Guide")

with st.sidebar.expander("❓ Understanding Your Values"):
    st.markdown("""
    **🫀 Key Heart Health Indicators:**
    
    **Blood Pressure Categories:**
    - Normal: <120/80 mm Hg
    - Elevated: 120-129/<80 mm Hg  
    - Stage 1 Hypertension: 130-139/80-89
    - Stage 2 Hypertension: ≥140/90
    
    **Cholesterol Levels:**
    - Optimal: <200 mg/dl total
    - Borderline: 200-239 mg/dl
    - High: ≥240 mg/dl
    
    **Heart Rate Zones:**
    - Resting: 60-100 bpm
    - Target Exercise: 50-85% max
    - Maximum: 220 - age
    """)

with st.sidebar.expander("🚨 Emergency Contacts"):
    st.markdown("""
    **Immediate Medical Help:**
    - **Emergency:** 911
    - **Poison Control:** 1-800-222-1222
    - **Suicide Prevention:** 988
    
    **Non-Emergency Medical:**
    - **Nurse Hotline:** Contact your insurance
    - **Telemedicine:** Your healthcare provider
    
    **Chest Pain Protocol:**
    1. Call 911 immediately
    2. Chew aspirin if no allergies
    3. Sit down and rest
    4. Loosen tight clothing
    """)