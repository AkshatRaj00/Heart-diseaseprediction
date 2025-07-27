# app.py - Fixed Heart Disease Prediction App (English Version)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Predictor", 
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 10px 0;
    }
    .company-credit {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background-color: rgba(0,0,0,0.7);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

# Company Credit
st.markdown("""
<div class="company-credit">
    🏢 Developed by OnePersonAI
</div>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>Advanced Machine Learning-Based Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# --- Info Section ---
st.markdown("""
<div class="info-box">
    <h4>🏥 About This System</h4>
    <p>This application uses machine learning algorithms to assess heart disease risk based on medical parameters.</p>
    <p><strong>⚠️ Medical Disclaimer:</strong> This prediction is for educational and informational purposes only. This system is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.</p>
</div>
""", unsafe_allow_html=True)

# --- Model Loading with Fallback ---
@st.cache_resource
def load_models():
    """Load models or create backup model"""
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, True
    except FileNotFoundError:
        # Create backup model
        np.random.seed(42)
        
        # Sample training data
        n_samples = 500
        X_sample = np.random.randn(n_samples, 13)
        
        # Simple logic for target creation
        y_sample = (X_sample[:, 0] + X_sample[:, 4] + X_sample[:, 7] > 0).astype(int)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X_sample)
        model.fit(X_scaled, y_sample)
        
        return model, scaler, False

# Load models
model, scaler, is_pretrained = load_models()

# Model status display
if is_pretrained:
    st.success("✅ Pre-trained model loaded successfully!")
else:
    st.warning("⚠️ Using demo model (Pre-trained model not found)")

# --- Sidebar for User Input ---
st.sidebar.header("👤 Patient Information")
st.sidebar.markdown("Please fill in your medical details:")

def user_input_features():
    """Collect user input features"""
    
    # Basic Information
    st.sidebar.subheader("📋 Basic Details")
    age = st.sidebar.slider('Age (years)', 29, 77, 48)
    sex = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    
    # Chest Pain Details
    st.sidebar.subheader("💗 Chest Pain Information")
    cp = st.sidebar.selectbox(
        'Chest Pain Type', 
        ('Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic')
    )
    
    # Vital Signs
    st.sidebar.subheader("🩺 Vital Signs")
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 129)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 240)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ('False', 'True'))
    
    # Heart Tests
    st.sidebar.subheader("🫀 Heart Tests")
    restecg = st.sidebar.selectbox(
        'Resting ECG Results', 
        ('Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy')
    )
    thalach = st.sidebar.slider('Maximum Heart Rate (bpm)', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('No', 'Yes'))
    
    # Advanced Parameters
    st.sidebar.subheader("🔬 Advanced Parameters")
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.2, 1.0, step=0.1)
    slope = st.sidebar.selectbox('ST Segment Slope', ('Upsloping', 'Flat', 'Downsloping'))
    ca = st.sidebar.slider('Number of Major Vessels', 0, 3, 0)
    thal = st.sidebar.selectbox('Thalassemia', ('Normal', 'Fixed Defect', 'Reversible Defect'))

    # Mapping dictionaries
    sex_map = {'Male': 1, 'Female': 0}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'False': 0, 'True': 1}
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    exang_map = {'No': 0, 'Yes': 1}
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}

    # Create data dictionary
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

# --- Display User Input Summary ---
st.subheader('📊 Your Input Summary')

# Always show the input summary
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Basic Information:**")
    st.write(f"• **Age:** {raw_data['age']} years")
    st.write(f"• **Gender:** {raw_data['sex']}")
    st.write(f"• **Chest Pain:** {raw_data['cp']}")
    st.write(f"• **Blood Pressure:** {raw_data['trestbps']} mm Hg")
    st.write(f"• **Cholesterol:** {raw_data['chol']} mg/dl")
    st.write(f"• **Fasting Blood Sugar:** {raw_data['fbs']}")

with col2:
    st.markdown("**Heart Parameters:**")
    st.write(f"• **Max Heart Rate:** {raw_data['thalach']} bpm")
    st.write(f"• **Exercise Angina:** {raw_data['exang']}")
    st.write(f"• **ST Depression:** {raw_data['oldpeak']}")
    st.write(f"• **Major Vessels:** {raw_data['ca']}")
    st.write(f"• **Thalassemia:** {raw_data['thal']}")
    st.write(f"• **Resting ECG:** {raw_data['restecg']}")

# --- Risk Assessment ---
def calculate_risk_score(data):
    """Calculate simple risk score"""
    risk_score = 0
    risk_factors = []
    
    if data['age'] > 55:
        risk_score += 1
        risk_factors.append(f"Age > 55 ({data['age']} years)")
    
    if data['sex'] == 'Male':
        risk_score += 1
        risk_factors.append("Male gender (higher risk)")
    
    if data['cp'] == 'Asymptomatic':
        risk_score += 2
        risk_factors.append("Asymptomatic chest pain (concerning)")
    
    if data['trestbps'] > 140:
        risk_score += 1
        risk_factors.append(f"High blood pressure ({data['trestbps']} mm Hg)")
    
    if data['chol'] > 240:
        risk_score += 1
        risk_factors.append(f"High cholesterol ({data['chol']} mg/dl)")
    
    if data['thalach'] < 120:
        risk_score += 1
        risk_factors.append(f"Low max heart rate ({data['thalach']} bpm)")
    
    if data['exang'] == 'Yes':
        risk_score += 1
        risk_factors.append("Exercise induced angina")
    
    if data['oldpeak'] > 2.0:
        risk_score += 1
        risk_factors.append(f"High ST depression ({data['oldpeak']})")
    
    return risk_score, risk_factors

# Calculate and display risk factors
risk_score, risk_factors = calculate_risk_score(raw_data)

st.subheader('⚠️ Risk Factor Analysis')

if risk_factors:
    st.markdown(f"""
    <div class="warning-box">
        <h4>🚨 Identified Risk Factors ({len(risk_factors)}/8):</h4>
        <ul>
    """, unsafe_allow_html=True)
    
    for factor in risk_factors:
        st.markdown(f"<li>{factor}</li>", unsafe_allow_html=True)
    
    st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # Risk level assessment
    if risk_score >= 4:
        st.error("🚨 **HIGH RISK**: Multiple risk factors detected!")
    elif risk_score >= 2:
        st.warning("⚠️ **MODERATE RISK**: Some risk factors present")
    else:
        st.info("ℹ️ **LOW-MODERATE RISK**: Few risk factors detected")
else:
    st.markdown("""
    <div class="success-box">
        <h4>✅ No Major Risk Factors Identified!</h4>
        <p>Your current parameters show minimal traditional risk factors.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Raw Data View ---
with st.expander("📋 View Raw Data"):
    st.dataframe(input_df, use_container_width=True)

# --- Main Prediction Section ---
st.markdown("---")
st.subheader('🔍 ML Model Prediction')

# Prediction button
predict_button = st.button('🎯 Predict Heart Disease Risk', type="primary", use_container_width=True)

if predict_button:
    try:
        with st.spinner('🤖 Machine Learning Model analyzing your data...'):
            # Scale input
            scaled_input = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(scaled_input)
            prediction_proba = model.predict_proba(scaled_input)
            
            # Results containers
            st.markdown("### 🎯 Prediction Results")
            
            # Main result
            if prediction[0] == 0:
                st.markdown("""
                <div class="success-box">
                    <h2>🎉 LOW RISK</h2>
                    <h3>You have a low risk of heart disease</h3>
                    <p>Based on the ML model's prediction, your current condition shows minimal risk for heart disease.</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                result_type = "Low Risk"
            else:
                st.markdown("""
                <div class="danger-box">
                    <h2>🚨 HIGH RISK</h2>
                    <h3>You have a high risk of heart disease</h3>
                    <p>Based on the ML model's prediction, you should consult with a cardiologist immediately.</p>
                </div>
                """, unsafe_allow_html=True)
                result_type = "High Risk"
            
            # Probability breakdown
            st.markdown("### 📊 Detailed Probability Analysis")
            
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            
            no_disease_prob = prediction_proba[0][0] * 100
            disease_prob = prediction_proba[0][1] * 100
            confidence = max(prediction_proba[0]) * 100
            
            with prob_col1:
                st.metric(
                    label="🟢 No Heart Disease",
                    value=f"{no_disease_prob:.1f}%",
                    delta="Safe Zone" if no_disease_prob > 50 else None
                )
            
            with prob_col2:
                st.metric(
                    label="🔴 Heart Disease Risk",
                    value=f"{disease_prob:.1f}%",
                    delta="Danger Zone" if disease_prob > 50 else None
                )
            
            with prob_col3:
                confidence_level = "Very High" if confidence > 80 else "High" if confidence > 60 else "Moderate"
                st.metric(
                    label="🎯 Model Confidence",
                    value=f"{confidence:.1f}%",
                    delta=confidence_level
                )
            
            # Visualization using Plotly (Fixed)
            st.markdown("### 📈 Risk Visualization")
            
            # Create interactive charts with Plotly
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Bar chart using Plotly
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=['No Disease', 'Heart Disease'],
                        y=[no_disease_prob, disease_prob],
                        marker_color=['#28a745', '#dc3545'],
                        text=[f'{no_disease_prob:.1f}%', f'{disease_prob:.1f}%'],
                        textposition='auto',
                    )
                ])
                
                fig_bar.update_layout(
                    title='Heart Disease Risk Prediction',
                    yaxis_title='Probability (%)',
                    showlegend=False,
                    height=400,
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_chart2:
                # Pie chart using Plotly
                fig_pie = go.Figure(data=[
                    go.Pie(
                        labels=['No Disease', 'Heart Disease'],
                        values=[no_disease_prob, disease_prob],
                        marker_colors=['#28a745', '#dc3545'],
                        textinfo='label+percent',
                        textfont_size=12
                    )
                ])
                
                fig_pie.update_layout(
                    title='Risk Distribution',
                    height=400
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Personalized Recommendations")
            
            if prediction[0] == 0:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Positive Results - Maintain Healthy Lifestyle:</h4>
                    <ul>
                        <li><strong>Continue healthy habits:</strong> Regular exercise and balanced diet</li>
                        <li><strong>Regular checkups:</strong> Annual comprehensive health screening</li>
                        <li><strong>Monitor vitals:</strong> Track blood pressure and cholesterol levels</li>
                        <li><strong>Lifestyle:</strong> Avoid smoking and limit alcohol consumption</li>
                        <li><strong>Stay active:</strong> 30 minutes of physical activity daily</li>
                        <li><strong>Stress management:</strong> Practice meditation and yoga</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="danger-box">
                    <h4>🚨 URGENT - Immediate Action Required:</h4>
                    <ul>
                        <li><strong>Doctor consultation:</strong> Schedule appointment with cardiologist immediately</li>
                        <li><strong>Comprehensive tests:</strong> Get ECG, Echocardiogram, and Stress test</li>
                        <li><strong>Lifestyle changes:</strong> Follow strict diet and exercise plan</li>
                        <li><strong>Medication:</strong> Take prescribed medications as directed</li>
                        <li><strong>Emergency plan:</strong> Seek immediate medical help for chest pain</li>
                        <li><strong>Regular monitoring:</strong> Daily check of BP, sugar, and cholesterol</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional risk factors to monitor
            st.markdown("### 📋 Monitor These Risk Factors")
            
            monitor_col1, monitor_col2 = st.columns(2)
            
            with monitor_col1:
                st.markdown("""
                **🔍 Key Parameters to Watch:**
                - Blood Pressure: < 140/90 mm Hg
                - Cholesterol: < 240 mg/dl
                - Blood Sugar: < 126 mg/dl
                - BMI: 18.5-24.9 kg/m²
                """)
            
            with monitor_col2:
                st.markdown("""
                **⚠️ Warning Signs:**
                - Chest pain or pressure
                - Shortness of breath
                - Irregular heartbeat
                - Excessive fatigue
                - Dizziness or fainting
                """)
    
    except Exception as e:
        st.error(f"❌ Prediction error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; margin-top: 30px;">
    <h4>📋 Important Medical Disclaimer</h4>
    <p><strong>This ML prediction tool is for educational and informational purposes only.</strong></p>
    <p>It is not a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p><strong>Always consult with a qualified healthcare provider for any health concerns.</strong></p>
    <br>
    <p><em>🤖 Powered by Machine Learning | 💻 Built with Streamlit | ❤️ Made for Health Awareness</em></p>
    <p><strong>🏢 Developed by OnePersonAI</strong></p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Additional Info ---
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Additional Information")

with st.sidebar.expander("❓ What do these values mean?"):
    st.markdown("""
    **Age**: Risk increases with advancing age
    
    **Gender**: Males generally have higher risk
    
    **Chest Pain Types**:
    - Typical Angina: Classic heart-related pain
    - Atypical Angina: Unusual chest pain
    - Non-Anginal: Not heart-related pain
    - Asymptomatic: No chest pain symptoms
    
    **Blood Pressure**: 
    - Normal: < 120/80
    - High: > 140/90 mm Hg
    
    **Cholesterol**:
    - Normal: < 200 mg/dl
    - High: > 240 mg/dl
    """)

with st.sidebar.expander("🏥 Emergency Contact"):
    st.markdown("""
    **Seek immediate medical attention if you experience:**
    - Severe chest pain
    - Difficulty breathing
    - Irregular heartbeat
    - Excessive sweating
    - Nausea with chest discomfort
    
    **Emergency Numbers:**
    - Emergency: 911 (US) / 102 (India)
    - Ambulance: 108 (India)
    """)

# Model information
if st.sidebar.button("ℹ️ About This Model"):
    st.sidebar.info(f"""
    **Model Type**: {'Pre-trained' if is_pretrained else 'Demo'} Random Forest
    
    **Input Features**: 13 medical parameters
    
    **Algorithm**: Machine Learning Classification
    
    **Purpose**: Risk Assessment Tool
    
    **Developer**: OnePersonAI
    
    **Accuracy**: For educational use only
    """)