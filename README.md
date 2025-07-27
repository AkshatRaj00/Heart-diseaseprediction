❤️ Heart Disease Prediction Web App
Introduction (परिचय)
Yeh ek AI-powered web application hai jo machine learning ka use karke heart disease hone ke risk ka anumaan lagati hai. Is app ka uddeshya logon ko unki medical parameters ke aadhar par unke heart health ke baare mein jaankari dena aur unhe sachet karna hai.

⚠️ Disclaimer: Yeh prediction tool kewal educational aur informational purposes ke liye banaya gaya hai. Yeh kisi bhi tarah se professional medical advice, diagnosis, ya treatment ka substitute nahi hai. Kisi bhi health concern ke liye hamesha qualified healthcare provider se consult karein.

Features (विशेषताएँ)
Interactive User Interface: Streamlit ka upyog karke ek user-friendly aur responsive design.

Real-time Prediction: User inputs ke aadhar par turant heart disease ka risk predict karta hai.

Prediction Probability: Heart disease hone ya na hone ki sambhavna (probability) percentages mein dikhata hai.

Risk Factor Analysis: User ke diye gaye inputs ke aadhar par common heart disease risk factors ko highlight karta hai.

Visualizations: Probabilities ko samajhne mein madad karne ke liye bar aur pie charts ka upyog.

Personalized Recommendations: Prediction result ke aadhar par swasth jeevan shaili aur agle kadamon ke liye sifarish karta hai.

Model Explainability (SHAP): (Agar aapne SHAP integrate kiya hai) Yeh feature batata hai ki aapke diye gaye har input parameter ka prediction par kitna asar pada. Red values risk badhati hain aur Blue values risk kam karti hain.

Responsive Design: Mobile aur desktop dono par acchi tarah se kaam karta hai.

Technical Stack (तकनीकी स्टैक)
Programming Language: Python

Web Framework: Streamlit

Machine Learning: Scikit-learn (RandomForestClassifier, StandardScaler)

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn, (Optional: Plotly for interactive charts)

Model Persistence: Joblib

Explainability: SHAP (Agar integrate kiya hai)

How to Use (कैसे उपयोग करें)
Apni Details Dalein: Left sidebar mein diye gaye sliders aur dropdown menus ka upyog karke apni medical details (jaise age, sex, cholesterol, blood pressure, etc.) bharein.

Predict Button Dabayein: 🎯 Predict Heart Disease Risk button par click karein.

Results Dekhein: App aapko prediction result, probabilities, aur personalized recommendations dikhayegi.

Local Setup (अपने कंप्यूटर पर कैसे चलायें)
Agar aap is project ko apne local machine par run karna chahte hain, toh in steps ko follow karein:

Repository Clone Karein:

git clone https://github.com/AkshatRaj00/Heart-Disease-Prediction-App.git
cd Heart-Disease-Prediction-App

Virtual Environment Banayein (Recommended):

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

Zaroori Libraries Install Karein:

pip install -r requirements.txt
# Agar SHAP use kar rahe hain toh yeh bhi install karein:
# pip install shap streamlit-shap

Machine Learning Model Train Aur Save Karein:
Yeh step heart_disease_model.pkl, scaler.pkl, aur X_train_scaled_sample.pkl files banayega.

python train_and_save_model.py

Confirm karein ki yeh files aapke project folder mein ban gayi hain.

Streamlit App Run Karein:

streamlit run app.py

Yeh command aapke default web browser mein app ko open kar dega.

Deployment (डिप्लॉयमेंट)
Yeh application Streamlit Community Cloud par deploy ki gayi hai, jisse yeh internet par live access ki ja sakti hai.

Live App Link: (Jab aap deploy kar lenge, toh yahan apni live app ka URL daal dena, jaise: https://your-app-name.streamlit.app/)

Contact (संपर्क)
Agar aapke koi sawaal ya sujhav hain, toh aap GitHub issues ke madhyam se ya mere GitHub profile par contact kar sakte hain.
