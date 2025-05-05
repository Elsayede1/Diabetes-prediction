import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from sklearn.ensemble import RandomForestClassifier

# ------- Page Setup --------
st.set_page_config(page_title="🩺 Diabetes Classifier", layout="wide")

# ------ Custom CSS with Background Colors ------
def set_custom_style():
    st.markdown(f"""
        <style>
            .stApp {{
                background: linear-gradient(to bottom right, #f0f8ff, #ffffff); /* Soft gradient background */
                background-attachment: fixed;
            }}

            section[data-testid="stSidebar"] > div:first-child {{
                background: linear-gradient(to bottom, #b3cde0, #e6f0f9); /* Light blue sidebar background */
                color: black;
            }}

            .main {{
                background-color: rgba(255, 255, 255, 0.97); /* Light background for the main area */
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            }}

            h1, h2, h3 {{
                color: #2c3e50; /* Darker color for headings */
            }}

            .stButton > button {{
                background-color: #3498db; /* Light Blue color for the buttons */
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }}

            .result-box {{
                border-radius: 10px;
                padding: 20px;
                color: white;
                font-size: 18px;
                margin-top: 20px;
            }}
        </style>
    """, unsafe_allow_html=True)

set_custom_style()

# ------- Load trained model -------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # Ensure the model file is available

model = load_model()

# ------- Title & Description -------
st.title("🩺 Smart Diabetes Predictor")
st.markdown("#### Predict whether you're at risk of **Prediabetes or Diabetes** using clinical and lifestyle indicators.")

# ------- Sidebar Inputs -------
st.sidebar.header("📋 Enter Your Personal Health Information")
inputs = {
    'HighBP': st.sidebar.selectbox("Do you have High Blood Pressure?", [0, 1]),
    'HighChol': st.sidebar.selectbox("Do you have High Cholesterol?", [0, 1]),
    'CholCheck': st.sidebar.selectbox("Had a Cholesterol Check recently (5 yrs)?", [0, 1]),
    'BMI': st.sidebar.slider("Body Mass Index", 10, 50, 25),
    'Smoker': st.sidebar.selectbox("Are you a smoker?", [0, 1]),
    'Stroke': st.sidebar.selectbox("Have you had a stroke before?", [0, 1]),
    'HeartDiseaseorAttack': st.sidebar.selectbox("History of Heart Disease or Heart Attack?", [0, 1]),
    'PhysActivity': st.sidebar.selectbox("Do you exercise regularly?", [0, 1]),
    'Fruits': st.sidebar.selectbox("Do you eat fruits daily?", [0, 1]),
    'Veggies': st.sidebar.selectbox("Do you eat vegetables daily?", [0, 1]),
    'HvyAlcoholConsump': st.sidebar.selectbox("Heavy Alcohol Consumption?", [0, 1]),
    'GenHlth': st.sidebar.slider("General Health (1 = Excellent, 5 = Poor)", 1, 5, 3),
    'MentHlth': st.sidebar.slider("Days Mental Health Not Good (past 30 days)", 0, 30, 5),
    'PhysHlth': st.sidebar.slider("Days Physical Health Not Good (past 30 days)", 0, 30, 5),
    'DiffWalk': st.sidebar.selectbox("Difficulty Walking?", [0, 1]),
    'Sex': st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1]),
    'Age': st.sidebar.slider("Age Group (1 = 18–24 to 14 = 80+)", 1, 14, 6),
    'Education': st.sidebar.slider("Education Level (1 = Least, 6 = College Grad)", 1, 6, 4),
    'Income': st.sidebar.slider("Income Group (1 = <10K, 8 = >75K)", 1, 8, 4)
}
user_df = pd.DataFrame([inputs])

# ------- Prediction -------
st.markdown("### 📊 Result")
if st.button("🔍 Run Prediction"):
    prob = model.predict_proba(user_df)[0][1]
    percentage = round(prob * 100, 2)

    if percentage < 20:
        st.markdown(f"<div class='result-box' style='background-color:#1abc9c'>"
                    f"✅ **Low Risk: {percentage}%**<br>Great! You’re currently at low risk for diabetes."
                    "</div>", unsafe_allow_html=True)
        st.info("""
        ### 🧠 Health Advice
        - Maintain a balanced diet (fruits, vegetables, whole grains)
        - Keep exercising regularly
        - Continue regular checkups every 1–2 years
        - Stay mentally active and manage stress
        """)
        st.success("💬 **“Health is a state of body. Wellness is a state of being.”**")
    elif 20 <= percentage < 50:
        st.markdown(f"<div class='result-box' style='background-color:#f39c12'>"
                    f"⚠️ **Moderate Risk: {percentage}%**<br>Consider making healthier lifestyle changes."
                    "</div>", unsafe_allow_html=True)
        st.warning("""
        ### 🧠 Health Advice
        - Improve dietary choices (reduce sugar, eat more fiber)
        - Increase physical activity (at least 150 mins/week)
        - Monitor weight and sleep patterns
        - Schedule a medical screening within 6 months
        """)
        st.info("💬 **“A healthy outside starts from the inside.”**")
    else:
        st.markdown(f"<div class='result-box' style='background-color:#e74c3c'>"
                    f"🚨 **High Risk: {percentage}%**<br>Please consult a healthcare provider soon."
                    "</div>", unsafe_allow_html=True)
        st.error("""
        ### 🧠 Health Advice
        - Immediate lifestyle changes are critical
        - Book a medical appointment (blood sugar testing, A1C)
        - Cut processed sugar and high-carb foods immediately
        - Prioritize daily walks, hydration, and portion control
        """)
        st.warning("💬 **“Act now. Prevention today is protection tomorrow.”**")
        
    # Coordinator Cell (Expandable)
    with st.expander("👨‍⚕️ Coordinator Cell - Virtual Health Assistant"):
        st.markdown("""
        **🧬 Personalized Next Steps**  
        - Track your BMI & waist-to-hip ratio monthly  
        - Use a journal to log meals and physical activity  
        - Ask your physician about HbA1c test  
        - Look into diabetes prevention programs or group coaching  

        **📞 Need Help?**  
        - National Diabetes Hotline (US): 1-800-DIABETES  
        - Visit [diabetes.org](https://www.diabetes.org) for resources
        """)
# ------- Footer -------
st.caption("🛡️ Early detection is the first step to prevention 🛡️")

st.markdown("""
---
**👨‍🔬 Developed by ESE Team**  
**📋 Team Members:**  
- Elsayed  
- Saif  
- Emad
""")
