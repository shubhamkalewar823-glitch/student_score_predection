import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Model Deployment App",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

st.title("Student Performance & Outcome Predictor")
st.write("Provide the input features below to generate a prediction using your KNeighborsRegressor model.")

if model is None:
    st.error("Model file 'model.pkl' not found! Please upload or save your trained model as 'model.pkl' in the same directory.")
else:
    with st.form("prediction_form"):
        st.subheader("Input Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=20)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            course = st.selectbox("Course", ["Computer Science", "Engineering", "Business", "Arts", "Science", "Other"])
            study_hours = st.slider("Study Hours (per week)", min_value=0.0, max_value=60.0, value=10.0)
            class_attendance = st.slider("Class Attendance (%)", min_value=0.0, max_value=100.0, value=85.0)
            internet_access = st.selectbox("Internet Access", ["Yes", "No"])
            
        with col2:
            sleep_hours = st.slider("Sleep Hours (per day)", min_value=0.0, max_value=16.0, value=7.0)
            sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good", "Excellent"])
            study_method = st.selectbox("Study Method", ["Self-study", "Group study", "Online courses", "Hybrid"])
            facility_rating = st.slider("Facility Rating (1-5)", min_value=1, max_value=5, value=3)
            exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Medium", "Hard"])
            
        submit_button = st.form_submit_button(label="Generate Prediction")

    if submit_button:
        # Create a DataFrame matching the training feature names
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'course': [course],
            'study_hours': [study_hours],
            'class_attendance': [class_attendance],
            'internet_access': [internet_access],
            'sleep_hours': [sleep_hours],
            'sleep_quality': [sleep_quality],
            'study_method': [study_method],
            'facility_rating': [facility_rating],
            'exam_difficulty': [exam_difficulty]
        })
        
        try:
            prediction = model.predict(input_data)
            st.success(f"### Predicted Result: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Tip: If your model is a raw KNeighborsRegressor without a built-in preprocessor/pipeline, ensure that categorical values are encoded identically to how they were during training.")
