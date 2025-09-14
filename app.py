# app.py
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("📚 Student Performance Predictor")
st.write("Predict a student's final score based on study data.")

# Load trained model
model = joblib.load("model.joblib")

# Select input type: CGPA or Percentage
input_type = st.radio("Select input type:", ["CGPA", "Percentage"])

if input_type == "CGPA":
    score_input = st.number_input("Enter CGPA (0-10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
else:
    score_input = st.number_input("Enter Percentage (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=0.5)

# Other input fields
hours = st.number_input("Hours studied per week", min_value=0.0, max_value=168.0, value=8.0, step=0.5)
attendance = st.number_input("Attendance percent", min_value=0.0, max_value=100.0, value=80.0, step=0.5)
assignments = st.slider("Assignments completed (out of 10)", 0, 10, 8)

# Select class or course
level = st.radio("Level:", ["School (Class 1-12)", "Higher Education (Course)"])

if level == "School (Class 1-12)":
    class_num = st.selectbox("Select Class", list(range(1, 13)))
    course = "School"  # default value for pipeline
else:
    class_num = None
    course = st.selectbox("Select Course", ["B.Tech", "MBA", "BCA", "MCA"])

# Prepare input for model
# Convert percentage to CGPA if needed (assuming 10-point scale)
if input_type == "Percentage":
    cgpa = score_input / 10
else:
    cgpa = score_input

if st.button("Predict final score"):
    X = pd.DataFrame([{
        "hours": hours,
        "attendance": attendance,
        "cgpa": cgpa,
        "assignments": assignments,
        "course": course
    }])
    pred = model.predict(X)[0]
    pred = max(0, min(100, pred))  # clip to 0-100

    st.metric(label="Predicted final score", value=f"{pred:.1f} / 100")

    if pred >= 85:
        st.success("Excellent — likely top performer 🎉")
    elif pred >= 70:
        st.info("Good — solid performance 👍")
    elif pred >= 50:
        st.warning("Average — needs improvement ⚠️")
    else:
        st.error("Low — significant improvement needed ⛔️")
