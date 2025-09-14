# app.py
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("📚 Student Performance Predictor")
st.write("Predict a student's final score based on study data.")

# Load trained model
model = joblib.load("model.joblib")

# Choose mode: single student or bulk upload
mode = st.radio("Select mode:", ["Single Student", "Upload CSV"])

# ------------------ Single Student Input ------------------
if mode == "Single Student":
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
        course = "School"  # placeholder for model
    else:
        class_num = None
        course = st.selectbox("Select Course", ["B.Tech", "MBA", "BCA", "MCA"])

    # Convert percentage to CGPA if needed
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


# ------------------ Bulk CSV Upload ------------------
else:
    st.write("Upload a CSV file with student details.")
    st.info("CSV columns required: hours, attendance, cgpa (or percentage), assignments, course")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # If percentage is given instead of cgpa, convert automatically
        if "percentage" in df.columns and "cgpa" not in df.columns:
            df["cgpa"] = df["percentage"] / 10

        # Keep only needed columns
        required_cols = ["hours", "attendance", "cgpa", "assignments", "course"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            st.error(f"Missing columns in CSV: {missing}")
        else:
            preds = model.predict(df[required_cols])
            df["Predicted Final Score"] = preds.clip(0, 100)

            st.success("✅ Predictions completed")
            st.dataframe(df)

            # Download option
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", csv_out, "predictions.csv", "text/csv")
