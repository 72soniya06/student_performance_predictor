# app.py
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("📚 Student Performance Predictor")
st.write("Predict a student's final score based on study data.")

# Load trained model
model = joblib.load("model.joblib")

# ---------- Option to choose input method ----------
option = st.radio("Select Input Method:", ["Manual Entry", "Upload CSV"])

# ---------- CSV Upload ----------
if option == "Upload CSV":
    st.write("Upload a CSV file with columns: name, roll_no, attendance, cgpa/percentage, assignments, internal_marks, course, branch, section")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Handle percentage → CGPA conversion
        if "percentage" in data.columns and "cgpa" not in data.columns:
            data["cgpa"] = data["percentage"] / 10

        # Ensure only model-required features are passed
        X = data[["attendance", "cgpa", "assignments", "course"]]
        preds = model.predict(X)
        data["Predicted Final Score"] = preds.clip(0, 100)

        st.success("✅ Predictions generated for uploaded CSV")
        st.dataframe(data)

        # Download results
        csv_out = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results as CSV", csv_out, "predictions.csv", "text/csv")

# ---------- Manual Entry ----------
else:
    st.subheader("🎓 Enter Student Details")

    name = st.text_input("Student Name")
    roll_no = st.text_input("Roll No.")

    # Select input type: CGPA or Percentage
    input_type = st.radio("Select input type:", ["CGPA", "Percentage"])
    if input_type == "CGPA":
        score_input = st.number_input("Enter CGPA (0-10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    else:
        score_input = st.number_input("Enter Percentage (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=0.5)

    attendance = st.number_input("Attendance percent", min_value=0.0, max_value=100.0, value=80.0, step=0.5)
    assignments = st.slider("Assignments completed (out of 10)", 0, 10, 8)
    internal_marks = st.number_input("Internal Marks (0-30)", min_value=0, max_value=30, value=20, step=1)

    # Course & Branch
    course = st.selectbox("Course", ["B.Tech", "MBA", "BCA", "MCA"])
    branch = st.selectbox("Branch", ["CSE", "ECE", "IT", "Civil", "Mechanical", "Other"])
    section = st.selectbox("Section", ["A", "B", "C", "D"])

    # Convert percentage → CGPA if needed
    if input_type == "Percentage":
        cgpa = score_input / 10
    else:
        cgpa = score_input

    if st.button("Predict final score"):
        X = pd.DataFrame([{
            "attendance": attendance,
            "cgpa": cgpa,
            "assignments": assignments,
            "course": course
        }])
        pred = model.predict(X)[0]
        pred = max(0, min(100, pred))

        st.subheader("📊 Prediction Result")
        st.metric(label="Predicted final score", value=f"{pred:.1f} / 100")

        st.write("### 📝 Student Details")
        st.table(pd.DataFrame([{
            "Name": name,
            "Roll No.": roll_no,
            "Course": course,
            "Branch": branch,
            "Section": section,
            "Internal Marks": internal_marks,
            "Predicted Final Score": f"{pred:.1f}/100"
        }]))

        if pred >= 85:
            st.success("Excellent — likely top performer 🎉")
        elif pred >= 70:
            st.info("Good — solid performance 👍")
        elif pred >= 50:
            st.warning("Average — needs improvement ⚠️")
        else:
            st.error("Low — significant improvement needed ⛔️")
