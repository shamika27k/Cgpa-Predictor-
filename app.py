import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("cgpa_model.pkl")

# Page configuration
st.set_page_config(page_title="Diploma CGPA Predictor", layout="centered")

# Custom Title Header
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🎓 Diploma CGPA Predictor</h1>
    <p style='text-align: center; color: #666;'>Predict your CGPA on a 10-point scale using semester-wise marks and credits</p>
    <hr style="border: 1px solid #ddd;">
""", unsafe_allow_html=True)

# Number of semesters
num_sems = st.number_input("🔢 How many semesters have you completed?", min_value=1, max_value=12, step=1)

sem_data = []

# Input fields in columns
for i in range(1, num_sems + 1):
    st.markdown(f"### 📚 Semester {i}")
    col1, col2, col3 = st.columns(3)
    with col1:
        obtained = st.number_input(f"✅ Obtained Marks", min_value=0, key=f"ob_{i}")
    with col2:
        total = st.number_input(f"🧮 Total Marks", min_value=1, key=f"tot_{i}")
    with col3:
        credit = st.number_input(f"🎓 Credits", min_value=1, key=f"cred_{i}")

    percentage = round((obtained / total) * 100, 2) if total else 0
    marks_per_credit = round(obtained / credit, 2) if credit else 0

    sem_data.append({
        "Semester": i,
        "Obtained": obtained,
        "Total": total,
        "Credits": credit,
        "Percentage": percentage,
        "Marks/Credit": marks_per_credit
    })

st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

# Predict Button
if st.button("📊 Predict CGPA"):
    df = pd.DataFrame(sem_data)

    invalid = False
    for idx, row in df.iterrows():
        if row["Obtained"] > row["Total"]:
            st.error(f"❌ Semester {int(row['Semester'])}: Obtained marks cannot exceed total marks.")
            invalid = True
        elif row["Percentage"] < 40:
            st.warning(f"⚠️ Semester {int(row['Semester'])} has a low percentage ({row['Percentage']}%). It may affect your CGPA.")

    if not invalid:
        total_obtained = df["Obtained"].sum()
        total_total = df["Total"].sum()
        total_credits = df["Credits"].sum()
        marks_per_credit = round(total_obtained / total_credits, 2) if total_credits else 0
        percentage = round((total_obtained / total_total) * 100, 2) if total_total else 0

        features = np.array([[total_obtained, total_total, total_credits, marks_per_credit, percentage]])
        cgpa = round(model.predict(features)[0], 2)

        # Result
        st.success(f"🎯 **Predicted CGPA:** {cgpa} / 10")

        # Cumulative Summary
        st.markdown("### 📌 Cumulative Summary")
        st.write(f"**Total Marks Obtained:** {total_obtained} / {total_total}")
        st.write(f"**Total Credits Earned:** {total_credits}")
        st.write(f"**Overall Percentage:** {percentage}%")
        st.write(f"**Average Marks per Credit:** {marks_per_credit}")

        # Semester-wise Table
        st.markdown("### 📚 Semester-wise Summary")
        st.dataframe(df, use_container_width=True)

        # GPA per Semester
        df["GPA (10-scale)"] = (df["Percentage"] / 10).round(2)
        st.markdown("### 🧮 Estimated GPA per Semester")
        st.dataframe(df[["Semester", "Percentage", "GPA (10-scale)"]], use_container_width=True)

        # Feedback
        st.markdown("### 💬 Performance Feedback")
        if cgpa >= 9:
            st.success("🌟 Excellent! You're on track for distinction.")
        elif cgpa >= 7:
            st.info("👍 Good! Stay consistent.")
        elif cgpa >= 6:
            st.warning("⚠️ Fair. Focus more to improve.")
        else:
            st.error("❗Needs improvement. Plan better and consider extra help.")
    else:
        st.warning("📛 CGPA prediction skipped due to invalid inputs above. Please fix and retry.")
