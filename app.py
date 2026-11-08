import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("Student Performance Prediction")
st.write("Predict student exam scores using Machine Learning.")

study_hours = st.number_input("Study Hours", 0, 24, 6)
attendance = st.number_input("Attendance (%)", 0, 100, 85)
previous_score = st.number_input("Previous Score", 0, 100, 70)

# Demonstration model
np.random.seed(42)
X = np.column_stack([
    np.random.randint(1, 10, 100),
    np.random.randint(50, 101, 100),
    np.random.randint(35, 96, 100)
])

y = 3*X[:,0] + 0.3*X[:,1] + 0.4*X[:,2] + np.random.normal(0,5,100)
y = np.clip(y, 0, 100)

model = LinearRegression()
model.fit(X, y)

if st.button("Predict Score"):
    score = model.predict([[study_hours, attendance, previous_score]])[0]
    st.success(f"Predicted Exam Score: {score:.2f}")
  
