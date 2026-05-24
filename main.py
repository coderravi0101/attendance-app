# =========================================================
# Simple Machine Learning Project using Streamlit
# Project: Student Marks Prediction
# =========================================================

# Importing required libraries
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# =========================================================
# App Title
# =========================================================

st.title("Student Marks Prediction App")

# =========================================================
# Sample Dataset
# Hours = Study Hours
# Marks = Exam Marks
# =========================================================

data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [20, 30, 40, 50, 60, 70, 80, 90]
}

# Creating DataFrame
df = pd.DataFrame(data)

# =========================================================
# Showing Dataset
# =========================================================

st.subheader("Training Dataset")
st.write(df)

# =========================================================
# Input and Output Variables
# X = Independent Variable
# y = Dependent Variable
# =========================================================

X = df[["Hours"]]
y = df["Marks"]

# =========================================================
# Creating Machine Learning Model
# Linear Regression Algorithm
# =========================================================

model = LinearRegression()

# =========================================================
# Training the Model
# Model learns relationship between Hours and Marks
# =========================================================

model.fit(X, y)

# =========================================================
# User Input
# Taking study hours from user
# =========================================================

hours = st.number_input(
    "Enter Study Hours",
    min_value=1,
    max_value=12,
    step=1
)

# =========================================================
# Prediction Button
# =========================================================

if st.button("Predict Marks"):

    # Predicting marks using trained model
    prediction = model.predict([[hours]])

    # Showing Prediction Result
    st.success(f"Predicted Marks: {prediction[0]:.2f}")

# =========================================================
# Extra Information
# =========================================================

st.info("This app predicts student marks based on study hours.")

# =========================================================
# End of Project
# =========================================================