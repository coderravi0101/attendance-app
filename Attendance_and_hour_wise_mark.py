# =========================================================
# Machine Learning Project using Streamlit
# User uploads dataset and selects columns manually
# =========================================================

# Import libraries
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# =========================================================
# App Title
# =========================================================

st.title("Automatic ML Prediction App")

# =========================================================
# Upload Dataset
# =========================================================

file = st.file_uploader(
    "Upload Your CSV Dataset",
    type=["csv"]
)

# =========================================================
# If file is uploaded
# =========================================================

if file is not None:

    # Reading dataset
    df = pd.read_csv(file)

    # Showing dataset
    st.subheader("Dataset Preview")
    st.write(df)

    # =====================================================
    # Selecting Input Columns
    # =====================================================

    input_columns = st.multiselect(
        "Choose Input Training Columns (X)",
        df.columns
    )

    # =====================================================
    # Selecting Output Column
    # =====================================================

    output_column = st.selectbox(
        "Choose Output Training Column (y)",
        df.columns
    )

    # =====================================================
    # Train Model Button
    # =====================================================

    if st.button("Train Model"):

        # Input data
        X = df[input_columns]

        # Output data
        y = df[output_column]

        # =================================================
        # Creating ML Model
        # =================================================

        model = LinearRegression()

        # =================================================
        # Training Model
        # =================================================

        model.fit(X, y)

        st.success("Model Trained Successfully!")

        # =================================================
        # Prediction Section
        # =================================================

        st.subheader("Enter Values for Prediction")

        user_data = []

        # Taking input from user
        for col in input_columns:

            value = st.number_input(
                f"Enter value for {col}"
            )

            user_data.append(value)

        # =================================================
        # Predict Button
        # =================================================

        if st.button("Predict"):

            # Predicting output
            prediction = model.predict([user_data])

            # Showing prediction
            st.success(
                f"Predicted Value: {prediction[0]}"
            )