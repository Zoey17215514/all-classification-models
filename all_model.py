import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Removed arff import: from scipy.io import arff


# Load the model
try:
    loaded_models = load('all_classification_models.joblib')
    st.success("All classification models loaded successfully.")
except FileNotFoundError:
    st.error("Error: 'all_classification_models.joblib' not found. Please ensure the models are saved.")
    loaded_models = None # Set to None to prevent errors if the file is not found


# Define the features to be used for prediction (ensure this matches what the models were trained on)
deployment_features = ['Gender', 'Weight', 'Height', 'FCVC', 'Age']

# Load the original data to fit the preprocessor correctly from CSV
try:
    csv_file_path = 'ObesityDataSet.csv' # Changed to CSV file path
    df = pd.read_csv(csv_file_path) # Changed to read from CSV

    # If the CSV contains byte strings, you might still need this decoding step
    # df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

except FileNotFoundError:
    st.error("Error: 'ObesityDataSet.csv' not found. Please ensure the data file is available.")
    df = None


# Create preprocessing pipelines for numerical and categorical features
# This needs to be fitted on the training data
if df is not None:
    categorical_cols_for_preprocessor = [col for col in deployment_features if col in df.columns and df[col].dtype == 'object']
    numerical_cols_for_preprocessor = [col for col in deployment_features if col in df.columns and df[col].dtype != 'object']

    numerical_transformer_deploy = StandardScaler()
    categorical_transformer_deploy = OneHotEncoder(handle_unknown='ignore', drop='first')

    preprocessor_deploy = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer_deploy, numerical_cols_for_preprocessor),
            ('cat', categorical_transformer_deploy, categorical_cols_for_preprocessor)
        ],
        remainder='passthrough'
    )

    # Fit the preprocessor with the training data used for the models
    X_train_deploy = df[deployment_features]
    preprocessor_deploy.fit(X_train_deploy)


# Streamlit App Title
st.title("Obesity Level Prediction")

if loaded_models is not None and df is not None:
    # Model Selection Dropdown
    model_choices = list(loaded_models.keys())
    selected_model_name = st.selectbox("Choose a Model:", model_choices)

    # Get the selected model
    selected_model = loaded_models[selected_model_name]

    st.write(f"Using the {selected_model_name} model for prediction.")

    # Create user input fields based on deployment_features
    st.header("Enter Your Data:")
    input_data = {}
    # Define mapping for FCVC text labels to numerical values
    fcvc_mapping = {"Never": 1.0, "Sometimes": 2.0, "Always": 3.0}
    fcvc_options = list(fcvc_mapping.keys())

    for col in deployment_features:
        if col in categorical_cols_for_preprocessor:
            options = list(df[col].unique())
            input_data[col] = st.selectbox(f"{col}:", options)
        elif col == 'FCVC': # Handle FCVC separately with selectbox
             selected_fcvc_text = st.selectbox("Frequency of consumption of vegetables:", fcvc_options)
             input_data[col] = fcvc_mapping[selected_fcvc_text] # Map text to numerical value
        elif col in numerical_cols_for_preprocessor:
            # Add units to the description
            if col == 'Weight':
                 input_data[col] = st.number_input(f"{col} (kg):", value=0.0, help="Enter weight in kilograms") # Updated label and help
            elif col == 'Height':
                 input_data[col] = st.number_input(f"{col} (m):", value=0.0, help="Enter height in meters") # Updated label and help
            elif col == 'Age':
                 input_data[col] = st.number_input(f"{col} (years):", value=0, help="Enter age in years") # Updated label and help
            else:
                input_data[col] = st.number_input(f"{col}:", value=0.0)

    # Predict (with submit button)
    if st.button("Predict Obesity Level"):
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data using the fitted preprocessor
        input_data_processed = preprocessor_deploy.transform(input_df[deployment_features])


        # Make prediction
        prediction = selected_model.predict(input_data_processed)

        st.header("Prediction Result:")
        st.write(f"Predicted Obesity Level: **{prediction[0]}**")

        # Add a simple interpretation based on the prediction
        if 'Obesity' in prediction[0]:
            st.write("This suggests a higher risk of health issues associated with obesity.")
        elif 'Overweight' in prediction[0]:
            st.write("This suggests you are at risk of developing obesity.")
        elif 'Normal_Weight' in prediction[0]:
            st.write("This suggests you are currently maintaining a healthy weight.")
        elif 'Insufficient_Weight' in prediction[0]:
            st.write("This suggests you are underweight, which can also lead to health concerns.")

else:
    st.warning("Model or data not loaded. Please ensure 'all_classification_models.joblib' and 'ObesityDataSet.csv' are in the correct directory.")
