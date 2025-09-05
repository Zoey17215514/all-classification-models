import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC # Import SVC

# Load the model
try:
    loaded_models = load('all_classification_models.joblib')
    st.success("All classification models loaded successfully.")
except FileNotFoundError:
    st.error("Error: 'all_classification_models.joblib' not found. Please ensure the models are saved.")
    loaded_models = None # Set to None to prevent errors if the file is not found


# Define the features to be used for prediction (ensure this matches what the models were trained on)
deployment_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP']

# Load the original data to fit the preprocessor correctly from CSV
try:
    csv_file_path = 'ObesityDataSet.csv' # Changed to CSV file path
    df = pd.read_csv(csv_file_path) # Changed to read from CSV

except FileNotFoundError:
    st.error("Error: 'ObesityDataSet.csv' not found. Please ensure the data file is available.")
    df = None

# Create preprocessing pipelines for numerical and categorical features
# This needs to be fitted on the training data
if df is not None:
    # Split data into training and testing sets for evaluation
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        df[deployment_features], df['NObeyesdad'], test_size=0.2, random_state=42, stratify=df['NObeyesdad']
    )

    # Identify categorical and numerical columns based on the deployment features
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

    # Fit the preprocessor with the training data (evaluation split) - Moved fitting outside the button click
    preprocessor_deploy.fit(X_train_eval)

    # Transform the test set for model evaluation
    X_test_processed_eval = preprocessor_deploy.transform(X_test_eval)
    y_test_eval = y_test_eval # Keep the original test labels

    # Calculate performance metrics for all loaded models on the test set
    model_performance_data = []
    if loaded_models:
        for name, model in loaded_models.items():
            try:
                # For SVC, ensure probability=True is set if not already
                if isinstance(model, SVC) and not hasattr(model, 'predict_proba'):
                     model.probability = True
                     # Refit the model to enable probabilities (this might take time)
                     # Fit on the transformed training data used for evaluation
                     model.fit(preprocessor_deploy.transform(X_train_eval), y_train_eval)


                y_pred_eval = model.predict(X_test_processed_eval)
                accuracy = accuracy_score(y_test_eval, y_pred_eval)
                precision = precision_score(y_test_eval, y_pred_eval, average='macro', zero_division=0)
                recall = recall_score(y_test_eval, y_pred_eval, average='macro', zero_division=0)
                f1 = f1_score(y_test_eval, y_pred_eval, average='macro', zero_division=0)
                model_performance_data.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                })
            except Exception as e:
                st.warning(f"Could not calculate performance metrics for {name}: {e}")
    model_performance_df = pd.DataFrame(model_performance_data)


# Streamlit App Title
st.title("Obesity Level Prediction Report")

if loaded_models is not None and df is not None:

    # Model Performance Comparison (Table and Line Chart)
    st.header("1. Model Performance Comparison")

    if not model_performance_df.empty:
        st.subheader("1.1 Model Performance Table (on Test Set)")
        st.dataframe(model_performance_df.set_index('Model').style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

        # Accuracy Over Models Line Chart
        model_performance_melted_line = model_performance_df.melt(id_vars='Model', var_name='Metric', value_name='Score', value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
        fig1, ax1 = plt.subplots(figsize=(8, 5)) # Smaller figure size
        sns.lineplot(x='Model', y='Score', hue='Metric', data=model_performance_melted_line, marker='o', ax=ax1)
        ax1.set_title('Model Performance Comparison (Line Plot)')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0.8, 1.0) # Adjust y-axis limits as needed
        ax1.legend(title='Metric')
        st.pyplot(fig1)
        plt.close(fig1)


    # Model Selection using Radio Buttons
    st.header("2. Model Selection")
    models_to_choose = list(loaded_models.keys())
    selected_model_name = st.radio("Select a Model for Prediction:", models_to_choose)


    st.header("3. User Input Data") # Changed section header
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
        elif col == 'NCP': # Handle NCP with number input and range
             # Changed to radio button input
             input_data[col] = st.radio("Number of main meals per day:", options=[1.0, 2.0, 3.0, 4.0])
        elif col in numerical_cols_for_preprocessor:
            # Add units to the description
            if col == 'Weight':
                 input_data[col] = st.number_input(f"{col} (kg):", value=0.0, min_value=0.0, help="Enter weight in kilograms") # Updated label and help
            elif col == 'Height':
                 input_data[col] = st.number_input(f"{col} (m):", value=0.0, min_value=0.0, help="Enter height in meters") # Updated label and help
            elif col == 'Age':
                 input_data[col] = st.number_input(f"{col} (years):", value=0, min_value=0, help="Enter age in years") # Updated label and help
            else:
                input_data[col] = st.number_input(f"{col}:", value=0.0)

    # Predict (with submit button)
    if st.button("Generate Prediction Report"):
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data using the fitted preprocessor
        input_data_processed = preprocessor_deploy.transform(input_df[deployment_features])

        st.header("4. Prediction Results")

        # Get the selected model
        if selected_model_name in loaded_models:
            model = loaded_models[selected_model_name]
            # Make prediction
            prediction = model.predict(input_data_processed)

            st.subheader(f"Prediction using {selected_model_name}:")
            st.write(f"Predicted Obesity Level: **{prediction[0]}**")

            # Add a simple interpretation based on the prediction
            st.subheader("Interpretation:")
            if 'Obesity' in prediction[0]:
                st.write("Based on the provided data and the selected model, the predicted obesity level falls into an 'Obesity' category. This indicates a higher risk of health issues associated with obesity.")
            elif 'Overweight' in prediction[0]:
                st.write("Based on the provided data and the selected model, the predicted obesity level falls into an 'Overweight' category. This suggests you are at risk of developing obesity.")
            elif 'Normal_Weight' in prediction[0]:
                st.write("Based on the provided data and the selected model, the predicted obesity level falls into the 'Normal Weight' category. This suggests you are currently maintaining a healthy weight.")
            elif 'Insufficient_Weight' in prediction[0]:
                st.write("Based on the provided data and the selected model, the predicted obesity level falls into the 'Insufficient Weight' category. This suggests you are underweight, which can also lead to health concerns.")


            # Add a pie chart for risk distribution (using predict_proba if available)
            if hasattr(model, 'predict_proba'):
                st.subheader("Risk Distribution by Obesity Level:")
                # Get the probability distribution for the prediction
                probabilities = model.predict_proba(input_data_processed)[0]

                # Get the class labels
                class_labels = model.classes_

                # Create a pandas Series for easy plotting
                risk_distribution = pd.Series(probabilities, index=class_labels)

                # # Filter for the desired classes - Removed this line to include all classes
                # target_classes = ['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
                # risk_distribution_filtered = risk_distribution[risk_distribution.index.isin(target_classes)]

                # Plot the pie chart
                fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                risk_distribution.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax_pie)
                ax_pie.set_title('Risk Distribution for Obesity Levels')
                ax_pie.set_ylabel('') # Remove the default y-label
                st.pyplot(fig_pie)
                plt.close(fig_pie)
            else:
                st.info("The selected model does not support probability prediction (predict_proba) for the pie chart.")


else:
    st.warning("Model or data not loaded. Please ensure 'all_classification_models.joblib' and 'ObesityDataSet.csv' are in the correct directory.")
