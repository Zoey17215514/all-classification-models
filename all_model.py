import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    y_train_deploy = df['NObeyesdad'] # Also need y_train for fitting the preprocessor with the target

    preprocessor_deploy.fit(X_train_deploy)


# Streamlit App Title
st.title("Obesity Level Prediction Report")

if loaded_models is not None and df is not None:
    # Create user input fields based on deployment_features
    st.header("1. User Input Data")
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
                 input_data[col] = st.number_input(f"{col} (years):", value=0.0, help="Enter age in years") # Updated label and help
            else:
                input_data[col] = st.number_input(f"{col}:", value=0.0)

    # Model Selection using Radio Buttons
    st.header("2. Model Selection")
    models_to_choose = ['Decision Tree', 'Random Forest', 'Support Vector Machine']
    selected_model_name = st.radio("Select a Model for Prediction:", models_to_choose)


    # Predict (with submit button)
    if st.button("Generate Prediction Report"):
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data using the fitted preprocessor
        input_data_processed = preprocessor_deploy.transform(input_df[deployment_features])

        st.header("3. Prediction Results")

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

            # --- Add Visualizations ---
            st.header("4. Model Performance and Insights")

            # 1. Accuracy Over Models (Line Chart)
            # Assuming you have a DataFrame or dictionary with model performance metrics
            model_performance_data = {
                'Model': ['Decision Tree', 'Random Forest', 'Support Vector Machine'],
                'Accuracy': [0.9456, 0.9504, 0.9598],
                'Precision': [0.9476, 0.9524, 0.9586],
                'Recall': [0.9440, 0.9485, 0.9586],
                'F1 Score': [0.9450, 0.9495, 0.9583]
            }
            model_performance_df = pd.DataFrame(model_performance_data)

            st.subheader("4.1 Model Performance Comparison")

            # Accuracy Over Models Line Chart
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.lineplot(x='Model', y='Accuracy', data=model_performance_df, marker='o', ax=ax1)
            ax1.set_title('Accuracy Comparison Across Models')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0.85, 1.0) # Set y-axis limits for better visualization
            st.pyplot(fig1)
            plt.close(fig1)


            # Comparison of Metrics (Grouped Bar Chart)
            model_performance_melted = model_performance_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            sns.barplot(x='Model', y='Score', hue='Metric', data=model_performance_melted, ax=ax2)
            ax2.set_title('Comparison of Performance Metrics Across Models')
            ax2.set_ylabel('Score')
            ax2.legend(title='Metric')
            st.pyplot(fig2)
            plt.close(fig2)


            # Feature Importance (Horizontal Bar Chart)
            if hasattr(model, 'feature_importances_'):
                st.subheader(f"4.2 Feature Importance ({selected_model_name})")
                # Get feature names after preprocessing
                try:
                    feature_names = []
                    for name, transformer, cols in preprocessor_deploy.transformers_:
                        if hasattr(transformer, 'get_feature_names_out'):
                             if isinstance(cols, str): # Handle single column case
                                 feature_names.extend(transformer.get_feature_names_out([cols]))
                             else: # Handle multiple columns
                                 feature_names.extend(transformer.get_feature_names_out(cols))
                        elif name == 'num': # For numerical columns, the names are the original column names
                             feature_names.extend(cols)

                    if preprocessor_deploy.remainder == 'passthrough':
                         all_input_cols = list(X_train_deploy.columns)
                         processed_cols = [col.split('__')[1] if '__' in col else col for col in feature_names]
                         remaining_cols = [col for col in all_input_cols if col not in numerical_cols_for_preprocessor and col not in categorical_cols_for_preprocessor]
                         feature_names.extend(remaining_cols)

                    importances = model.feature_importances_
                    if len(importances) == len(feature_names):
                        feat_importances = pd.Series(importances, index=feature_names)
                        feat_importances = feat_importances.sort_values(ascending=False)

                        fig4, ax4 = plt.subplots(figsize=(10, 7))
                        feat_importances.plot(kind='barh', ax=ax4)
                        ax4.set_title(f'Feature Importances ({selected_model_name})')
                        ax4.set_xlabel('Importance')
                        ax4.invert_yaxis()
                        st.pyplot(fig4)
                        plt.close(fig4)
                    else:
                         st.warning("Could not match feature importances to feature names. Number of importances and feature names do not match.")

                except Exception as e:
                    st.error(f"An error occurred while generating Feature Importance chart: {e}")

            elif hasattr(model, 'coef_'):
                 st.subheader(f"4.2 Feature Coefficients ({selected_model_name})")
                 try:
                    feature_names = []
                    for name, transformer, cols in preprocessor_deploy.transformers_:
                        if hasattr(transformer, 'get_feature_names_out'):
                             if isinstance(cols, str):
                                 feature_names.extend(transformer.get_feature_names_out([cols]))
                             else:
                                 feature_names.extend(transformer.get_feature_names_out(cols))
                        elif name == 'num':
                             feature_names.extend(cols)

                    if preprocessor_deploy.remainder == 'passthrough':
                         all_input_cols = list(X_train_deploy.columns)
                         processed_cols = [col.split('__')[1] if '__' in col else col for col in feature_names]
                         remaining_cols = [col for col in all_input_cols if col not in numerical_cols_for_preprocessor and col not in categorical_cols_for_preprocessor]
                         feature_names.extend(remaining_cols)

                    coef_values = np.abs(model.coef_).mean(axis=0)

                    if len(coef_values) == len(feature_names):
                         feat_coef = pd.Series(coef_values, index=feature_names)
                         feat_coef = feat_coef.sort_values(ascending=False)

                         fig_coef, ax_coef = plt.subplots(figsize=(10, 7))
                         feat_coef.plot(kind='barh', ax=ax_coef)
                         ax_coef.set_title(f'Feature Coefficients (Absolute Mean) ({selected_model_name})')
                         ax_coef.set_xlabel('Absolute Mean Coefficient Value')
                         ax_coef.invert_yaxis()
                         st.pyplot(fig_coef)
                         plt.close(fig_coef)
                    else:
                         st.warning("Could not match feature coefficients to feature names. Number of coefficients and feature names do not match.")

                 except Exception as e:
                    st.error(f"An error occurred while generating Feature Coefficients chart: {e}")

            # Prediction Distribution (Pie / Donut Chart)
            # For a single prediction, a distribution chart is not meaningful.
            # This is typically for showing the distribution of predictions on a dataset.
            # If you want to show the distribution of predicted classes on the test set,
            # you would need to load or calculate this based on your test set evaluation.
            # st.subheader("4.4 Predicted Class Distribution (Test Set)")
            # Add code to calculate and display prediction distribution if test data is available.


        else:
            st.warning(f"Model '{selected_model_name}' not found in loaded models.")


else:
    st.warning("Model or data not loaded. Please ensure 'all_classification_models.joblib' and 'ObesityDataSet.csv' are in the correct directory.")
