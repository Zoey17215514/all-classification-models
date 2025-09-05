import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
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

    # Fit the preprocessor with the training data used for the models
    X_train_deploy = df[deployment_features]
    y_train_deploy = df['NObeyesdad'] # Also need y_train for fitting the preprocessor with the target

    preprocessor_deploy.fit(X_train_deploy)

    # Preprocess the entire dataset for model evaluation
    X_processed_full = preprocessor_deploy.transform(df[deployment_features])
    y_full = df['NObeyesdad']

    # Calculate performance metrics for all loaded models on the full dataset
    # In a real-world scenario, you would use a separate test set for unbiased evaluation.
    # For demonstration in this deployment app, we are using the full dataset.
    model_performance_data = []
    if loaded_models:
        for name, model in loaded_models.items():
            try:
                # For SVC, ensure probability=True is set if not already
                if isinstance(model, SVC) and not hasattr(model, 'predict_proba'):
                     model.probability = True
                     # Refit the model to enable probabilities (this might take time)
                     model.fit(preprocessor_deploy.transform(df[deployment_features]), df['NObeyesdad'])


                y_pred_full = model.predict(X_processed_full)
                accuracy = accuracy_score(y_full, y_pred_full)
                precision = precision_score(y_full, y_pred_full, average='macro', zero_division=0)
                recall = recall_score(y_full, y_pred_full, average='macro', zero_division=0)
                f1 = f1_score(y_full, y_pred_full, average='macro', zero_division=0)
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
                 input_data[col] = st.number_input(f"{col} (years):", value=0, help="Enter age in years") # Updated label and help
            else:
                input_data[col] = st.number_input(f"{col}:", value=0.0)

    # Model Selection using Radio Buttons
    st.header("2. Model Selection")
    models_to_choose = list(loaded_models.keys())
    selected_model_name = st.radio("Select a Model for Prediction:", models_to_choose)

    # --- Add Visualizations and Comparison ---
    st.header("3. Model Performance and Insights")

    if not model_performance_df.empty:
        st.subheader("3.1 Model Performance Comparison")

        # Display a table of performance metrics
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

        # Comparison of Metrics (Grouped Bar Chart)
        model_performance_melted_bar = model_performance_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
        fig2, ax2 = plt.subplots(figsize=(10, 6)) # Smaller figure size
        sns.barplot(x='Model', y='Score', hue='Metric', data=model_performance_melted_bar, ax=ax2)
        ax2.set_title('Comparison of Performance Metrics (Bar Plot)')
        ax2.set_ylabel('Score')
        ax2.legend(title='Metric')
        st.pyplot(fig2)
        plt.close(fig2)


    # Feature Importance (Horizontal Bar Chart)
    if hasattr(model, 'feature_importances_'):
        st.subheader(f"3.2 Feature Importance ({selected_model_name})")
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

                fig4, ax4 = plt.subplots(figsize=(8, 6)) # Smaller figure size
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
         st.subheader(f"3.2 Feature Coefficients ({selected_model_name})")
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

                 fig_coef, ax_coef = plt.subplots(figsize=(8, 6)) # Smaller figure size
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
