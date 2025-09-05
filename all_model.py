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
from sklearn.inspection import permutation_importance # Import permutation_importance
import json # Import json to load metrics

# Load the model
try:
    loaded_models = load('all_classification_models.joblib')
    st.success("All classification models loaded successfully.")
except FileNotFoundError:
    st.error("Error: 'all_classification_models.joblib' not found. Please ensure the models are saved.")
    loaded_models = None # Set to None to prevent errors if the file is not found

# Load K-Fold Cross-Validation Results
kfold_results = {}
try:
    # Load results for Decision Tree
    with open('dt_kfold_results.json', 'r') as f:
        kfold_results['Decision Tree'] = json.load(f)
    # Add loading for other models' results here as they are saved
    # Example:
    # with open('rf_kfold_results.json', 'r') as f:
    #     kfold_results['Random Forest'] = json.load(f)
    # with open('svm_kfold_results.json', 'r') as f:
    #     kfold_results['Support Vector Machine'] = json.load(f)

    st.success("K-Fold results loaded successfully.")
except FileNotFoundError:
    st.warning("K-Fold results file(s) not found. Performance table will only show results from the single test split.")
except json.JSONDecodeError:
     st.error("Error decoding K-Fold results JSON file.")
except Exception as e:
     st.error(f"An unexpected error occurred while loading K-Fold results: {e}")


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
    # With the current deployment_features, there are no categorical columns.
    categorical_cols_for_preprocessor = [col for col in deployment_features if col in df.columns and df[col].dtype == 'object']
    numerical_cols_for_preprocessor = [col for col in deployment_features if col in df.columns and df[col].dtype != 'object']

    numerical_transformer_deploy = StandardScaler()
    # categorical_transformer_deploy = OneHotEncoder(handle_unknown='ignore', drop='first') # No categorical features for these models

    # Simplify the preprocessor since only numerical features are used
    preprocessor_deploy = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer_deploy, numerical_cols_for_preprocessor)
            # Removed 'cat' transformer as there are no categorical features in deployment_features
        ],
        remainder='passthrough'
    )

    # Fit the preprocessor with the training data (evaluation split) - Moved fitting outside the button click
    preprocessor_deploy.fit(X_train_eval)

    # Transform the test set for model evaluation
    X_test_processed_eval = preprocessor_deploy.transform(X_test_eval)
    y_test_eval = y_test_eval # Keep the original test labels

    # Calculate performance metrics for all loaded models on the test set
    # Also prepare data for the performance table, including K-Fold results if loaded
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

                # Calculate metrics on the single test split (for demonstration in the app)
                y_pred_eval = model.predict(X_test_processed_eval)
                accuracy_single = accuracy_score(y_test_eval, y_pred_eval)
                precision_single = precision_score(y_test_eval, y_pred_eval, average='macro', zero_division=0)
                recall_single = recall_score(y_test_eval, y_pred_eval, average='macro', zero_division=0)
                f1_single = f1_score(y_test_eval, y_pred_eval, average='macro', zero_division=0)

                # Get K-Fold results if loaded
                if name in kfold_results:
                     kfold_res = kfold_results[name]
                     model_performance_data.append({
                        'Model': name,
                        'Accuracy (Test)': accuracy_single,
                        'Precision (Test)': precision_single,
                        'Recall (Test)': recall_single,
                        'F1 Score (Test)': f1_single,
                        'Accuracy (K-Fold Avg)': kfold_res.get('Avg Accuracy'),
                        'Accuracy (K-Fold Std)': kfold_res.get('Std Accuracy'),
                        'Precision (K-Fold Avg)': kfold_res.get('Avg Precision'),
                        'Precision (K-Fold Std)': kfold_res.get('Std Precision'),
                        'Recall (K-Fold Avg)': kfold_res.get('Avg Recall'),
                        'Recall (K-Fold Std)': kfold_res.get('Std Recall'),
                        'F1 Score (K-Fold Avg)': kfold_res.get('Avg F1 Score'),
                        'F1 Score (K-Fold Std)': kfold_res.get('Std F1 Score')
                     })
                else:
                    # If K-Fold results not loaded, just show single test split results
                     model_performance_data.append({
                        'Model': name,
                        'Accuracy (Test)': accuracy_single,
                        'Precision (Test)': precision_single,
                        'Recall (Test)': recall_single,
                        'F1 Score (Test)': f1_single
                     })

            except Exception as e:
                st.warning(f"Could not calculate performance metrics for {name}: {e}")
    model_performance_df = pd.DataFrame(model_performance_data)

    # Format standard deviation columns
    for col in model_performance_df.columns:
        if 'Std' in col:
            model_performance_df[col] = model_performance_df[col].apply(lambda x: f"&plusmn; {x:.4f}" if pd.notna(x) else "")
            # Combine Avg and Std columns for display
            avg_col = col.replace('Std', 'Avg')
            original_metric = col.replace(' (K-Fold Std)', '')
            if avg_col in model_performance_df.columns:
                 model_performance_df[original_metric + ' (K-Fold)'] = model_performance_df[avg_col].round(4).astype(str) + model_performance_df[col]
                 # Drop the separate Avg and Std columns after combining
                 model_performance_df = model_performance_df.drop(columns=[avg_col, col])


# Streamlit App Title
st.title("Obesity Level Prediction Report")

if loaded_models is not None and df is not None:

    # Model Performance Comparison (Table and Line Chart)
    st.header("1. Model Performance Comparison")

    if not model_performance_df.empty:
        st.subheader("1.1 Model Performance Table") # Removed "(on Test Set)" as it now includes K-Fold
        # Reorder columns for better display: Model, Test metrics, K-Fold metrics
        col_order = ['Model']
        test_cols = [col for col in model_performance_df.columns if '(Test)' in col]
        kfold_cols = [col for col in model_performance_df.columns if '(K-Fold)' in col]

        col_order.extend(test_cols)
        # Sort K-Fold columns to group metrics
        kfold_cols_sorted = []
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
             kfold_cols_sorted.extend([col for col in kfold_cols if col.startswith(metric)])

        col_order.extend(kfold_cols_sorted)

        # Ensure all columns are included, in case there are others
        final_cols = col_order + [col for col in model_performance_df.columns if col not in col_order]

        # Select and display columns, avoiding highlighting formatted strings
        cols_to_highlight = [col for col in model_performance_df.columns if '(Test)' in col or ('(K-Fold)' in col and '&plusmn;' not in str(model_performance_df[col].iloc[0]))]


        st.dataframe(
            model_performance_df[final_cols].style.highlight_max(
                subset=cols_to_highlight, # Highlight only the numerical columns
                axis=0,
                color='lightgreen'
            ),
            use_container_width=True
        )


        # Accuracy Over Models Line Chart
        # The line chart is best suited for single-value comparisons, so we'll use the Test or Avg K-Fold if available
        chart_data = model_performance_df.copy()
        # Use Avg K-Fold if available, otherwise use Test
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
             avg_kfold_col = metric + ' (K-Fold)'
             test_col = metric + ' (Test)'
             if avg_kfold_col in chart_data.columns:
                 # Need to extract numerical value from formatted string for chart
                 chart_data[metric] = chart_data[avg_kfold_col].apply(lambda x: float(x.split('&plusmn;')[0].strip()) if isinstance(x, str) and '&plusmn;' in x else x)
                 chart_data = chart_data.drop(columns=[avg_kfold_col])
             elif test_col in chart_data.columns:
                 chart_data[metric] = chart_data[test_col]
                 chart_data = chart_data.drop(columns=[test_col])


        model_performance_melted_line = chart_data.melt(id_vars='Model', var_name='Metric', value_name='Score', value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
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

    # Create columns for Age, Height, and Weight
    col_age, col_height, col_weight = st.columns(3)
    with col_age:
        col = 'Age'
        # Check if the column is in deployment_features before creating the input
        if col in deployment_features:
             input_data[col] = st.number_input(f"{col} (years):", value=0, min_value=0, help="Enter age in years") # Updated label and help

    with col_height:
        col = 'Height'
        if col in deployment_features:
            input_data[col] = st.number_input(f"{col} (m):", value=0.0, min_value=0.0, help="Enter height in meters") # Updated label and help

    with col_weight:
        col = 'Weight'
        if col in deployment_features:
            input_data[col] = st.number_input(f"{col} (kg):", value=0.0, min_value=0.0, help="Enter weight in kilograms") # Updated label and help

    # Create columns for FCVC and NCP
    col_fcvc, col_ncp = st.columns(2)

    with col_fcvc:
        col = 'FCVC'
        if col in deployment_features:
             selected_fcvc_text = st.selectbox("Frequency of consumption of vegetables:", fcvc_options)
             input_data[col] = fcvc_mapping[selected_fcvc_text] # Map text to numerical value

    with col_ncp:
        col = 'NCP'
        if col in deployment_features:
             # Changed to radio button input
             input_data[col] = st.radio("Number of main meals per day:", options=[1.0, 2.0, 3.0, 4.0])


    # Handle any remaining deployment features that were not explicitly placed in columns
    remaining_features = [col for col in deployment_features if col not in ['Age', 'Height', 'Weight', 'FCVC', 'NCP']]
    for col in remaining_features:
         # Check if the column is in deployment_features before creating the input
         if col in deployment_features:
             if col in categorical_cols_for_preprocessor:
                options = list(df[col].unique())
                input_data[col] = st.selectbox(f"{col}:", options)
             elif col in numerical_cols_for_preprocessor:
                input_data[col] = st.number_input(f"{col}:", value=0.0, min_value=0.0) # Assuming remaining numerical features should also be non-negative


    # Predict (with submit button)
    if st.button("Generate Prediction Report"):
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data using the fitted preprocessor
        # Ensure the input DataFrame has the same columns as the training data used for the preprocessor
        # This might require adding missing columns with default values (e.g., 0 for one-hot encoded)
        # For simplicity with current numerical-only features, we can proceed directly
        try:
            input_data_processed = preprocessor_deploy.transform(input_df[deployment_features])
        except ValueError as e:
             st.error(f"Error during preprocessing: {e}. Please check if all required features are provided.")
             input_data_processed = None # Set to None to prevent further errors


        st.header("4. Prediction Results")

        if input_data_processed is not None:
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

                # Add Feature Importance chart below prediction results
                st.subheader(f"Feature Relevance ({selected_model_name})") # Changed title to be more general

                # Get feature names directly from deployment_features since they are numerical
                feature_names = deployment_features

                # Add explicit check for SVM kernel
                is_svm = isinstance(model, SVC)
                svm_is_linear = is_svm and model.kernel == 'linear'

                if hasattr(model, 'feature_importances_'): # Use 'model' which is the selected model
                    st.subheader(f"Feature Importances ({selected_model_name})") # Specific title for importance
                    importances = model.feature_importances_ # Use the selected model's importances
                    if len(importances) == len(feature_names):
                        feat_importances = pd.Series(importances, index=feature_names)
                        feat_importances = feat_importances.sort_values(ascending=False)

                        fig4, ax4 = plt.subplots(figsize=(8, 6)) # Smaller figure size
                        feat_importances.plot(kind='barh', ax=ax4)
                        ax4.set_title(f'Feature Importances ({selected_model_name})') # Updated title
                        ax4.set_xlabel('Importance')
                        ax4.invert_yaxis()
                        st.pyplot(fig4)
                        plt.close(fig4)
                    else:
                        st.warning(f"Could not match feature importances to feature names. Number of importances ({len(importances)}) and feature names ({len(feature_names)}) do not match.")

                elif hasattr(model, 'coef_') and svm_is_linear: # Check if it has coef_ AND it's a linear SVM
                     st.subheader(f"Feature Coefficients (Absolute Mean) ({selected_model_name})") # Specific title for coefficients
                     # For multi-class, coef_ is shape (n_classes, n_features). Take the mean of absolute values.
                     coef_values = np.abs(model.coef_).mean(axis=0) # Use the selected model's coefficients

                     if len(coef_values) == len(feature_names):
                         feat_coef = pd.Series(coef_values, index=feature_names)
                         feat_coef = feat_coef.sort_values(ascending=False)

                         fig_coef, ax_coef = plt.subplots(figsize=(8, 6)) # Smaller figure size
                         feat_coef.plot(kind='barh', ax=ax_coef)
                         ax_coef.set_title(f'Feature Coefficients (Absolute Mean) ({selected_model_name})') # Updated title
                         ax_coef.set_xlabel('Absolute Mean Coefficient Value')
                         ax_coef.invert_yaxis()
                         st.pyplot(fig_coef)
                         plt.close(fig_coef)
                     else:
                         st.warning(f"Could not match feature coefficients to feature names. Number of coefficients ({len(coef_values)}) and feature names ({len(feature_names)}) do not match.")
                elif is_svm and not svm_is_linear:
                     st.info(f"The selected SVM model uses a non-linear kernel ({model.kernel}) and therefore does not have coefficients to display feature relevance.")
                     # Add Permutation Importance chart for non-linear SVM
                     st.subheader(f"Permutation Importance ({selected_model_name})")
                     try:
                        # Calculate permutation importance on the test set
                        # Use the preprocessor and model within a pipeline for consistent transformation
                        # Create a temporary pipeline for permutation importance calculation
                        # Note: Permutation importance works on the original features if the model is a pipeline
                        # However, if the model itself is loaded *without* its preprocessor pipeline,
                        # permutation_importance needs the *preprocessed* data X_test_processed_eval
                        # Since loaded_models contain just the classifier, we use X_test_processed_eval
                        result = permutation_importance(model, X_test_processed_eval, y_test_eval, n_repeats=10, random_state=42, n_jobs=-1)

                        # Get the importance scores and sort them
                        # Feature names should correspond to the columns of X_test_processed_eval
                        # Since X_test_processed_eval is from preprocessor_deploy, its columns correspond to deployment_features
                        sorted_importances_idx = result.importances_mean.argsort()
                        sorted_importances = result.importances_mean[sorted_importances_idx]
                        # Use the original deployment_features names for the plot labels
                        sorted_feature_names = [deployment_features[i] for i in sorted_importances_idx]


                        # Create the bar chart
                        fig_perm, ax_perm = plt.subplots(figsize=(8, 6))
                        ax_perm.barh(sorted_feature_names, sorted_importances)
                        ax_perm.set_title(f"Permutation Importance ({selected_model_name})")
                        ax_perm.set_xlabel("Mean Decrease in Accuracy") # Or other metric if specified
                        st.pyplot(fig_perm)
                        plt.close(fig_perm)

                     except Exception as e:
                         st.error(f"An error occurred while generating Permutation Importance chart: {e}")

                else:
                    st.info(f"The selected model ({selected_model_name}) does not have feature importances or coefficients to display.")


else:
    st.warning("Model or data not loaded. Please ensure 'all_k-fold_models.joblib' and 'ObesityDataSet.csv' are in the correct directory.")
