import streamlit as st
import pandas as pd
import pickle
import zipfile

# Load models
@st.cache_resource
def load_models():
    with zipfile.ZipFile('/mount/src/waze_customer_churn_analysis/ChurnModelandPreprocessing.zip', 'r') as zip_ref:
        zip_ref.extractall()

    with open('Churn_model.pkl', 'rb') as f:
        Churn_model = pickle.load(f)

    return Churn_model 

# Load the models
Churn_model = load_models()

# Streamlit App Code
st.title("Waze Customer Churn Analysis")

# File uploader for test data
uploaded_file = st.file_uploader("Upload test file as .csv", type=["csv"])

# Placeholder for results to allow clearing
results_placeholder = st.empty()

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        test_data = pd.read_csv(uploaded_file, header=None)
        test_data.columns = ['activity_days', 'drives', 'driving_days', 'n_days_after_onboarding', 
                             'total_sessions', 'sessions', 'total_navigations_fav2', 
                             'total_navigations_fav1', 'driven_km_drives', 'duration_minutes_drives']
        
        num_columns = test_data.shape[1]

        # Check for the expected number of columns
        if num_columns < 13:
            st.error("Uploaded file must contain exactly 13 columns for features.")
        else:
            for col in test_data.select_dtypes(include=['object']).columns:
                test_data[col] = pd.Categorical(test_data[col]).codes
            
            # First, use the model to predict churn
            if st.button("Predict"):
                # Clear previous results
                results_placeholder.empty()

                # Predict using the model
                binary_pred = Churn_model.predict(test_data)
                results_data = []

                # Collect the results
                for pred in binary_pred:
                    result = "Churned Customer" if pred == 0 else "Retained Customer"
                    results_data.append([result])  # Store result in a single list

                # Create a DataFrame with one column for predictions
                results_df = pd.DataFrame(results_data, columns=["Prediction"])

                # Display the DataFrame in Streamlit without index
                results_placeholder.write(results_df)

    except Exception as e:
        st.error(f"Error reading the file: {e}")

# About section with updated information
st.sidebar.title("About")
st.sidebar.info("""
This app classifies the data from Waze using a pre-trained XGBoost model to determine if a customer is predicted to be churned or not.
Upload a CSV file with test data (10 columns for features) and receive a prediction.
""")
