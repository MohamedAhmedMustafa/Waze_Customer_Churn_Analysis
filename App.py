#import pip
#pip.main(['install','cmake'])
#pip.main(['install','pyarrow'])
#pip.main(['install','streamlit'])
from My_function import preprocessing_df
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
        num_columns = test_data.shape[1]

        if num_columns < 12:
            st.error("Uploaded file must contain at least 12 columns.")
        else:
            print(test_data)
            # Use only the first 100 columns if there are more
            # Convert 'device' column to categorical codes
            test_data['device'] = pd.Categorical(test_data['device']).codes
            
            # Drop the 'ID' column
            test_data = test_data.drop(columns=['ID'])
            
            # Drop any rows with missing values
            test_data = test_data.dropna()
            
            # Calculate the 95th percentile for each column
            percentiles_99 = test_data.quantile(0.95)
            
            # Filter out records where values exceed the 95th percentile in any column
            test_data_filtered = test_data[(test_data <= percentiles_99).all(axis=1)]
            
            # Reassign filtered data back to test_data
            test_data = test_data_filtered
            
            # Drop the 'device' column
            test_data = test_data.drop(columns=['device'])
            
            # Drop rows with missing values again, if any
            test_data = test_data.dropna()
            
            # First, use XGBoost to predict if the ECG is abnormal
            if st.button("Predict"):
                # Clear previous results
                results_placeholder.empty()

                # Predict using XGBoost
                binary_pred = Churn_model.predict(test_data)
                results_data = []

                # Loop through each prediction and collect the result
                for idx, pred in enumerate(binary_pred):
                    result = "Churned Customer" if pred == 0 else "Retained Customer"
                    row = [result]
                    results_data.append(row)

                # Create a DataFrame without the Index column
                results_df = pd.DataFrame(results_data, columns=["Binary Prediction", "Detailed Prediction"])

                # Display the DataFrame in Streamlit without index
                results_placeholder.write(results_df)

    except Exception as e:
        st.error(f"Error reading the file: {e}")

# About section with updated information
st.sidebar.title("About")
st.sidebar.info("""
This app classifies the data from Waze cmopany using a pre-trained XGBoost model to determine if a customer is predicted to be churned or not.
Upload a CSV file with test data (12 columns for features) and receive a prediction.
""")
