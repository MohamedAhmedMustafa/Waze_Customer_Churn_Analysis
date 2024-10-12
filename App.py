#import pip
#pip.main(['install','cmake'])
#pip.main(['install','pyarrow'])
#pip.main(['install','streamlit'])
import streamlit as st
import pandas as pd
import pickle
import zipfile

# Load models
@st.cache_resource
def load_models():
    with zipfile.ZipFile('ChurnModelandPreprocessing.zip', 'r') as zip_ref:
        zip_ref.extractall()

    with open('Churn_model.pkl', 'rb') as f:
        Churn_model = pickle.load(f)

    with open('preprocessing.pkl', 'rb') as f:
        loaded_function = pickle.load(f)

    return Churn_model ,loaded_func 

# Load the models
Churn_model ,preprocessing_df = load_models()

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
            # Use only the first 100 columns if there are more
            test_data = preprocessing_df(test_data)
            X_test = test_data.iloc[:, :]

            # First, use XGBoost to predict if the ECG is abnormal
            if st.button("Predict"):
                # Clear previous results
                results_placeholder.empty()

                # Predict using XGBoost
                binary_pred = Churn_model.predict(X_test)
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
