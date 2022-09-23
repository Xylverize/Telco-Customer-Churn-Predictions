import json
import pandas as pd
import pickle
import streamlit as st
import requests

# load pipeline
pipe = pickle.load(open("model/preprocessor.pkl", "rb"))

st.title("Aplikasi Prediksi Churn")

gender = st.selectbox("Customer Gender", ["Male", "Female"])

seniorCitizen = st.selectbox("Whether the customer is a senior citizen or not", ["No", "Yes"] )

partner = st.selectbox("Whether the customer has a partner or not", ["No", "Yes"])

dependent = st.selectbox(" Whether the customer has dependents or not", ["No", "Yes"])

tenure = st.number_input("Number of months the customer has stayed with the company? (Tenure)", min_value=1)

phoneService = st.selectbox("Phone service ", ["No", "Yes"])

multipleLines = st.selectbox("Multiple Lines ", ["No", "Yes","No phone service"])

internetService = st.selectbox("Internet Service Provider ", ["No", "DSL", "Fiber optic"])

onlineSecurity = st.selectbox("Online Security ", ["No", "Yes","No internet service"])

onlineBackup = st.selectbox("Online Backup ", ["No", "Yes","No internet service"])

deviceProtection = st.selectbox("Device Protection", ["No", "Yes","No internet service"])

techSupport = st.selectbox("Tech Support ", ["No", "Yes","No internet service"])

streamingTV = st.selectbox("Streaming TV ", ["No", "Yes","No internet service"])
       
streamingMovies = st.selectbox("Streaming Movies ", ["No", "Yes","No internet service"])

contract = st.selectbox("Contract ", ["Month-to-month", "One year", "Two year"])

paperlessBilling = st.selectbox("Paperless Billing ", ["Yes", "No"])

paymentMethod = st.selectbox("Payment Method ",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

monthlyCharges = st.number_input("Monthly Charges", value=18.0,min_value=18.0)    

totalCharges = st.number_input("Total Charges ",value=18.0,min_value=18.0) 


# Create dictionary with all customer information
new_data = {
        "gender": gender,
        "SeniorCitizen": seniorCitizen,
        "Partner": partner,
        "Dependents": dependent,
        "tenure": tenure,
        "PhoneService": phoneService,
        "MultipleLines": multipleLines,
        "InternetService": internetService,
        "OnlineSecurity": onlineSecurity,
        "OnlineBackup": onlineBackup,
        "DeviceProtection": deviceProtection,
        "TechSupport": techSupport,
        "StreamingTV": streamingTV,
        "StreamingMovies": streamingMovies,
        "Contract": contract,
        "PaperlessBilling": paperlessBilling,
        "PaymentMethod": paymentMethod,
        "MonthlyCharges": monthlyCharges,
        "TotalCharges": totalCharges,
    }



new_data = pd.DataFrame([new_data])

# build feature
new_data = pipe.transform(new_data)
new_data = new_data.tolist()

# inference
URL = "https://tf-serving-xyla-backend.herokuapp.com/v1/models/churn_model:predict"
param = json.dumps({
        "signature_name":"serving_default",
        "instances":new_data
    })
r = requests.post(URL, data=param)

if r.status_code == 200:
    res = r.json()
    if res['predictions'][0][0] > 0.5:
        st.title("Churn")
    else:
        st.title("Not Churn")
else:
    st.title("Unexpected Error")