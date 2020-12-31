"""Webpage of Bulky Prediction
This page includes input data uploading and the corresponding prediction downloading
Please edit if necessary
"""

# Import Libraries
import streamlit as st
import pickle
import pandas as pd
import base64
import xgboost as xgb
import lime
from lime import lime_tabular  # lime.lime_tabular doesn't work here, has to use "from lime import lime_tabular instead"


# User Parameters
ORGMODEL = "Model/test.pkl"

pred_varlist = [
        # Utilization
        "90-day DRP Readmission in Past 2 Years", "No. Visits to Emergency Department in Past 6 Months",
        "No. Visits to Specialist of Clinic in Past 1 Year",
        "Length of Stay for This Inpatient Visit", "Length of Stay for Last Inpatient Visit",
        "No. Days Since the Discharge of Last Inpatient Visit",
        "Average Length of Stay for Inpatient Visits in Past 1 Year",
        # Diagnosis
        "No. Diagnosis with Drug-Related Problem in Past 1 Year",
        "No. Diagnosis with Chronic Disease in Past 2 Years", "No. Diagnosis with Pulmonary Disease in Past 2 Years",
        "No. Diagnosis with Diabetes without Long-term Complication in Past 2 Years",
        "No. Diagnosis with Asthma in Past 2 Years", "No. Diagnosis with Hypertension in Past 2 Years",
        "No. Diagnosis with Coronary Heart Disease in Past 2 Years",
        "No. Diagnosis with Heart Failure in Past 2 Years", "No. Diagnosis with COPD Diagnosis in Past 2 Years",
        "No. Diseases in the Charlson Comorbidity Index Being Diagnosed in Past 2 Years",
        # Lab
        "Average Anion Gap Value Being Tested in Past 1 Year", "Average Bilirubin Value Being Tested in Past 1 Year",
        # Medication
        "No. Cardiovascular Medicines Taken in Past 1 Year", "No. Nervous Medicines Taken in Past 1 Year",
        "No. Respiratory Medicines Taken in Past 1 Year", "No. ATC Drug Groups Taken in Past 1 Year",
        # Speciality
        "Visit Gastroenterology Last Time", "No. Visits to the Specialty of General Surgery in Past 1 Year",
        "Visit Geriatric Medication Last Time", "Visit Urology Last Time",
        "No. Specialities Being Visited in Past 1-2 Years", "No. Discharge Specialties Being Visited in Past 1 Year",
        # Demographics
        "Age in Year"
    ]
pred_varlist_short = [
    # Utilization
    "CT_RD90_DRP_LMD_P0_24", "P0_6ED", "P0_12SOC", "LOS_CURRENT", "LOS_LASTTIME",
    "DAYS_SINCE_LAST_INP", "DURATION_AVG_P0_12",
    # Diagnosis
    "DRP_COUNTS_P0_12", "CD_COUNTS_P0_24", "CCI_PULMONARY", "CCI_DIABETES_NO_LONG",
    "CDMS_ASTHMA", "CDMS_HYPERTENSION", "CDMS_CHD", "CDMS_HF", "CDMS_COPD", "NUM_CCI",
    # Lab
    "P0_12_AVG_33037-3", "P0_12_AVG_14631-6",
    # Medication
    "GRPC_NUM_ATCMED_P0_12", "GRPN_NUM_ATCMED_P0_12", "GRPR_NUM_ATCMED_P0_12", "NUM_DRUGGRP_P0_12",
    # Speciality
    "GASTRO_LASTTIME", "GENSUR_NUM_VISIT_P0_12", "GERMED_LASTTIME", "UROLOGY_LASTTIME",
    "NUM_SPEC_P12_24", "NUM_DISCHARGE_SPEC_P0_12",
    # Demographics
    "AGE"
]
spec_colnames = pd.DataFrame({'Feature': pred_varlist, 'Column Name': pred_varlist_short})

"""Webpage Content"""

# Step 1: Ensure Column Order
st.markdown("""* **Step 1:** Ensure **30 Features** are Defined with Specified Column Names""")
disporder_but_pred = st.button("Display Specified Column Names")
if disporder_but_pred:
    st.dataframe(spec_colnames)

# Step 2: Upload Input Data
st.markdown("""* **Step 2:** Upload Data in **CSV** Format""")
pred_datafile = st.file_uploader("", type=['CSV'])
if pred_datafile is not None:
    pred_dataset = pd.read_csv(pred_datafile)
    pred_dataset.drop(columns=["Unnamed: 0"], inplace=True)
    pred_dataset.columns = pred_varlist_short

# Step 3: Predict
st.markdown("""* **Step 3:** Download Prediction""")
orgmodel = pickle.load(open(ORGMODEL, 'rb'))
if pred_datafile is not None:
    pred_dataset["Prediction"] = orgmodel.predict_proba(pred_dataset)[:, 1]

    # Display Some Results
    st.markdown("*Display Some Results*")
    st.dataframe(pred_dataset.head())

    b64 = base64.b64encode(pred_dataset.to_csv(index=False).encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download Prediction</a> (Right-click and save as <filename>.csv)'
    st.markdown(href, unsafe_allow_html=True)


