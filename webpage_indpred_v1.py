"""Webpage of Individual Prediction
This page includes inputs of one record and its prediction
Please edit if necessary
"""

# Import Libraries
import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb
import lime
from lime import lime_tabular  # lime.lime_tabular doesn't work here, has to use "from lime import lime_tabular instead"


# User Parameters
ORGMODEL = "Model/test.pkl"



"""Webpage Content"""

# Step 1: Input Data
st.markdown("""* **Step 1:** Input Values of 30 Features""")
st.write("Utilization")
util_c1, util_c2, util_c3 = st.beta_columns(3)
pred_CT_RD90_DRP_LMD_P0_24 = util_c1.number_input("90-day DRP Readmission in Past 2 Years", format="%.0f")
pred_P0_6ED = util_c1.number_input("No. Visits to Emergency Department in Past 6 Months", format="%.0f")
pred_P0_12SOC = util_c1.number_input("No. Visits to Specialist of Clinic in Past 1 Yr", format="%.0f")
pred_LOS_CURRENT = util_c2.number_input("Length of Stay for This Inpatient Visit (Days)", format="%.0f")
pred_LOS_LASTTIME = util_c2.number_input("Length of Stay for Last Inpatient Visit (Days)", format="%.0f")
pred_DAYS_SINCE_LAST_INP = util_c3.number_input("No. Days Since the Discharge of Last Inpatient Visit", format="%.0f")
pred_DURATION_AVG_P0_12 = util_c3.number_input("Ave. Length of Stay for Inpatient Visits in Past 1 Yr (Days)",
                                               format="%.0f")

st.write("Diagnosis")
diag_c1, diag_c2, diag_c3 = st.beta_columns(3)
pred_DRP_COUNTS_P0_12 = diag_c1.number_input("No. Diagnosis with Drug-Related Problem in Past 1 Year", format="%.0f")
pred_CD_COUNTS_P0_24 = diag_c1.number_input("No. Diagnosis with Chronic Disease in Past 2 Years", format="%.0f")
pred_CCI_PULMONARY = diag_c1.number_input("No. Diagnosis with Pulmonary Disease in Past 2 Years", format="%.0f")
pred_CCI_DIABETES_NO_LONG = diag_c1.number_input(
    "No. Diagnosis with Diabetes without Long-term Complication in Past 2 Yrs", format="%.0f")
pred_CDMS_ASTHMA = diag_c2.number_input("No. Diagnosis with Asthma in Past 2 Years", format="%.0f")
pred_CDMS_HYPERTENSION = diag_c2.number_input("No. Diagnosis with Hypertension in Past 2 Years", format="%.0f")
pred_CDMS_CHD = diag_c2.number_input("No. Diagnosis with Coronary Heart Disease in Past 2 Years", format="%.0f")
pred_CDMS_HF = diag_c3.number_input("No. Diagnosis with Heart Failure in Past 2 Years", format="%.0f")
pred_CDMS_COPD = diag_c3.number_input("No. Diagnosis with COPD Diagnosis in Past 2 Years", format="%.0f")
pred_NUM_CCI = diag_c3.number_input(
    "No. Diseases in Charlson Comorbidity Index Diagnosed in Past 2 Yrs", format="%.0f")

st.write("Lab")
lab_c1, lab_c2 = st.beta_columns(2)
pred_P0_12_AVG_33037_3 = lab_c1.number_input("Average Anion Gap Value Being Tested in Past 1 Year")
pred_P0_12_AVG_14631_6 = lab_c2.number_input("Average Bilirubin Value Being Tested in Past 1 Year")

st.write("Medication")
med_c1, med_c2 = st.beta_columns(2)
pred_GRPC_NUM_ATCMED_P0_12 = med_c1.number_input("No. Cardiovascular Medicines Taken in Past 1 Year", format="%.0f")
pred_GRPN_NUM_ATCMED_P0_12 = med_c1.number_input("No. Nervous Medicines Taken in Past 1 Year", format="%.0f")
pred_GRPR_NUM_ATCMED_P0_12 = med_c2.number_input("No. Respiratory Medicines Taken in Past 1 Year", format="%.0f")
pred_NUM_DRUGGRP_P0_12 = med_c2.number_input("No. ATC Drug Groups Taken in Past 1 Year", format="%.0f")

st.write("Specialty")
spec_c1, spec_c2, spec_c3 = st.beta_columns(3)
pred_GASTRO_LASTTIME = spec_c1.radio("Visit Gastroenterology Last Time", options=[True, False])
pred_GENSUR_NUM_VISIT_P0_12 = spec_c1.number_input("No. Visits to the Specialty of General Surgery in Past 1 Year", format="%.0f")
pred_GERMED_LASTTIME = spec_c2.radio("Visit Geriatric Medication Last Time", options=[True, False])
pred_NUM_SPEC_P12_24 = spec_c2.number_input("No. Specialities Being Visited in Past 1-2 Years", format="%.0f")
pred_UROLOGY_LASTTIME = spec_c3.radio("Visit Urology Last Time", options=[True, False])
pred_NUM_DISCHARGE_SPEC_P0_12 = spec_c3.number_input("No. Discharge Specialties Being Visited in Past 1 Year", format="%.0f")

st.write("Demographics")
demo_c1, demo_c2 = st.beta_columns(2)
pred_AGE = demo_c1.number_input("Age in Year", format="%.0f")

pred_data = pd.DataFrame({"CT_RD90_DRP_LMD_P0_24": [pred_CT_RD90_DRP_LMD_P0_24], "P0_6ED": [pred_P0_6ED],
                          "P0_12SOC": [pred_P0_12SOC], "LOS_CURRENT": [pred_LOS_CURRENT],
                          "LOS_LASTTIME": [pred_LOS_LASTTIME], "DAYS_SINCE_LAST_INP": [pred_DAYS_SINCE_LAST_INP],
                          "DURATION_AVG_P0_12": [pred_DURATION_AVG_P0_12],
                          "DRP_COUNTS_P0_12": [pred_DRP_COUNTS_P0_12], "CD_COUNTS_P0_24": [pred_CD_COUNTS_P0_24],
                          "CCI_PULMONARY": [pred_CCI_PULMONARY], "CCI_DIABETES_NO_LONG": [pred_CCI_DIABETES_NO_LONG],
                          "CDMS_ASTHMA": [pred_CDMS_ASTHMA], "CDMS_HYPERTENSION": [pred_CDMS_HYPERTENSION],
                          "CDMS_CHD": [pred_CDMS_CHD], "CDMS_HF": [pred_CDMS_HF], "CDMS_COPD": [pred_CDMS_COPD],
                          "NUM_CCI": [pred_NUM_CCI],
                          "P0_12_AVG_33037-3": [pred_P0_12_AVG_33037_3], "P0_12_AVG_14631-6": [pred_P0_12_AVG_14631_6],
                          "GRPC_NUM_ATCMED_P0_12": [pred_GRPC_NUM_ATCMED_P0_12],
                          "GRPN_NUM_ATCMED_P0_12": [pred_GRPN_NUM_ATCMED_P0_12],
                          "GRPR_NUM_ATCMED_P0_12": [pred_GRPR_NUM_ATCMED_P0_12],
                          "NUM_DRUGGRP_P0_12": [pred_NUM_DRUGGRP_P0_12],
                          "GASTRO_LASTTIME": [pred_GASTRO_LASTTIME],
                          "GENSUR_NUM_VISIT_P0_12": [pred_GENSUR_NUM_VISIT_P0_12],
                          "GERMED_LASTTIME": [pred_GERMED_LASTTIME], "UROLOGY_LASTTIME": [pred_UROLOGY_LASTTIME],
                          "NUM_SPEC_P12_24": [pred_NUM_SPEC_P12_24],
                          "NUM_DISCHARGE_SPEC_P0_12": [pred_NUM_DISCHARGE_SPEC_P0_12],
                          "AGE": [pred_AGE]})

# Step 2: Predict
st.markdown("""* **Step 2:** Predict the Probability of 90-Day Drug-Related Readmission""")
pred_c1, pred_c2 = st.beta_columns(2)
pred_but = pred_c1.button("Predict")

if pred_but:
    orgmodel = pickle.load(open(ORGMODEL, 'rb'))
    pred_c2.write(round(orgmodel.predict_proba(pred_data)[0][1], 3))
