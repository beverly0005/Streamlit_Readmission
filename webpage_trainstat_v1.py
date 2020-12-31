"""Webpage of Retrain (Dev)
This page includes the statistics on original data
Please edit if necessary
"""

# Import Libraries
import streamlit as st
import pandas as pd
from utils import *

# User Parameters
DATAPATH = 'Data/'
DF_TRAIN_SUM = 'train_sum.csv'

pred_varlist = [
        # Utilization
        "90-day DRP Readm in Past 2 Yrs", "No. Visits to ED in Past 6 Months",
        "No. Visits to SOC in Past 1 Yr",
        "LOS for This Time", "LOS for Last Time",
        "Days Since Last Inpat. Disch",
        "Ave LOS for Inpatient Visits in Past 1 Yr",
        # Diag
        "No. Diag with DRP in Past 1 Yr",
        "No. Diag with Chronic Disease in Past 2 Yrs", "No. Diag with Pulmonary Disease in Past 2 Yrs",
        "No. Diag with Diabetes without Long-term Complication in Past 2 Yrs",
        "No. Diag with Asthma in Past 2 Yrs", "No. Diag with Hypertension in Past 2 Yrs",
        "No. Diag with Coronary Heart Disease in Past 2 Yrs",
        "No. Diag with Heart Failure in Past 2 Yrs", "No. Diag with COPD Diag in Past 2 Yrs",
        "No. Diseases in the Charlson Comorbidity Index Being Diagnosed in Past 2 Yrs",
        # Lab
        "Ave Anion Gap Value Tested in Past 1 Yr", "Ave Bilirubin Value Tested in Past 1 Yr",
        # Medication
        "No. Cardiovascular Med Taken in Past 1 Yr", "No. Nervous Med Taken in Past 1 Yr",
        "No. Respiratory Med Taken in Past 1 Yr", "No. ATC Groups Taken in Past 1 Yr",
        # Speciality
        "Visit Gastroenterology Last Time", "No. Visits to the Specialty of General Surgery in Past 1 Yr",
        "Visit Geriatric Medication Last Time", "Visit Urology Last Time",
        "No. Specialities Visited in Past 1-2 Yrs", "No. Discharge Specialties Visited in Past 1 Yr",
        # Demographics
        "Age in Yr"
    ]
pred_varlist_short = [
    # Utilization
    "CT_RD90_DRP_LMD_P0_24", "P0_6ED", "P0_12SOC", "LOS_CURRENT", "LOS_LASTTIME",
    "DAYS_SINCE_LAST_INP", "DURATION_AVG_P0_12",
    # Diag
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


# Load Data
df_train_sum = pd.read_csv(DATAPATH + DF_TRAIN_SUM)
df_train_sum = pd.merge(df_train_sum, spec_colnames, left_on=['variable'], right_on=['Column Name'])
df_train_sum.drop(columns=['Unnamed: 0', 'variable'], inplace=True)
df_train_sum_varlist = df_train_sum.columns
df_train_sum = df_train_sum[['Feature'] + [i for i in df_train_sum_varlist if i != 'Feature']]


"""Webpage Content"""
st.header("Summary on 30 Features in Final Model")
st.warning("""Notes: **miss** refers to the number of missing values in a feature, **sd** refers to 
            the standard deviation of a feature, **unique** refers to the number of unique values in a feature, 
            **mode** refers to the mode of a feature, **q25** refers to the 25 percentile of a feature, 
            **q75** refers to the 75 percentile of a feature, **outlier_3d** refers to the number of outliers 
            beyond 3 standard deviation from the mean of a feature, **per** refers to the percent of the 
            corresponding conditions in a feature,""")

sel_fea = multiselect_dropdown("Display Statistics of Selected Variables",
                               options=df_train_sum['Feature'].values, default=[])
if sel_fea==[]:
    st.dataframe(df_train_sum)
else:
    st.dataframe(df_train_sum[df_train_sum['Feature'].isin(sel_fea)])
