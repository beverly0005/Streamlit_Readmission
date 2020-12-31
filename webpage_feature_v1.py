"""Webpage of Feature List (Dev)
This page includes the feature importance in the final model, the list of selected features in the final model of
different categories, as well as the list of original features.
Please edit if necessary
"""

# Import Libraries
import streamlit as st
from PIL import Image
import pandas as pd


# User Parameters
FEAIMP_FIG = 'Figure/feaimp_fig.png'

DATAPATH = 'Data/'
ALLCOLUMNFILE = 'allcolumns.xls'
DF_COL_UTIL = pd.read_excel(DATAPATH + ALLCOLUMNFILE, sheet_name='utilization', header=None)
DF_COL_DIAG = pd.read_excel(DATAPATH + ALLCOLUMNFILE, sheet_name='diagnosis', header=None)
DF_COL_LAB = pd.read_excel(DATAPATH + ALLCOLUMNFILE, sheet_name='labtest', header=None)
DF_COL_MED = pd.read_excel(DATAPATH + ALLCOLUMNFILE, sheet_name='medication', header=None)
DF_COL_SPEC = pd.read_excel(DATAPATH + ALLCOLUMNFILE, sheet_name='specialty', header=None)
DF_COL_DEMO = pd.read_excel(DATAPATH + ALLCOLUMNFILE, sheet_name='demographics', header=None)


"""Webpage Content"""

st.markdown("""There are 381 features in the original data. We start training models on these 381 features, but 
are able to reduce to **30 features** in the final model.""")

# Feature List
if st.sidebar.checkbox("Display Feature List"):
    st.header("Feature Category")
    disp_feature = st.radio("", ("Utilization", "Diagnosis", "Lab Test", "Medication",
                                 "Specialty", "Demographics"))
    if disp_feature == "Utilization":
        st.subheader("""7 out of """ + str(DF_COL_UTIL.shape[0]) + """ features has been selected in the final model:
                     90-day DRP Readmission in Past 2 Years,
                     No. Visits to Specialist of Clinic in Past 1 Year,
                     No. Visits to Emergency Department in Past 6 Months,
                     No. Days Since the Discharge of Last Inpatient Visit,
                     Length of Stay for This Inpatient Visit,
                     Length of Stay for Last Inpatient Visit,
                     Average Length of Stay for Inpatient Visits in Past 1 Year""")
        if st.checkbox("Show Original List"):
            st.warning("""Notes: The table below shows the original column names without interpretation. 
                       All column names with **P0_12** refer to the features created using past 1 year data, 
                       **P12_24** refer to those using past 1-2 year data, **LASTTIME** refer to those 
                       using data of immediately before this inpatient visit, **CURRENT** refer to those 
                       using the data of current inpatient visit, **BEF_DISCHARGE** refer to those using the 
                       data of current inpatient visit before the discharge date.""")
            st.dataframe(DF_COL_UTIL.rename(columns={0:'Feature Name'}))
    elif disp_feature == "Diagnosis":
        st.subheader("""10 out of """ + str(DF_COL_DIAG.shape[0]) + """ features has been selected in the final model:
                     No. Diagnosis with Drug-Related Problem in Past 1 Year, 
                     No. Diagnosis with COPD Diagnosis in Past 2 Years, 
                     No. Diagnosis with Chronic Disease in Past 2 Years, 
                     No. Diagnosis with Asthma in Past 2 Years, 
                     No. Diagnosis with Pulmonary Disease in Past 2 Years,
                     No. Diagnosis with Hypertension in Past 2 Years,
                     No. Diagnosis with Heart Failure in Past 2 Years,
                     No. Diagnosis with Coronary Heart Disease in Past 2 Years,
                     No. Diagnosis with Diabetes without Long-term Complication in Past 2 Years,
                     No. Diseases in the Charlson Comorbidity Index Being Diagnosed in Past 2 Years""")
        if st.checkbox("Show Original List"):
            st.warning("""Notes: The table below shows the original column names without interpretation. 
                                   All column names with **P0_12** refer to the features created using past 1 year data, 
                                   **P12_24** refer to those using past 1-2 year data, **LASTTIME** refer to those 
                                   using data of immediately before this inpatient visit, **CURRENT** refer to those 
                                   using the data of current inpatient visit, **BEF_DISCHARGE** refer to those using the 
                                   data of current inpatient visit before the discharge date.""")
            st.dataframe(DF_COL_DIAG.rename(columns={0:'Feature Name'}))
    elif disp_feature == "Lab Test":
        st.subheader("""2 out of """ + str(DF_COL_LAB.shape[0]) + """ features has been selected in the final model:
                     Average Bilirubin Value Being Tested in Past 1 Year,
                     Average Anion Gap Value Being Tested in Past 1 Year""")
        if st.checkbox("Show Original List"):
            st.warning("""Notes: The table below shows the original column names without interpretation. 
                                   All column names with **P0_12** refer to the features created using past 1 year data, 
                                   **P12_24** refer to those using past 1-2 year data, **LASTTIME** refer to those 
                                   using data of immediately before this inpatient visit, **CURRENT** refer to those 
                                   using the data of current inpatient visit, **BEF_DISCHARGE** refer to those using the 
                                   data of current inpatient visit before the discharge date.""")
            st.dataframe(DF_COL_LAB.rename(columns={0:'Feature Name'}))
    elif disp_feature == "Medication":
        st.subheader("""4 out of """ + str(DF_COL_MED.shape[0]) + """ features has been selected in the final model:
                    No. Respiratory Medicines Taken in Past 1 Year,
                    No. Nervous Medicines Taken in Past 1 Year,
                    No. Cardiovascular Medicines Taken in Past 1 Year, 
                    No. ATC Drug Groups Taken in Past 1 Year""")
        if st.checkbox("Show Original List"):
            st.warning("""Notes: The table below shows the original column names without interpretation. 
                                   All column names with **P0_12** refer to the features created using past 1 year data, 
                                   **P12_24** refer to those using past 1-2 year data, **LASTTIME** refer to those 
                                   using data of immediately before this inpatient visit, **CURRENT** refer to those 
                                   using the data of current inpatient visit, **BEF_DISCHARGE** refer to those using the 
                                   data of current inpatient visit before the discharge date.""")
            st.dataframe(DF_COL_MED.rename(columns={0:'Feature Name'}))
    elif disp_feature == "Specialty":
        st.subheader("""6 out of """ + str(DF_COL_SPEC.shape[0]) + """ features has been selected in the final model:
                    Visit Gastroenterology Last Time, 
                    Visit Urology Last Time,
                    Visit Geriatric Medication Last Time,
                    No. Visits to the Specialty of General Surgery in Past 1 Year,
                    No. Specialities Being Visited in Past 1-2 Years,
                    No. Discharge Specialties Being Visited in Past 1 Year""")
        if st.checkbox("Show Original List"):
            st.warning("""Notes: The table below shows the original column names without interpretation. 
                                   All column names with **P0_12** refer to the features created using past 1 year data, 
                                   **P12_24** refer to those using past 1-2 year data, **LASTTIME** refer to those 
                                   using data of immediately before this inpatient visit, **CURRENT** refer to those 
                                   using the data of current inpatient visit, **BEF_DISCHARGE** refer to those using the 
                                   data of current inpatient visit before the discharge date.""")
            st.dataframe(DF_COL_SPEC.rename(columns={0:'Feature Name'}))
    elif disp_feature == "Demographics":
        st.subheader("""1 out of """ + str(DF_COL_DEMO.shape[0]) + """ features has been selected in the final model:
                    Age""")
        if st.checkbox("Show Original List"):
            st.warning("""Notes: The table below shows the original column names without interpretation. 
                                   All column names with **P0_12** refer to the features created using past 1 year data, 
                                   **P12_24** refer to those using past 1-2 year data, **LASTTIME** refer to those 
                                   using data of immediately before this inpatient visit, **CURRENT** refer to those 
                                   using the data of current inpatient visit, **BEF_DISCHARGE** refer to those using the 
                                   data of current inpatient visit before the discharge date.""")
            st.dataframe(DF_COL_DEMO.rename(columns={0: 'Feature Name'}))

# Feature Importance
st.header("Feature Importance: Top 15 in Final Model")
st.markdown("""*Features importance is measured by the number of splits in the model.*""")
st.image(Image.open(FEAIMP_FIG), use_column_width=True)
