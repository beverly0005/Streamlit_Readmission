"""Webpage of About
This page includes the motivation of the project, model performance and methodology.
Please add in necessary information
"""

# Import Libraries
import streamlit as st
from PIL import Image


# User Parameters
MODELFULL_PERF_FIG = 'Figure/modelfull_perf_fig.png'
STUDY_DSG = 'Figure/studydesign.png'


"""Webpage Content"""

# Motivation
st.header("Motivation")
st.markdown("""Drug-related problem (such as adverse drug reactions, adherence problems, underuse) is one of 
many factors mostly avoidable in readmission events. To reduce readmission caused by drug-related problems, 
pharmacies can do home visits to high risk patients and deliver services such as medication reconciliation, 
discharge counselling and drug reviews. This intervention also helps patients save time waiting for discharge
medication on campus. 
\n Hence, it is critical to identify the patients with high risk of drug-related readmission. Our model
contributes to predict **the risk of drug-related readmission for patients with chronic diseases** at the 
event level with high accuracy.
""")

# Model Performance
st.header("Model Performance")
st.image(Image.open(MODELFULL_PERF_FIG), use_column_width=True)

# Methodology
st.header("Methodology")
st.markdown("""Our model is trained on 117k inpatient records with 381 features in the original data. The time range of data 
and the study design are shown as below.""")
st.image(Image.open(STUDY_DSG), use_column_width=True)
