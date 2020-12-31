"""Webpage of Visualization
This page includes input feature selection and visualization
Please edit if necessary
"""

# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

# User Parameters
DATAPATH = 'Data/'
VISUALFILE = 'visualdata.csv'

retrain_varlist_short = [
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
    "AGE"]
retrain_varlist = [
    # Utilization
    "90-day DRP Readm. in Past 2 Yrs", "No. ED Visits in Past 6 Mths",
    "No. SOC Visits in Past 1 Yr",
    "LOS This Time", "LOS Last Time",
    "Days Since Last Inpat. Disch.",
    "Ave. LOS in Past 1 Yr",
    # Diagnosis
    "No. DRP Diag. in Past 1 Yr",
    "No. CD Diag. in Past 2 Ys", "No. Pulmonary Diag. in Past 2 Yrs",
    "No. Diab w/o long. Complica. Diag. in Past 2 Yrs",
    "No. Asthma Diag. in Past 2 Yrs", "No. Hyperten. Diag. in Past 2 Yrs",
    "No. CHD Diag. in Past 2 Yrs",
    "No. HF Diag. in Past 2 Yrs", "No. COPD Diag. in Past 2 Yrs",
    "No. CCI Diag. in Past 2 Yrs",
    # Lab
    "Ave Anion Gap in Past 1 Yr", "Ave Bilirubin in Past 1 Yr",
    # Medication
    "No. Cardio. Med in Past 1 Yr", "No. Nerv. Med in Past 1 Yr",
    "No. Respir. Med in Past 1 Yr", "No. ATC Groups in Past 1 Yr",
    # Speciality
    "Visit Gastro. Last Time", "Visits to Gen. Surgery in Past 1 Yr",
    "Visit Geriatric Med. Last Time", "Visit Urology Last Time",
    "No. Spec. Visited in Past 1-2 Yrs", "No. Disch. Spec. Visited in Past 1 Yr",
    # Demographics
    "Age in Year"
]
spec_colnames = pd.DataFrame({'Feature': retrain_varlist, 'Column Name': retrain_varlist_short})

# Load data
visual_data = pd.read_csv(DATAPATH + VISUALFILE)
visual_data.drop(columns=["Unnamed: 0"], inplace=True)


"""Webpage Content"""
st.markdown("""**Display Target Distribution and Visualization of Selected Features**""")

# Visualization (Target)
tmp = visual_data.groupby(['TARGET'])['EVENT_ID'].sum().reset_index()
tmp['Per'] = tmp['EVENT_ID'] / visual_data['EVENT_ID'].sum()
fig, ax = plt.subplots(figsize=(5, 3))
ax = sns.barplot(x=tmp['TARGET'], y=tmp['Per'], palette=['blue', 'red'], alpha=0.6)
ax.set_xlabel('Target Distribution', size=7)
ax.set_ylabel('Percent', size=7)
ax.tick_params(labelsize=6)
st.pyplot(fig)

# Visualization (Features)
disp_filter = st.sidebar.radio("", options=["Target is TRUE", "Target is FALSE", "All", "Comparison"])
sel_visual = multiselect_dropdown("Visualization of Selected Variables",
                                  options=spec_colnames['Feature'].values, default=[])

vis_c1, vis_c2 = st.beta_columns(2)
visual_but = vis_c1.button("Display")
visual_but_all = vis_c2.button("All Display")

if disp_filter == "All" and visual_but:
    if sel_visual == []:
        st.warning("You need to select features to display first.")

    index = 0
    for i in spec_colnames[spec_colnames['Feature'].isin(sel_visual)]['Column Name'].values:
        index += 1

        # Data Process
        tmp_cut = visual_data.groupby(i)["EVENT_ID"].sum().reset_index()
        tmp_cut["Per"] = tmp_cut["EVENT_ID"] / tmp_cut["EVENT_ID"].sum()
        tmp_cut.sort_values(by=[i], ascending=True, inplace=True)
        tmp_cut['cumsum'] = tmp_cut["Per"].cumsum()

        figdata, add_label = visualization_dataproc(i, tmp_cut)

        # Display
        fig = visualization_plot(figdata, i, add_label, spec_colnames)

        if divmod(index, 2)[1] == 1:
            vis_c1.pyplot(fig)
        else:
            vis_c2.pyplot(fig)

elif disp_filter == "All" and visual_but_all:
    index = 0
    for i in retrain_varlist_short:
        index += 1

        # Data Process
        tmp_cut = visual_data.groupby(i)["EVENT_ID"].sum().reset_index()
        tmp_cut["Per"] = tmp_cut["EVENT_ID"] / tmp_cut["EVENT_ID"].sum()
        tmp_cut.sort_values(by=[i], ascending=True, inplace=True)
        tmp_cut['cumsum'] = tmp_cut["Per"].cumsum()

        figdata, add_label = visualization_dataproc(i, tmp_cut)

        # Display
        fig = visualization_plot(figdata, i, add_label, spec_colnames)

        if divmod(index, 2)[1] == 1:
            vis_c1.pyplot(fig)
        else:
            vis_c2.pyplot(fig)

elif disp_filter == "Target is TRUE" and visual_but_all:
    index = 0
    for i in retrain_varlist_short:
        index += 1

        # Data Process
        tmp_cut = visual_data[visual_data['TARGET']==True].groupby(i)["EVENT_ID"].sum().reset_index()
        tmp_cut["Per"] = tmp_cut["EVENT_ID"] / tmp_cut["EVENT_ID"].sum()
        tmp_cut.sort_values(by=[i], ascending=True, inplace=True)
        tmp_cut['cumsum'] = tmp_cut["Per"].cumsum()

        figdata, add_label = visualization_dataproc(i, tmp_cut)

        # Display
        fig = visualization_plot(figdata, i, add_label, spec_colnames, plot_color='red',
                                 ylabel='Percent in Target = TRUE')

        if divmod(index, 2)[1] == 1:
            vis_c1.pyplot(fig)
        else:
            vis_c2.pyplot(fig)

elif disp_filter == "Target is TRUE" and visual_but:
    if sel_visual == []:
        st.warning("You need to select features to display first.")

    index = 0
    for i in spec_colnames[spec_colnames['Feature'].isin(sel_visual)]['Column Name'].values:
        index += 1

        # Data Process
        tmp_cut = visual_data[visual_data['TARGET']==True].groupby(i)["EVENT_ID"].sum().reset_index()
        tmp_cut["Per"] = tmp_cut["EVENT_ID"] / tmp_cut["EVENT_ID"].sum()
        tmp_cut.sort_values(by=[i], ascending=True, inplace=True)
        tmp_cut['cumsum'] = tmp_cut["Per"].cumsum()

        figdata, add_label = visualization_dataproc(i, tmp_cut)

        # Display
        fig = visualization_plot(figdata, i, add_label, spec_colnames, plot_color='red',
                                 ylabel='Percent in Target = TRUE')

        if divmod(index, 2)[1] == 1:
            vis_c1.pyplot(fig)
        else:
            vis_c2.pyplot(fig)

elif disp_filter == "Target is FALSE" and visual_but_all:
    index = 0
    for i in retrain_varlist_short:
        index += 1

        # Data Process
        tmp_cut = visual_data[visual_data['TARGET']==False].groupby(i)["EVENT_ID"].sum().reset_index()
        tmp_cut["Per"] = tmp_cut["EVENT_ID"] / tmp_cut["EVENT_ID"].sum()
        tmp_cut.sort_values(by=[i], ascending=True, inplace=True)
        tmp_cut['cumsum'] = tmp_cut["Per"].cumsum()

        figdata, add_label = visualization_dataproc(i, tmp_cut)

        # Display
        fig = visualization_plot(figdata, i, add_label, spec_colnames, plot_color='blue',
                                 ylabel='Percent in Target = FALSE')

        if divmod(index, 2)[1] == 1:
            vis_c1.pyplot(fig)
        else:
            vis_c2.pyplot(fig)

elif disp_filter == "Target is FALSE" and visual_but:
    if sel_visual == []:
        st.warning("You need to select features to display first.")

    index = 0
    for i in spec_colnames[spec_colnames['Feature'].isin(sel_visual)]['Column Name'].values:
        index += 1

        # Data Process
        tmp_cut = visual_data[visual_data['TARGET']==False].groupby(i)["EVENT_ID"].sum().reset_index()
        tmp_cut["Per"] = tmp_cut["EVENT_ID"] / tmp_cut["EVENT_ID"].sum()
        tmp_cut.sort_values(by=[i], ascending=True, inplace=True)
        tmp_cut['cumsum'] = tmp_cut["Per"].cumsum()

        figdata, add_label = visualization_dataproc(i, tmp_cut)

        # Display
        fig = visualization_plot(figdata, i, add_label, spec_colnames, plot_color='blue',
                                 ylabel='Percent in Target = FALSE')

        if divmod(index, 2)[1] == 1:
            vis_c1.pyplot(fig)
        else:
            vis_c2.pyplot(fig)

elif disp_filter == "Comparison" and visual_but_all:
    for i in retrain_varlist_short:
        # Data Process
        tmp_cut_true = visual_data[visual_data['TARGET']==True].groupby(i)["EVENT_ID"].sum().reset_index()
        tmp_cut_true["Per"] = tmp_cut_true["EVENT_ID"] / tmp_cut_true["EVENT_ID"].sum()
        tmp_cut_true.sort_values(by=[i], ascending=True, inplace=True)
        tmp_cut_true['cumsum'] = tmp_cut_true["Per"].cumsum()

        tmp_cut_false = visual_data[visual_data['TARGET'] == False].groupby(i)["EVENT_ID"].sum().reset_index()
        tmp_cut_false["Per"] = tmp_cut_false["EVENT_ID"] / tmp_cut_false["EVENT_ID"].sum()
        tmp_cut_false.sort_values(by=[i], ascending=True, inplace=True)
        tmp_cut_false['cumsum'] = tmp_cut_false["Per"].cumsum()

        figdata_true, add_label_true = visualization_dataproc(i, tmp_cut_true)
        figdata_false, add_label_false = visualization_dataproc(i, tmp_cut_false)

        # Display
        fig_true = visualization_plot(figdata_true, i, add_label_true, spec_colnames, plot_color='red',
                                      ylabel='Percent in Target = TRUE',
                                      ylim=max(visualization_ylim(figdata_true, i),
                                               visualization_ylim(figdata_false, i)))
        fig_false = visualization_plot(figdata_false, i, add_label_false, spec_colnames, plot_color='blue',
                                       ylabel='Percent in Target = FALSE',
                                       ylim=max(visualization_ylim(figdata_true, i),
                                                visualization_ylim(figdata_false, i)))

        vis_c2.pyplot(fig_true)
        vis_c1.pyplot(fig_false)
#
elif disp_filter == "Comparison" and visual_but:
    if sel_visual == []:
        st.warning("You need to select features to display first.")

    for i in spec_colnames[spec_colnames['Feature'].isin(sel_visual)]['Column Name'].values:
        # Data Process
        tmp_cut_true = visual_data[visual_data['TARGET']==True].groupby(i)["EVENT_ID"].sum().reset_index()
        tmp_cut_true["Per"] = tmp_cut_true["EVENT_ID"] / tmp_cut_true["EVENT_ID"].sum()
        tmp_cut_true.sort_values(by=[i], ascending=True, inplace=True)
        tmp_cut_true['cumsum'] = tmp_cut_true["Per"].cumsum()

        tmp_cut_false = visual_data[visual_data['TARGET'] == False].groupby(i)["EVENT_ID"].sum().reset_index()
        tmp_cut_false["Per"] = tmp_cut_false["EVENT_ID"] / tmp_cut_false["EVENT_ID"].sum()
        tmp_cut_false.sort_values(by=[i], ascending=True, inplace=True)
        tmp_cut_false['cumsum'] = tmp_cut_false["Per"].cumsum()

        figdata_true, add_label_true = visualization_dataproc(i, tmp_cut_true)
        figdata_false, add_label_false = visualization_dataproc(i, tmp_cut_false)

        # Display
        tmp_ylim = max(visualization_ylim(figdata_true, i), visualization_ylim(figdata_false, i))
        fig_true = visualization_plot(figdata_true, i, add_label_true, spec_colnames, plot_color='red',
                                      ylabel='Percent in Target = TRUE',
                                      ylim=tmp_ylim)
        fig_false = visualization_plot(figdata_false, i, add_label_false, spec_colnames, plot_color='blue',
                                       ylabel='Percent in Target = FALSE',
                                       ylim=tmp_ylim)

        vis_c2.pyplot(fig_true)
        vis_c1.pyplot(fig_false)

