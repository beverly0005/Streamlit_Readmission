# Utilization functions

# Import libraries
import hashlib
import base64
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Hash input
def generate_hashes(input):
    return hashlib.sha256(str.encode(input)).hexdigest()


# Background Image
def get_base64_of_bin_file(file):
    with open(file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(main_bg):
    page_bg_img = '''
        <style>
        body {
        background-image: url("data:image/jpg;base64, %s");
        background-size: cover;
        }
        </style> 
    ''' % get_base64_of_bin_file(main_bg)
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Limit length of dropdown list
def limit_dropdownlist(height):
    st.write("""
            <style type="text/css" media="screen">
            div[role="listbox"] ul {
            height: %s;} </style>
            """ % height,
             unsafe_allow_html=True,
             )


# Multi-selection from dropdown list
def multiselect_dropdown(label, options, default, format_func=str):
    """multiselect extension that enables default to be a subset list of the list of objects
     - not a list of strings

     Assumes that options have unique format_func representations

     cf. https://github.com/streamlit/streamlit/issues/352
     """
    options_ = {format_func(option): option for option in options}
    default_ = [format_func(option) for option in default]
    selections = st.multiselect(
        label, options=list(options_.keys()), default=default_, format_func=format_func
    )
    return [options_[format_func(selection)] for selection in selections]


# Process Data for Visualization
def visualization_dataproc(i, tmp_cut):
    """Process Data for Better Visualization of Original Data
       Output will be data directly used for plots and labels added to X label

        -i: column name
        -tmp_cut: data used for visualization
    """

    if i in ["P0_6ED", "LOS_CURRENT", "DRP_COUNTS_P0_12", "CD_COUNTS_P0_24", "CDMS_HYPERTENSION",
             "CDMS_CHD", "NUM_CCI", "GRPC_NUM_ATCMED_P0_12",
             "NUM_DRUGGRP_P0_12", "NUM_SPEC_P12_24", "NUM_DISCHARGE_SPEC_P0_12"]:
        # Cut off at 95 percentile for these features
        tmp_data = tmp_cut[tmp_cut['cumsum'] <= 0.95]

        # Avoid cutting too much
        if tmp_data[i].nunique() >= 2:
            figdata = tmp_data.copy()
            add_label = ' (Cut off at 95 Pct)'
        else:
            figdata = tmp_cut.copy()
            add_label = ''

    elif i in ["CT_RD90_DRP_LMD_P0_24", "CCI_PULMONARY", "CCI_DIABETES_NO_LONG", "CDMS_ASTHMA", "CDMS_HF",
               "CDMS_COPD", "GENSUR_NUM_VISIT_P0_12", "GRPN_NUM_ATCMED_P0_12", "GRPR_NUM_ATCMED_P0_12"]:
        # Simplify too many categories
        tmp_cut.loc[tmp_cut[i] > 1, i] = ">1"
        tmp_cut[i].astype(str)
        figdata = tmp_cut.groupby([i])["Per"].sum().reset_index()
        add_label = ''

    elif i in ["P0_12SOC"]:
        # Simplify too many categories
        tmp_cut.loc[tmp_cut[i] > 8, i] = ">8"
        tmp_cut[i].astype(str)
        figdata = tmp_cut.groupby([i])["Per"].sum().reset_index()
        add_label = ''

    elif i in ["DURATION_AVG_P0_12", "AGE"]:
        # Remove missing
        tmp_cut = tmp_cut[tmp_cut[i] > -999]
        tmp_cut[i] = np.floor(tmp_cut[i] / 20) * 20
        tmp_cut.loc[tmp_cut[i] >= 100, i] = '>99'
        figdata = tmp_cut.groupby([i])["Per"].sum().reset_index()
        add_label = " (Rm Miss.)"

    elif i in ["P0_12_AVG_33037-3", "P0_12_AVG_14631-6"]:
        # Remove missing and cut of at 95 percentile
        tmp_cut = tmp_cut[(tmp_cut[i] > -999) & (tmp_cut["cumsum"] <= 0.95)]
        tmp_cut[i] = np.floor(tmp_cut[i] / 1) * 1
        figdata = tmp_cut.groupby([i])["Per"].sum().reset_index()
        add_label = " (Cut off at 95 Pct, Rm Miss.)"

    elif i in ["LOS_LASTTIME"]:
        # Cut off at 95 percentile and remove missing
        figdata = tmp_cut[(tmp_cut['cumsum'] <= 0.95) & (tmp_cut[i] != -999)]
        add_label = ' (Cut off at 95 Pct, Rm Miss.)'

    elif i in ["DAYS_SINCE_LAST_INP"]:
        # Group into 100 bind
        tmp_cut.loc[tmp_cut[i] < 730, i] = np.floor(tmp_cut.loc[tmp_cut[i] < 730, i] / 100) * 100
        tmp_cut.loc[tmp_cut[i] == 730, i] = '>729'
        tmp_cut[i] = tmp_cut[i].apply(lambda x: str(x).replace(".0", ""))
        figdata = tmp_cut.groupby([i])["Per"].sum().reset_index()
        add_label = ' (Xtick shows Lower Bound)'

    elif i in ["GASTRO_LASTTIME", "GERMED_LASTTIME", "UROLOGY_LASTTIME"]:
        # Remove missing
        tmp_cut = tmp_cut[tmp_cut[i] > -999]
        figdata = tmp_cut.groupby([i])["Per"].sum().reset_index()
        add_label = ' (Rm Miss.)'
    return figdata, add_label


# Plot graphs: find proper ylim max
def visualization_ylim(figdata, i):
    """Find Ylim Maximum for Plot

        -i: column name
        -figdata: data directly used for plot
    """
    if figdata["Per"].max() > 0.8:
        ylim = 1
    elif figdata["Per"].max() > 0.1:
        ylim = np.ceil(figdata["Per"].max() / 0.1) * 0.1 + 0.1
    else:
        ylim = np.ceil(figdata["Per"].max() / 0.01) * 0.01 + 0.01
    return ylim


def visualization_plot(figdata, i, add_label, spec_colnames, plot_color='grey', ylabel='Percent', ylim=-999):
    """Plot Bar Graph or Line Graph

        -i: column name
        -tmp_cut: data directly used for plot
        -add_label: label added to X label
        -spec_colnames: interpretation on short column name to display
    """

    if figdata[i].nunique() <= 10:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(figdata.index, figdata['Per'], alpha=0.6, color=plot_color)
        plt.xticks(figdata.index, figdata[i].values)
        if ylim == -999:
            ax.set_ylim(0, visualization_ylim(figdata, i))
        else:
            ax.set_ylim(0, ylim)
        ax.set_xlabel(spec_colnames[spec_colnames['Column Name'] == i]['Feature'].values[0] + add_label)
        ax.set_ylabel(ylabel)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax = sns.lineplot(x=figdata[i], y=figdata['Per'], color=plot_color)
        if ylim == -999:
            ax.set_ylim(0, visualization_ylim(figdata, i))
        else:
            ax.set_ylim(0, ylim)
        ax.set_xlabel(spec_colnames[spec_colnames['Column Name'] == i]['Feature'].values[0] + add_label)
        ax.set_ylabel(ylabel)

    return fig
