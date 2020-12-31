"""Web App Demo for KTPH 90-Day DRP Readmission for Patient with Chronic Disease

Thanks to JCharisTech & J-Secur1ty: https://www.youtube.com/watch?v=DBBpLrhKzC8, which I refer to develop some
functionality of the application.
The script aims to be a demo or even a template to develop an application for traditional predictive model.
More functionality could be added and modified according to the specific needs.

Developer: Beverly
Date: December 2020
"""


# Libraries
import streamlit as st
import os

# Basic libraries
import pandas as pd


# User Parameters
MAIN_BG = '/Users/zhiyuwang/Desktop/Streamlit_tutorial/Figure/olga-thelavart-vS3idIiYxX0-unsplash.jpg'
WORKDIR = '/Users/zhiyuwang/Desktop/Streamlit_tutorial/Code/'

# Main
def main():
    # Header of App
    st.title("Predict 90-day Drug-Related Readmission for Patients with Chronic Diseases")

    # Add background image
    # set_background(MAIN_BG)

    # Menu
    st.sidebar.text("Streamlit version: " + str(st.__version__))
    menu = ["About", "Feature List (Dev)", "Re-train (Dev)",
            "Visualization", "Individual Prediction", "Bulky Prediction"]
    choice = st.sidebar.selectbox("Section:", menu)

    if choice == "About":
        exec(open(WORKDIR + 'webpage_about_v1.py').read())

    elif choice == "Feature List (Dev)":  # Only for development stage
        exec(open(WORKDIR + 'webpage_feature_v1.py').read())

    elif choice == "Re-train (Dev)":  # Only for development stage

        # Re-train Model
        if st.sidebar.checkbox("Re-train Model"):
            st.header("Re-train Model")
            exec(open(WORKDIR + 'webpage_retrain_v1.py').read())

        # Statistics on original training data
        exec(open(WORKDIR + 'webpage_trainstat_v1.py').read())

    elif choice == "Visualization":
        exec(open(WORKDIR + 'webpage_visual_v1.py').read())

    elif choice == "Individual Prediction":
        exec(open(WORKDIR + 'webpage_indpred_v1.py').read())

    else:
        exec(open(WORKDIR + 'webpage_bulkpred_v1.py').read())


if __name__ == '__main__':
    main()
