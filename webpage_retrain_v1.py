"""Webpage of Retrain (Dev)
This page includes the retraining of the model with new data.
Please edit if necessary
"""

# Import Libraries
import streamlit as st
import pickle
import base64


# User Parameters
retrain_varlist = [
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
        "Age in Year",
        # TARGET
        "Target: 90-Day Readmission Due to DRP"
    ]
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
    "AGE",
    # Target
    "TARGET"
]


# Self-defined functions
def print_metric(model, X, y):
    pred_y = model.predict_proba(X)[:, 1]

    prec, rec, prob = precision_recall_curve(y, pred_y)
    tmp = pd.DataFrame({'prec': prec[:-1], 'rec': rec[:-1], 'prob': prob})
    tmp['f1'] = 2 * tmp['prec'] * tmp['rec'] / (tmp['prec'] + tmp['rec'])
    f1_star = tmp[tmp['f1'] == tmp['f1'].max()]['f1'].values[0]
    prec_star = tmp[tmp['f1'] == tmp['f1'].max()]['prec'].values[0]
    rec_star = tmp[tmp['f1'] == tmp['f1'].max()]['rec'].values[0]

    accuracy = []
    for i in prob:
        accuracy.append(accuracy_score(y, [1 if m > i else 0 for m in pred_y]))
    accuracy_star = np.array(accuracy).max()

    return {'AUC': roc_auc_score(y, pred_y), 'F1': f1_star, 'Precision': prec_star,
            'Recall': rec_star, 'Accuracy': accuracy_star}


def split_data(data):
    TARGET_VAR = 'TARGET'

    # Random Selection
    samplelist = pd.DataFrame(data.index).sample(frac=1, random_state=1000, replace=False)
    samplelist = samplelist[0].values

    # Split train, dev, test
    X_train = data[data.index.isin(samplelist[:round(len(samplelist)*0.80)])].drop(columns=[TARGET_VAR])
    X_dev = data[data.index.isin(samplelist[round(len(samplelist)*0.80):])].drop(columns=[TARGET_VAR])
    y_train = data[data.index.isin(samplelist[:round(len(samplelist)*0.80)])][TARGET_VAR]
    y_dev = data[data.index.isin(samplelist[round(len(samplelist)*0.80):])][TARGET_VAR]

    return X_train, y_train, X_dev, y_dev


def fit_model(X_train, y_train, X_dev, y_dev):
    model = xgb.XGBClassifier(colsample_bytree=1, min_child_weight=1, learning_rate=0.05, metrics='auc',
                              n_estimators=1000, random_state=1234, colsample_bylevel=0.5, max_depth=5,
                              min_split_loss=0.01, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                              subsample=1)
    model.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], eval_metric='auc', early_stopping_rounds=200,
              verbose=False)
    return model


""""Webpage Content"""

# Step 1: Ensure columns in order
st.markdown("""* **Step 1:** Ensure **31 Columns** of Re-trained Data (Incl. Target Variable) in Specified Order""")
disporder_but = st.button("Display Specified Column Order")
if disporder_but:
    st.dataframe(pd.DataFrame({'Variable': retrain_varlist}))

# Step 2: Upload Data
st.markdown("""* **Step 2:** Upload Re-trained Data in **CSV** Format""")
retrain_datafile = st.file_uploader("", type=['CSV'])
if retrain_datafile is not None:
    retrain_data = pd.read_csv(retrain_datafile)
    retrain_data.drop(columns=["Unnamed: 0"], inplace=True)
    retrain_data.columns = retrain_varlist_short

# Step 3: Re-train model
st.markdown("""* **Step 3:** Re-train Model on Uploaded Data""")
if retrain_datafile is not None:
    st.markdown("""Re-train XGBoost model on the uploaded data. Split the data into 80% training set and 20% validation
    set. **The performance of the re-trained model on the whole uploaded data is as below:**""")
    X_train, y_train, X_dev, y_dev = split_data(retrain_data)
    model = fit_model(X_train, y_train, X_dev, y_dev)
    retrain_metric = print_metric(model, retrain_data.drop(columns=['TARGET']), retrain_data['TARGET'])
    st.dataframe(pd.DataFrame({
        "No. Records": [retrain_data.shape[0]],
        "AUC": [retrain_metric["AUC"]],
        "Max F1": [retrain_metric["F1"]],
        "Max Precision": [retrain_metric["Precision"]],
        "Max Recall": [retrain_metric["Recall"]],
        "Max Accuracy": [retrain_metric["Accuracy"]]
    }))

    # Download Model
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}">Download Trained Model</a> (Right-click and save as <model_name>.pkl)'
    st.markdown(href, unsafe_allow_html=True)

