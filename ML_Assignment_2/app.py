
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Classification Model Evaluation", layout="wide")
st.title("Machine Learning Classification Model Evaluation")
st.write("Upload a labelled test CSV, select a model and view its evaluation results.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

@st.cache_resource
def load_metadata():
    return joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    metadata = load_metadata()
except Exception as error:
    st.error(f"Could not load model metadata: {error}")
    st.stop()

uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])
model_name = st.selectbox("Select classification model", list(metadata["model_files"].keys()))

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        target = metadata["target_column"]
        features = metadata["feature_columns"]
        required = features + [target]
        missing = [column for column in required if column not in data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            st.stop()
        if data.empty:
            st.error("The uploaded CSV has no rows.")
            st.stop()

        X = data[features].apply(pd.to_numeric, errors="coerce")
        y_true = data[target]
        if y_true.isna().any():
            st.error("The target column contains missing values.")
            st.stop()

        model_path = os.path.join(MODEL_DIR, metadata["model_files"][model_name])
        model = load_model(model_path)
        y_pred = model.predict(X)

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred)
        }
        try:
            probabilities = model.predict_proba(X)
            if len(model.classes_) == 2:
                metrics["AUC"] = roc_auc_score(y_true, probabilities[:, 1])
            else:
                metrics["AUC"] = roc_auc_score(
                    y_true, probabilities, multi_class="ovr",
                    average="weighted", labels=model.classes_
                )
        except Exception:
            metrics["AUC"] = np.nan

        st.subheader("Evaluation Metrics")
        columns = st.columns(3)
        for index, (metric, value) in enumerate(metrics.items()):
            display_value = "Not available" if pd.isna(value) else f"{value:.4f}"
            columns[index % 3].metric(metric, display_value)

        st.subheader("Confusion Matrix")
        labels = model.classes_
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels))

        st.subheader("Classification Report")
        report = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())

        st.success(f"Evaluation completed using {model_name}. The saved model was loaded; no retraining was performed.")
    except Exception as error:
        st.error(f"Unable to evaluate the uploaded file: {error}")
else:
    st.info("Upload test_data.csv to display evaluation results.")
