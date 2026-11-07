
app_code = r'''
import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification Model Evaluation")
st.write(
    "A complete dashboard for dataset exploration, "
    "five-model comparison and detailed model evaluation."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_FILE = os.path.join(BASE_DIR, "test_data.csv")


# --------------------------------------------------
# Load saved models and metadata
# --------------------------------------------------

@st.cache_resource
def load_metadata():
    return joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))


@st.cache_resource
def load_model(path):
    return joblib.load(path)


try:
    metadata = load_metadata()
except Exception as error:
    st.error(f"Unable to load model metadata: {error}")
    st.stop()


# --------------------------------------------------
# Dataset selection
# --------------------------------------------------

st.sidebar.header("Dataset Selection")

uploaded_file = st.sidebar.file_uploader(
    "Upload another labelled CSV (optional)",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded dataset loaded.")
    else:
        if not os.path.exists(DEFAULT_FILE):
            st.error("Default test_data.csv was not found.")
            st.stop()

        data = pd.read_csv(DEFAULT_FILE)
        st.sidebar.info("Using default test_data.csv.")

except Exception as error:
    st.error(f"Unable to read the dataset: {error}")
    st.stop()


# --------------------------------------------------
# Validate dataset
# --------------------------------------------------

target = metadata["target_column"]
features = metadata["feature_columns"]
required_columns = features + [target]

missing_columns = [
    col for col in required_columns if col not in data.columns
]

if missing_columns:
    st.error(
        "The dataset is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

if data.empty:
    st.error("The selected dataset is empty.")
    st.stop()

X = data[features].apply(pd.to_numeric, errors="coerce")
y_true = data[target]

if y_true.isna().any():
    st.error("The target column contains missing values.")
    st.stop()


# --------------------------------------------------
# Load all five models and calculate their metrics
# --------------------------------------------------

@st.cache_data
def evaluate_all_models(data_features, data_target, model_files, model_dir):
    results = {}
    predictions = {}

    for name, filename in model_files.items():

        model_path = os.path.join(model_dir, filename)
        model = load_model(model_path)

        predicted = model.predict(data_features)
        predictions[name] = predicted

        metrics = {
            "Accuracy": accuracy_score(data_target, predicted),
            "Precision": precision_score(
                data_target, predicted,
                average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                data_target, predicted,
                average="weighted", zero_division=0
            ),
            "F1 Score": f1_score(
                data_target, predicted,
                average="weighted", zero_division=0
            ),
            "MCC": matthews_corrcoef(data_target, predicted)
        }

        try:
            probabilities = model.predict_proba(data_features)

            if len(model.classes_) == 2:
                metrics["AUC"] = roc_auc_score(
                    data_target, probabilities[:, 1]
                )
            else:
                metrics["AUC"] = roc_auc_score(
                    data_target,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                    labels=model.classes_
                )

        except Exception:
            metrics["AUC"] = np.nan

        results[name] = metrics

    return results, predictions


try:
    model_results, all_predictions = evaluate_all_models(
        X,
        y_true,
        metadata["model_files"],
        MODEL_DIR
    )

except Exception as error:
    st.error(f"Model evaluation failed: {error}")
    st.stop()


results_df = pd.DataFrame(model_results).T
comparison_df = results_df.copy()

model_names = list(metadata["model_files"].keys())


# --------------------------------------------------
# Dashboard navigation
# --------------------------------------------------

overview_tab, comparison_tab, detail_tab, download_tab = st.tabs(
    [
        "Dataset Overview",
        "Five-Model Comparison",
        "Detailed Model Evaluation",
        "Download Reports"
    ]
)


# ==================================================
# TAB 1: DATASET OVERVIEW
# ==================================================

with overview_tab:

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(data):,}")
    col2.metric("Total Features", len(features))
    col3.metric("Target Classes", data[target].nunique())
    col4.metric(
        "Missing Values",
        int(data[required_columns].isna().sum().sum())
    )

    st.subheader("Dataset Information")

    st.write("**Target Column:**", target)
    st.write("**Feature Columns:**", ", ".join(features))

    st.subheader("Sample Dataset")

    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("Dataset Statistics")

    st.dataframe(
        data[features].describe().T,
        use_container_width=True
    )

    st.subheader("Target Class Distribution")

    target_counts = data[target].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    target_counts.plot(kind="bar", ax=ax)

    ax.set_title("Distribution of Target Classes")
    ax.set_xlabel("Target Class")
    ax.set_ylabel("Number of Records")
    ax.tick_params(axis="x", rotation=0)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Missing Values by Column")

    missing_values = data[required_columns].isna().sum()
    missing_df = missing_values.reset_index()
    missing_df.columns = ["Column", "Missing Values"]

    st.dataframe(missing_df, use_container_width=True)


# ==================================================
# TAB 2: FIVE-MODEL COMPARISON
# ==================================================

with comparison_tab:

    st.header("Performance Comparison of All Five Models")

    st.write(
        "The following table compares all saved classification "
        "models using the same selected dataset."
    )

    display_df = comparison_df.copy().round(4)

    display_df.index.name = "Classification Model"

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.subheader("Best Model for Each Metric")

    best_columns = st.columns(3)

    metric_names = [
        "Accuracy", "AUC", "Precision",
        "Recall", "F1 Score", "MCC"
    ]

    for index, metric in enumerate(metric_names):

        if comparison_df[metric].notna().any():

            best_model = comparison_df[metric].idxmax()
            best_value = comparison_df.loc[best_model, metric]

            best_columns[index % 3].metric(
                f"Best {metric}",
                f"{best_value:.4f}",
                best_model
            )

    st.subheader("Overall Performance Comparison")

    selected_metrics = st.multiselect(
        "Select metrics to compare",
        metric_names,
        default=metric_names
    )

    if selected_metrics:

        fig, ax = plt.subplots(figsize=(12, 5))

        comparison_df[selected_metrics].plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Comparison of Five Classification Models")
        ax.set_xlabel("Classification Models")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Evaluation Metrics")
        ax.tick_params(axis="x", rotation=20)

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Individual Metric Charts")

    for metric in metric_names:

        fig, ax = plt.subplots(figsize=(9, 3))

        comparison_df[metric].dropna().sort_values().plot(
            kind="barh",
            ax=ax
        )

        ax.set_title(f"{metric} Comparison")
        ax.set_xlabel("Score")
        ax.set_ylabel("Model")
        ax.set_xlim(0, 1.05)

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Model Ranking by F1 Score")

    ranking = comparison_df["F1 Score"].sort_values(
        ascending=False
    ).reset_index()

    ranking.columns = ["Model", "F1 Score"]
    ranking.index = ranking.index + 1

    st.dataframe(ranking, use_container_width=True)


# ==================================================
# TAB 3: DETAILED MODEL EVALUATION
# ==================================================

with detail_tab:

    st.header("Detailed Model Evaluation")

    selected_model = st.selectbox(
        "Choose a model for detailed evaluation",
        model_names
    )

    selected_path = os.path.join(
        MODEL_DIR,
        metadata["model_files"][selected_model]
    )

    selected_model_object = load_model(selected_path)
    selected_predictions = all_predictions[selected_model]

    st.subheader(f"{selected_model} - Performance Metrics")

    selected_metrics = results_df.loc[selected_model]

    metric_columns = st.columns(3)

    for index, (metric, value) in enumerate(selected_metrics.items()):

        display_value = (
            "Not available"
            if pd.isna(value)
            else f"{value:.4f}"
        )

        metric_columns[index % 3].metric(
            metric,
            display_value
        )

    st.subheader("Confusion Matrix")

    labels = selected_model_object.classes_

    matrix = confusion_matrix(
        y_true,
        selected_predictions,
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(matrix, interpolation="nearest")
    ax.set_title(f"Confusion Matrix - {selected_model}")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center"
            )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        selected_predictions,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    st.subheader("ROC Curve")

    try:
        probabilities = selected_model_object.predict_proba(X)

        if len(selected_model_object.classes_) == 2:

            positive_class = selected_model_object.classes_[1]

            binary_target = (
                np.asarray(y_true) == positive_class
            ).astype(int)

            fpr, tpr, thresholds = roc_curve(
                binary_target,
                probabilities[:, 1]
            )

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.plot(
                fpr, tpr,
                label=f"AUC = {selected_metrics['AUC']:.4f}"
            )

            ax.plot([0, 1], [0, 1], linestyle="--")

            ax.set_title(f"ROC Curve - {selected_model}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info(
                "The ROC curve shown here is configured for binary "
                "classification. AUC is calculated for multiclass models "
                "using one-vs-rest."
            )

    except Exception:
        st.info("ROC curve is not available for this model.")

    st.subheader("Actual vs Predicted Values")

    prediction_df = pd.DataFrame({
        "Actual": y_true.values,
        "Predicted": selected_predictions
    })

    st.dataframe(prediction_df.head(20), use_container_width=True)


# ==================================================
# TAB 4: DOWNLOAD REPORTS
# ==================================================

with download_tab:

    st.header("Download Evaluation Reports")

    st.write(
        "Download the model comparison report and prediction "
        "results for your assignment documentation."
    )

    # Create complete text report
    report_parts = []

    report_parts.append(
        "MACHINE LEARNING CLASSIFICATION MODEL EVALUATION REPORT"
    )

    report_parts.append("=" * 65)

    report_parts.append(f"Dataset records: {len(data)}")
    report_parts.append(f"Number of features: {len(features)}")
    report_parts.append(f"Target column: {target}")
    report_parts.append(
        f"Number of target classes: {data[target].nunique()}"
    )

    report_parts.append("\nFIVE-MODEL PERFORMANCE COMPARISON")
    report_parts.append("=" * 65)

    report_parts.append(
        comparison_df.round(4).to_string()
    )

    for name in model_names:

        report_parts.append(
            f"\n\nDETAILED EVALUATION: {name}"
        )

        report_parts.append("-" * 65)

        report_parts.append(
            pd.Series(model_results[name]).round(4).to_string()
        )

        report_parts.append("\nClassification Report:")

        report_parts.append(
            classification_report(
                y_true,
                all_predictions[name],
                zero_division=0
            )
        )

        model_object = load_model(
            os.path.join(
                MODEL_DIR,
                metadata["model_files"][name]
            )
        )

        report_parts.append("\nConfusion Matrix:")

        report_parts.append(
            str(
                confusion_matrix(
                    y_true,
                    all_predictions[name],
                    labels=model_object.classes_
                )
            )
        )

    full_report = "\n".join(report_parts)

    st.download_button(
        label="Download Complete Evaluation Report (TXT)",
        data=full_report,
        file_name="ML_Classification_Evaluation_Report.txt",
        mime="text/plain"
    )

    # Download model comparison as CSV
    comparison_csv = comparison_df.to_csv(index=True)

    st.download_button(
        label="Download Five-Model Comparison (CSV)",
        data=comparison_csv,
        file_name="Five_Model_Comparison.csv",
        mime="text/csv"
    )

    # Download predictions from all models
    prediction_output = pd.DataFrame({
        "Actual": y_true.values
    })

    for name in model_names:
        prediction_output[name] = all_predictions[name]

    prediction_csv = prediction_output.to_csv(index=False)

    st.download_button(
        label="Download All Model Predictions (CSV)",
        data=prediction_csv,
        file_name="All_Model_Predictions.csv",
        mime="text/csv"
    )

    # Download selected model predictions
    selected_for_download = st.selectbox(
        "Select model for individual prediction download",
        model_names,
        key="download_model"
    )

    individual_predictions = pd.DataFrame({
        "Actual": y_true.values,
        "Predicted": all_predictions[selected_for_download]
    })

    st.download_button(
        label="Download Selected Model Predictions",
        data=individual_predictions.to_csv(index=False),
        file_name=f"{selected_for_download.replace(' ', '_')}_Predictions.csv",
        mime="text/csv"
    )

    st.success("Your evaluation reports are ready to download.")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Assignment 2 | "
    "Classification Models and Performance Evaluation"
)
'''

# Save the updated Streamlit application
with open(
    os.path.join(PROJECT_DIR, "app.py"),
    "w",
    encoding="utf-8"
) as file:
    file.write(app_code)

print("Updated app.py created successfully!")
print("Features added:")
print("1. Dataset overview")
print("2. Five-model performance comparison")
print("3. Performance charts")
print("4. Detailed model evaluation")
print("5. Confusion matrices and ROC curves")
print("6. Downloadable evaluation reports")
print("7. Automatic default test_data.csv loading")
