
import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Machine Learning Model Dashboard",
    page_icon="🧬",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# --------------------------------------------------
# Load metadata
# --------------------------------------------------

metadata_path = os.path.join(MODEL_DIR, "metadata.joblib")

if not os.path.exists(metadata_path):
    st.error("Metadata file not found. Please check the model folder.")
    st.stop()

metadata = joblib.load(metadata_path)

model_files = metadata["model_files"]
model_names = list(model_files.keys())

features = metadata.get("features", metadata.get("feature_columns", []))
target_column = metadata.get(
    "target_column",
    metadata.get("target", "target")
)

# --------------------------------------------------
# Theme selector
# --------------------------------------------------

if "app_theme" not in st.session_state:
    st.session_state.app_theme = "Dark"

top_left, top_right = st.columns([5, 1])

with top_right:
    selected_theme = st.selectbox(
        "Appearance",
        ["Dark", "Light"],
        index=0 if st.session_state.app_theme == "Dark" else 1
    )

st.session_state.app_theme = selected_theme

is_dark = selected_theme == "Dark"

if is_dark:
    bg_colors = ["#07111f", "#101b35", "#17143b", "#082b3b"]
    text_color = "#eaf4ff"
    card_color = "rgba(255,255,255,0.09)"
    border_color = "rgba(255,255,255,0.17)"
    plot_bg = "#101b35"
    plot_fg = "#eaf4ff"
else:
    bg_colors = ["#eff6ff", "#ffffff", "#e0e7ff", "#f0fdfa"]
    text_color = "#172554"
    card_color = "rgba(255,255,255,0.85)"
    border_color = "rgba(30,64,175,0.18)"
    plot_bg = "#ffffff"
    plot_fg = "#172554"

st.markdown(
    f'''
    <style>
    .stApp {{
        background: linear-gradient(
            -45deg,
            {", ".join(bg_colors)}
        );
        background-size: 400% 400%;
        animation: gradientMove 18s ease infinite;
        color: {text_color};
    }}

    @keyframes gradientMove {{
        0% {{background-position: 0% 50%;}}
        50% {{background-position: 100% 50%;}}
        100% {{background-position: 0% 50%;}}
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stMetric"] {{
        background: {card_color};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 15px;
    }}

    [data-testid="stTabs"] button {{
        font-weight: 600;
    }}

    h1, h2, h3, h4, p, label {{
        color: {text_color};
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {border_color};
        border-radius: 10px;
    }}

    </style>
    ''',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Dashboard heading
# --------------------------------------------------

st.title("🧬 Machine Learning Classification Dashboard")

st.markdown(
    "Interactive analysis, model comparison and performance evaluation "
    "using five machine learning classification algorithms."
)

st.caption("Dataset: Breast Cancer Wisconsin (Diagnostic)")

st.divider()

# --------------------------------------------------
# Load saved models
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


loaded_models = {}

for model_name, model_file in model_files.items():
    model_path = os.path.join(MODEL_DIR, model_file)

    if os.path.exists(model_path):
        loaded_models[model_name] = load_model(model_path)

if not loaded_models:
    st.error("No trained model files were found.")
    st.stop()

# --------------------------------------------------
# Dataset selection
# --------------------------------------------------

st.subheader("📂 Dataset")

default_file = os.path.join(BASE_DIR, "test_data.csv")

uploaded_file = st.file_uploader(
    "Upload another labelled test CSV file (optional)",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        dataset_source = "Uploaded dataset"
    elif os.path.exists(default_file):
        data = pd.read_csv(default_file)
        dataset_source = "Default test_data.csv"
    else:
        st.warning(
            "Default test_data.csv was not found. "
            "Please upload a labelled CSV file."
        )
        st.stop()

except Exception as error:
    st.error(f"Unable to read the CSV file: {error}")
    st.stop()

if not features:
    features = [
        column for column in data.columns
        if column != target_column
    ]

missing_features = [col for col in features if col not in data.columns]

if missing_features:
    st.error(f"Missing feature columns: {missing_features}")
    st.stop()

if target_column not in data.columns:
    st.error(
        f"Target column '{target_column}' is missing. "
        "This dashboard requires labelled data for model evaluation."
    )
    st.stop()

if len(data) == 0:
    st.error("The selected dataset is empty.")
    st.stop()

data = data.copy()
data[features] = data[features].apply(
    pd.to_numeric,
    errors="coerce"
)

data = data.dropna(subset=[target_column])

if len(data) == 0:
    st.error("No valid labelled records are available.")
    st.stop()

X = data[features]
y = data[target_column]

st.caption(f"Dataset source: {dataset_source}")

# --------------------------------------------------
# Evaluate all models
# --------------------------------------------------

def evaluate_model(model, X, y):
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(
        y, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y, predictions, average="weighted", zero_division=0
    )
    mcc = matthews_corrcoef(y, predictions)

    auc = np.nan
    probabilities = None

    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)

            if len(np.unique(y)) == 2 and probabilities.shape[1] == 2:
                auc = roc_auc_score(y, probabilities[:, 1])

            elif len(np.unique(y)) > 2:
                auc = roc_auc_score(
                    y,
                    probabilities,
                    multi_class="ovr",
                    average="weighted"
                )

        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)

            if len(np.unique(y)) == 2:
                auc = roc_auc_score(y, scores)

    except Exception:
        auc = np.nan

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc,
        "MCC": mcc,
        "Predictions": predictions,
        "Probabilities": probabilities
    }


@st.cache_data
def evaluate_all_models(X, y, model_names, model_files, model_dir):
    results = {}

    for name in model_names:
        model_path = os.path.join(model_dir, model_files[name])

        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        results[name] = evaluate_model(model, X, y)

    return results


with st.spinner("Evaluating all five machine learning models..."):
    results = evaluate_all_models(
        X,
        y,
        list(loaded_models.keys()),
        model_files,
        MODEL_DIR
    )

if not results:
    st.error("Unable to evaluate the saved models.")
    st.stop()

# --------------------------------------------------
# Create comparison dataframe
# --------------------------------------------------

metric_names = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

comparison_data = []

for name, result in results.items():
    row = {"Model": name}

    for metric in metric_names:
        row[metric] = result[metric]

    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

# --------------------------------------------------
# Dashboard tabs
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "🏆 Model Comparison",
    "🔍 Detailed Evaluation",
    "📥 Download Reports"
])

# ==================================================
# TAB 1 - DATASET OVERVIEW
# ==================================================

with tab1:

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(data):,}")
    col2.metric("Total Features", len(features))
    col3.metric("Target Classes", y.nunique())
    col4.metric("Missing Values", int(data[features].isna().sum().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Target Class Distribution")

        class_counts = y.value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(plot_bg)
        ax.set_facecolor(plot_bg)

        ax.bar(
            class_counts.index.astype(str),
            class_counts.values
        )

        ax.set_xlabel("Class", color=plot_fg)
        ax.set_ylabel("Number of Records", color=plot_fg)
        ax.set_title("Class Distribution", color=plot_fg)
        ax.tick_params(colors=plot_fg)

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Missing Values by Feature")

        missing = data[features].isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor(plot_bg)
            ax.set_facecolor(plot_bg)

            ax.barh(missing.index, missing.values)
            ax.set_xlabel("Missing Values", color=plot_fg)
            ax.tick_params(colors=plot_fg)
            ax.set_title("Missing Values", color=plot_fg)

            st.pyplot(fig)
            plt.close(fig)
        else:
            st.success("No missing feature values found.")

    st.subheader("Statistical Summary")
    st.dataframe(
        data[features].describe().T,
        use_container_width=True
    )

    st.subheader("Dataset Information")

    st.write(f"Dataset contains **{len(data)}** valid records.")
    st.write(f"There are **{len(features)}** input features.")
    st.write(f"The target column is **{target_column}**.")
    st.write(f"The dataset contains **{y.nunique()}** target classes.")

# ==================================================
# TAB 2 - MODEL COMPARISON
# ==================================================

with tab2:

    st.header("Comparison of All Five Models")

    st.markdown(
        "The following results are calculated using the same selected "
        "labelled dataset for all available saved models."
    )

    st.subheader("Performance Comparison")

    st.dataframe(
        comparison_df.style.format({
            metric: "{:.4f}" for metric in metric_names
        }),
        use_container_width=True
    )

    st.subheader("Best Model by Metric")

    best_cols = st.columns(3)

    for index, metric in enumerate(metric_names):
        best_index = comparison_df[metric].idxmax()

        if pd.isna(comparison_df.loc[best_index, metric]):
            continue

        best_model = comparison_df.loc[best_index, "Model"]
        best_value = comparison_df.loc[best_index, metric]

        with best_cols[index % 3]:
            st.metric(
                f"Best {metric}",
                best_model,
                f"{best_value:.4f}"
            )

    st.subheader("Select Metrics to Compare")

    selected_metrics = st.multiselect(
        "Choose performance metrics",
        metric_names,
        default=["Accuracy", "Precision", "Recall", "F1 Score"]
    )

    if selected_metrics:
        chart_df = comparison_df.set_index("Model")[selected_metrics]

        st.bar_chart(chart_df, use_container_width=True)

    st.subheader("F1 Score Ranking")

    ranking = comparison_df.sort_values(
        "F1 Score",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(plot_bg)
    ax.set_facecolor(plot_bg)

    ax.bar(
        ranking["Model"],
        ranking["F1 Score"]
    )

    ax.set_ylabel("F1 Score", color=plot_fg)
    ax.set_title("Model Ranking Based on F1 Score", color=plot_fg)
    ax.tick_params(axis="x", colors=plot_fg, rotation=20)
    ax.tick_params(axis="y", colors=plot_fg)

    st.pyplot(fig)
    plt.close(fig)

# ==================================================
# TAB 3 - DETAILED MODEL EVALUATION
# ==================================================

with tab3:

    st.header("Detailed Model Evaluation")

    selected_model = st.selectbox(
        "Select a model for detailed evaluation",
        list(results.keys())
    )

    model_result = results[selected_model]
    predictions = model_result["Predictions"]

    st.subheader(f"{selected_model} Performance")

    metric_cols = st.columns(3)

    display_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC",
        "MCC"
    ]

    for index, metric in enumerate(display_metrics):
        value = model_result[metric]

        with metric_cols[index % 3]:
            st.metric(
                metric,
                "N/A" if pd.isna(value) else f"{value:.4f}"
            )

    # Confusion matrix

    st.subheader("Confusion Matrix")

    labels = np.unique(np.concatenate([
        np.asarray(y),
        np.asarray(predictions)
    ]))

    cm = confusion_matrix(y, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(plot_bg)
    ax.set_facecolor(plot_bg)

    image = ax.imshow(cm, interpolation="nearest")

    ax.set_title("Confusion Matrix", color=plot_fg)
    ax.set_xlabel("Predicted Label", color=plot_fg)
    ax.set_ylabel("Actual Label", color=plot_fg)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, color=plot_fg)
    ax.set_yticklabels(labels, color=plot_fg)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    fig.colorbar(image, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    # Classification report

    st.subheader("Classification Report")

    report = classification_report(
        y,
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T
    st.dataframe(report_df, use_container_width=True)

    # ROC curve

    st.subheader("ROC Curve")

    probabilities = model_result["Probabilities"]

    if len(np.unique(y)) == 2 and probabilities is not None:
        try:
            model = loaded_models[selected_model]
            classes = model.classes_

            positive_class = classes[1]
            positive_index = list(classes).index(positive_class)

            y_binary = (np.asarray(y) == positive_class).astype(int)

            fpr, tpr, thresholds = roc_curve(
                y_binary,
                probabilities[:, positive_index]
            )

            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor(plot_bg)
            ax.set_facecolor(plot_bg)

            ax.plot(fpr, tpr, label="ROC Curve")
            ax.plot([0, 1], [0, 1], linestyle="--")

            ax.set_xlabel("False Positive Rate", color=plot_fg)
            ax.set_ylabel("True Positive Rate", color=plot_fg)
            ax.set_title("ROC Curve", color=plot_fg)
            ax.tick_params(colors=plot_fg)
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

        except Exception as error:
            st.info(f"ROC curve could not be displayed: {error}")
    else:
        st.info(
            "ROC curve is displayed for binary classification models "
            "with probability predictions."
        )

    # Prediction preview

    st.subheader("Actual vs Predicted Values")

    prediction_preview = pd.DataFrame({
        "Actual": np.asarray(y),
        "Predicted": np.asarray(predictions)
    })

    prediction_preview["Correct Prediction"] = (
        prediction_preview["Actual"]
        == prediction_preview["Predicted"]
    )

    st.dataframe(
        prediction_preview.head(20),
        use_container_width=True
    )

    correct_count = int(
        prediction_preview["Correct Prediction"].sum()
    )

    incorrect_count = len(prediction_preview) - correct_count

    c1, c2 = st.columns(2)
    c1.metric("Correct Predictions", correct_count)
    c2.metric("Incorrect Predictions", incorrect_count)

# ==================================================
# TAB 4 - DOWNLOAD REPORTS
# ==================================================

with tab4:

    st.header("Download Evaluation Reports")

    st.write(
        "Download the model comparison, prediction results "
        "and detailed evaluation report."
    )

    # Comparison CSV

    comparison_csv = comparison_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📊 Download Model Comparison CSV",
        data=comparison_csv,
        file_name="model_comparison_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    # All predictions CSV

    all_predictions = pd.DataFrame({
        "Actual": np.asarray(y)
    })

    for name, result in results.items():
        all_predictions[name + "_Prediction"] = result["Predictions"]

    predictions_csv = all_predictions.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download All Model Predictions CSV",
        data=predictions_csv,
        file_name="all_model_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Detailed text report

    report_lines = []

    report_lines.append("MACHINE LEARNING MODEL EVALUATION REPORT")
    report_lines.append("=" * 55)
    report_lines.append(f"Dataset source: {dataset_source}")
    report_lines.append(f"Total records: {len(data)}")
    report_lines.append(f"Total features: {len(features)}")
    report_lines.append(f"Target column: {target_column}")
    report_lines.append("")

    report_lines.append("MODEL PERFORMANCE COMPARISON")
    report_lines.append("-" * 55)
    report_lines.append(comparison_df.to_string(index=False))
    report_lines.append("")

    for name, result in results.items():

        report_lines.append("=" * 55)
        report_lines.append(f"MODEL: {name}")
        report_lines.append("=" * 55)

        for metric in metric_names:
            value = result[metric]
            report_lines.append(
                f"{metric}: "
                f"{'N/A' if pd.isna(value) else f'{value:.4f}'}"
            )

        report_lines.append("")
        report_lines.append("CLASSIFICATION REPORT")
        report_lines.append(
            classification_report(
                y,
                result["Predictions"],
                zero_division=0
            )
        )

        report_lines.append("CONFUSION MATRIX")
        report_lines.append(
            str(confusion_matrix(y, result["Predictions"]))
        )

        report_lines.append("")

    full_report = "\n".join(report_lines)

    st.download_button(
        "📄 Download Detailed Evaluation Report (TXT)",
        data=full_report.encode("utf-8"),
        file_name="detailed_model_evaluation_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    # Selected model prediction CSV

    report_model = st.selectbox(
        "Select model for individual prediction report",
        list(results.keys()),
        key="download_model"
    )

    selected_predictions = pd.DataFrame({
        "Actual": np.asarray(y),
        "Predicted": results[report_model]["Predictions"]
    })

    selected_predictions["Correct"] = (
        selected_predictions["Actual"]
        == selected_predictions["Predicted"]
    )

    selected_csv = selected_predictions.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📑 Download Selected Model Predictions",
        data=selected_csv,
        file_name=report_model.replace(" ", "_") + "_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.markdown(
    "### Machine Learning Assignment - 2"
)

st.caption(
    "Classification Models | Model Evaluation | "
    "Performance Comparison | Streamlit Deployment"
)
