
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ==================================================
# 1. Page configuration
# ==================================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard_cleaned.csv"

MODEL_PATHS = {
    "Random Forest": BASE_DIR / "fraud_detection_model.joblib",
    "XGBoost": BASE_DIR / "xgboost_fraud_model.joblib",
}

st.title("🛡️ Credit Card Fraud Detection System")
st.write(
    "An ML application for transaction risk prediction, "
    "model evaluation, feature importance, and SHAP explanations."
)

# ==================================================
# 2. Load models and dataset
# ==================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


# ==================================================
# 3. Select model
# ==================================================
selected_model = st.selectbox(
    "Choose a machine learning model",
    list(MODEL_PATHS.keys()),
)

model_path = MODEL_PATHS[selected_model]

if not model_path.exists():
    st.error(f"Model file not found: {model_path.name}")
    st.stop()

try:
    artifact = load_model(str(model_path))
    model = artifact["model"]
    default_threshold = float(artifact["threshold"])
    features = artifact["features"]
except Exception as exc:
    st.error(f"Could not load the selected model: {exc}")
    st.stop()

st.caption(
    f"Selected model: {selected_model} | "
    f"Saved default threshold: {default_threshold:.2f}"
)

# ==================================================
# 4. Prediction threshold
# ==================================================
threshold = st.slider(
    "Fraud probability threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(np.clip(default_threshold, 0.05, 0.95)),
    step=0.05,
    help=(
        "Lower thresholds flag more transactions as potential fraud. "
        "Higher thresholds usually reduce alerts but may miss more fraud."
    ),
)

st.info(
    f"Transactions with fraud probability >= {threshold:.2f} "
    "are classified as potential fraud."
)

# ==================================================
# 5. Prediction input
# ==================================================
input_mode = st.radio(
    "Choose prediction method",
    ["Sample transaction", "Upload CSV"],
    horizontal=True,
)

# This remains available to every section below.
input_data = None
actual_labels = None
prediction_source = None

if input_mode == "Sample transaction":
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH.name}")
        st.stop()

    try:
        df = load_dataset(str(DATA_PATH))
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()

    st.write("Choose a transaction from the cleaned dataset.")

    row_number = st.number_input(
        "Dataset row number",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )

    sample_row = df.iloc[[int(row_number)]].copy()

    actual_labels = (
        sample_row["Class"].astype(int).reset_index(drop=True)
        if "Class" in sample_row.columns
        else None
    )

    missing = [col for col in features if col not in sample_row.columns]

    if missing:
        st.error(f"Missing required features: {missing}")
        st.stop()

    input_data = sample_row[features].copy()
    prediction_source = f"Dataset row {int(row_number)}"

    if actual_labels is not None:
        st.caption(
            "This dataset row has an actual label. "
            "Predictions on dataset examples are demonstrations, "
            "not independent performance tests."
        )

else:
    uploaded_file = st.file_uploader(
        "Upload a CSV containing the 30 transaction features",
        type=["csv"],
        help=(
            "Required columns: Time, V1-V28, and Amount. "
            "An optional Class column allows evaluation."
        ),
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
            st.stop()

        # Keep labels separate from the model input.
        if "Class" in uploaded_df.columns:
            raw_labels = pd.to_numeric(
                uploaded_df["Class"], errors="coerce"
            )

            if raw_labels.isna().any() or not raw_labels.isin([0, 1]).all():
                st.error("The Class column must contain only 0 and 1.")
                st.stop()

            actual_labels = raw_labels.astype(int).reset_index(drop=True)

        missing = [
            col for col in features
            if col not in uploaded_df.columns
        ]

        if missing:
            st.error(
                "Your CSV is missing required model features: "
                + ", ".join(missing)
            )
            st.stop()

        input_data = uploaded_df[features].copy()
        prediction_source = uploaded_file.name

# ==================================================
# 6. Run predictions
# ==================================================
if input_data is not None:
    if input_data.empty:
        st.warning("The input contains no transactions.")
        st.stop()

    # Validate and convert feature columns to numeric values.
    try:
        input_data = input_data.apply(
            pd.to_numeric, errors="raise"
        ).astype(float)
    except (ValueError, TypeError):
        st.error("All transaction features must contain numeric values.")
        st.stop()

    if not np.isfinite(input_data.to_numpy()).all():
        st.error(
            "The input contains missing or infinite values. "
            "Please clean the CSV and upload it again."
        )
        st.stop()

    st.divider()
    st.subheader("Transaction Predictions")

    if st.button("🚀 Generate Predictions", type="primary"):
        try:
            probabilities = model.predict_proba(input_data)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            # Store results so they persist across Streamlit reruns.
            st.session_state["fraud_results"] = {
                "source": prediction_source,
                "model_name": selected_model,
                "features": list(features),
                "input_data": input_data.copy(),
                "probabilities": probabilities.copy(),
                "predictions": predictions.copy(),
                "actual_labels": (
                    actual_labels.copy()
                    if actual_labels is not None
                    else None
                ),
                "threshold": threshold,
            }

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ==================================================
# 7. Display saved prediction results
# ==================================================
results = st.session_state.get("fraud_results")

# Clear stale results when the model or source changes.
if results is not None:
    if (
        results["model_name"] != selected_model
        or results["source"] != prediction_source
    ):
        st.session_state.pop("fraud_results", None)
        results = None

if results is not None:
    result_input = results["input_data"]
    probabilities = results["probabilities"]
    predictions = results["predictions"]
    labels = results["actual_labels"]

    # Recalculate predictions if the threshold slider changed.
    current_threshold = threshold

    if current_threshold != results["threshold"]:
        predictions = (
            probabilities >= current_threshold
        ).astype(int)

        results["predictions"] = predictions
        results["threshold"] = current_threshold
        st.session_state["fraud_results"] = results

    total = len(predictions)
    fraud_count = int(np.sum(predictions == 1))
    legitimate_count = total - fraud_count

    st.divider()
    st.subheader("Prediction Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions processed", total)
    c2.metric("Potential fraud", fraud_count)
    c3.metric("Likely legitimate", legitimate_count)

    output_df = result_input.copy()
    output_df["Fraud_Probability"] = probabilities
    output_df["Predicted_Class"] = predictions
    output_df["Prediction"] = np.where(
        predictions == 1,
        "Potential fraud",
        "Likely legitimate",
    )

    st.dataframe(output_df, use_container_width=True)

    st.download_button(
        "⬇️ Download Predictions CSV",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )

    # ==================================================
    # 8. Evaluation metrics
    # ==================================================
    st.divider()
    st.subheader("Evaluation Against Actual Labels")

    if labels is not None:
        y_true = np.asarray(labels, dtype=int)
        y_pred = np.asarray(predictions, dtype=int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_true, y_pred, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True negatives", int(tn))
        c2.metric("False alarms", int(fp))
        c3.metric("Missed fraud", int(fn))
        c4.metric("Fraud detected", int(tp))

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{precision:.3f}")
        c2.metric("Recall", f"{recall:.3f}")
        c3.metric("F1-score", f"{f1:.3f}")

        st.caption(
            "These metrics evaluate only the supplied records. "
            "Small or previously seen demo samples do not establish "
            "real-world model performance."
        )
    else:
        st.info(
            "No Class column provided. Predictions are shown "
            "without evaluation metrics."
        )

    # ==================================================
    # 9. Feature importance
    # ==================================================
    st.divider()
    st.header("🔍 Model Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        top_n = st.slider(
            "Number of features to display",
            min_value=5,
            max_value=len(features),
            value=min(15, len(features)),
            key=f"importance_top_n_{selected_model}",
        )

        chart_df = importance_df.head(top_n).sort_values(
            "Importance", ascending=True
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(chart_df["Feature"], chart_df["Importance"])
        ax.set_xlabel("Built-in feature importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {top_n} Features — {selected_model}")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        with st.expander("View importance scores"):
            st.dataframe(
                importance_df.head(top_n).reset_index(drop=True),
                use_container_width=True,
            )
    else:
        st.info("Feature importance is unavailable for this model.")

    # ==================================================
    # 10. SHAP individual explanation
    # ==================================================
    st.divider()
    st.header("🧠 Explain an Individual Prediction")

    st.write(
        "Select a transaction from the predictions above to inspect "
        "which features pushed the model output toward or away from fraud."
    )

    selected_row = st.selectbox(
        "Choose a transaction to explain",
        options=list(range(len(result_input))),
        format_func=lambda i: (
            f"Transaction {i + 1} — "
            f"{'Potential fraud' if predictions[i] == 1 else 'Likely legitimate'} "
            f"({probabilities[i]:.2%} fraud probability)"
        ),
        key="shap_transaction",
    )

    if st.button("🧠 Explain prediction", key="explain_shap"):
        try:
            with st.spinner("Calculating SHAP explanation..."):
                explainer = shap.TreeExplainer(model)

                one_row = result_input.iloc[[selected_row]]
                shap_result = explainer(one_row)

                shap_values = np.asarray(shap_result.values)

                # Binary XGBoost and Random Forest explanations
                # normally return (rows, features). If a model
                # returns class-specific values, select fraud class.
                if shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 1]

                contributions = pd.DataFrame({
                    "Feature": features,
                    "SHAP Value": shap_values[0],
                })

                contributions["Absolute Impact"] = (
                    contributions["SHAP Value"].abs()
                )

                top_contributions = (
                    contributions.nlargest(10, "Absolute Impact")
                    .sort_values("SHAP Value")
                )

                st.write(
                    f"**Predicted result:** "
                    f"{'Potential fraud' if predictions[selected_row] == 1 else 'Likely legitimate'}"
                )
                st.write(
                    f"**Fraud probability:** "
                    f"{probabilities[selected_row]:.4%}"
                )

                st.caption(
                    "For the usual binary tree-model SHAP output, "
                    "positive values push the raw model score toward fraud "
                    "and negative values push it away. These are not percentage "
                    "points and do not establish causation."
                )

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.barh(
                    top_contributions["Feature"],
                    top_contributions["SHAP Value"],
                )
                ax.axvline(0, linewidth=1)
                ax.set_xlabel("SHAP value (model output scale)")
                ax.set_ylabel("Feature")
                ax.set_title("Top 10 Contributions for This Transaction")
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                st.dataframe(
                    contributions.sort_values(
                        "Absolute Impact", ascending=False
                    ).head(10)[["Feature", "SHAP Value"]],
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Could not generate SHAP explanation: {exc}")

else:
    st.info(
        "Choose a prediction method and click **Generate Predictions** "
        "to view results, feature importance, and individual SHAP explanations."
    )

# ==================================================
# 11. Footer
# ==================================================
st.divider()
st.caption(
    "Educational fraud-detection demonstration using a public dataset. "
    "Predictions are model estimates, not definitive fraud determinations."
)