import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# Page configuration
st.set_page_config(page_title="Milling Failure Classifier", layout="wide")

st.title("Milling Machine Predictive Maintenance Evaluator")
st.markdown("Evaluate pre-trained classification models on unseen test data without retraining.")

# Model registry
MODEL_MAP = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

# Sidebar controls
st.sidebar.header("Evaluation Controls")
selected_model_name = st.sidebar.selectbox("Choose Classification Model", list(MODEL_MAP.keys()))
target_col = st.sidebar.text_input("Target Column Name", value="Machine failure")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Test Data Preview")
    st.dataframe(data.head(5), use_container_width=True)

    if target_col not in data.columns:
        st.error(f"Target column '{target_col}' not found in the uploaded file.")
    else:
        X_test = data.drop(columns=[target_col])
        y_test = data[target_col]

        try:
            # Load preprocessor and model artifact
            scaler = joblib.load("model/preprocessor.joblib")
            model = joblib.load(MODEL_MAP[selected_model_name])

            # Preprocess features without retraining
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

            # Step 5(c) Display Metrics
            st.markdown(f"### Performance Metrics: **{selected_model_name}**")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            st.markdown("---")
            vcol1, vcol2 = st.columns(2)

            # Confusion Matrix
            with vcol1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")
                st.pyplot(fig)

            # Classification Report
            with vcol2:
                st.subheader("Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=3), use_container_width=True)

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
else:
    st.info("Upload 'test_data.csv' via the sidebar to view metrics and confusion matrix.")