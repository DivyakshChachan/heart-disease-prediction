import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

from src.data_preprocessing import load_dataset, get_feature_lists, build_preprocessor, TARGET_COLUMN, split_X_y
from src.model_evaluation import evaluate_classifier, plot_confusion_matrix
from sklearn.model_selection import train_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "heart_disease_dataset.csv")

st.set_page_config(page_title="Heart Disease Risk Prediction", layout="wide")

st.title("Heart Disease Risk Prediction")

@st.cache_resource
def load_models():
    lr_path = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
    xgb_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    models = {}
    if os.path.exists(lr_path):
        models["Logistic Regression"] = joblib.load(lr_path)
    if os.path.exists(xgb_path):
        models["XGBoost"] = joblib.load(xgb_path)
    return models

models = load_models()

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "XGBoost"]) if models else st.sidebar.selectbox("Choose Model", ["Logistic Regression"]) 

st.sidebar.markdown("---")
page = st.sidebar.radio("Page", ["Home", "Predict", "Compare", "Data Insights"]) 

# Load data if exists
if os.path.exists(DATA_PATH):
    df = load_dataset(DATA_PATH)
else:
    df = None

# Home Page
if page == "Home":
    st.markdown("**Overview**: End-to-end ML pipeline with explainability (SHAP) and Streamlit app.")
    if models and df is not None:
        st.success("Models loaded and dataset available.")
    elif df is None:
        st.warning("Dataset not found. Please place 'heart_disease_dataset.csv' in data/.")
    elif not models:
        st.warning("Models not found. Train and save models to models/ directory.")

# Predict Page
elif page == "Predict":
    st.subheader("Patient Inputs")

    # Default ranges sourced from common UCI heart dataset ranges
    age = st.number_input("Age", 18, 100, 54)
    sex = st.selectbox("Sex", [0, 1], help="0: female, 1: male")
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 130)
    chol = st.number_input("Cholesterol (chol)", 100, 600, 246)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate (thalach)", 60, 220, 150)
    exang = st.selectbox("Exercise-induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    input_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        if not models:
            st.error("No trained models available.")
        else:
            model = models.get(model_choice) or list(models.values())[0]
            start = time.time()
            proba = model.predict_proba(input_df)[0,1]
            pred = int(proba >= 0.5)
            latency = (time.time() - start) * 1000
            st.metric("Risk Probability", f"{proba:.2%}")
            st.caption(f"Prediction latency: {latency:.1f} ms")

            with st.expander("SHAP Explanation"):
                try:
                    # Get background data for explainer
                    if df is not None:
                        # Remove target column from background data to match input features
                        background_data = df.drop(columns=['target']).sample(min(100, len(df)), random_state=42)
                        # Create explainer with background data
                        explainer = shap.Explainer(model.predict_proba, background_data)
                        shap_values = explainer(input_df)
                        
                        # Create waterfall plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(shap_values[0], show=False, ax=ax)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Background data needed for SHAP explanations. Load dataset first.")
                except Exception as e:
                    st.error(f"SHAP explanation failed: {str(e)}")
                    st.info("Try using the Data Insights page to load the dataset first.")

# Compare Page
elif page == "Compare":
    st.subheader("Model Comparison")
    if df is None or not models:
        st.info("Need dataset and trained models.")
    else:
        # Load models and get performance metrics
        from src.model_evaluation import evaluate_classifier
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Logistic Regression")
            if "Logistic Regression" in models:
                lr_model = models["Logistic Regression"]
                # Get test predictions
                X, y = split_X_y(df)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                lr_metrics = evaluate_classifier(lr_model, X_test, y_test)
                
                st.metric("ROC-AUC", f"{lr_metrics['roc_auc']:.3f}")
                st.metric("Precision", f"{lr_metrics['precision']:.3f}")
                st.metric("Recall", f"{lr_metrics['recall']:.3f}")
                st.metric("F1-Score", f"{lr_metrics['f1']:.3f}")
        
        with col2:
            st.subheader("XGBoost")
            if "XGBoost" in models:
                xgb_model = models["XGBoost"]
                xgb_metrics = evaluate_classifier(xgb_model, X_test, y_test)
                
                st.metric("ROC-AUC", f"{xgb_metrics['roc_auc']:.3f}")
                st.metric("Precision", f"{xgb_metrics['precision']:.3f}")
                st.metric("Recall", f"{xgb_metrics['recall']:.3f}")
                st.metric("F1-Score", f"{xgb_metrics['f1']:.3f}")
        
        # Show confusion matrices
        st.subheader("Confusion Matrices")
        col1, col2 = st.columns(2)
        
        with col1:
            if "Logistic Regression" in models:
                fig, ax = plt.subplots(figsize=(6, 4))
                plot_confusion_matrix(lr_metrics, ax)
                st.pyplot(fig)
                plt.close(fig)
        
        with col2:
            if "XGBoost" in models:
                fig, ax = plt.subplots(figsize=(6, 4))
                plot_confusion_matrix(xgb_metrics, ax)
                st.pyplot(fig)
                plt.close(fig)

# Data Insights Page
elif page == "Data Insights":
    st.subheader("Data Insights")
    if df is None:
        st.info("Dataset not found.")
    else:
        st.write(f"Dataset shape: {df.shape}")
        st.write(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10))
            
            st.subheader("Basic Statistics")
            st.dataframe(df.describe())
        
        with col2:
            st.subheader("Feature Correlations")
            # Correlation heatmap
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            import seaborn as sns
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
