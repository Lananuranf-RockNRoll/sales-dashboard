import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# CACHE MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/online_shoppers_intention.csv")

# =========================
# IMPORT MODULES
# =========================
from src.eda import (
    get_basic_metrics,
    plot_revenue_distribution,
    plot_numeric_distribution,
    plot_correlation,
    plot_categorical_distribution
)

from src.evaluation import (
    plot_confusion_matrix,
    plot_roc_curve
)

@st.cache_data(show_spinner=True)
def compute_shap(X_sample):

    debug_info = {}

    best_model = model.best_estimator_
    preprocessor = best_model.named_steps["preprocessor"]
    classifier = best_model.named_steps["classifier"]

    debug_info["classifier_type"] = str(type(classifier))

    # ======================
    # TRANSFORM
    # ======================
    X_transformed = preprocessor.transform(X_sample)

    debug_info["is_sparse"] = hasattr(X_transformed, "toarray")

    # Kalau sparse → jadikan dense
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    debug_info["shape_after_transform"] = X_transformed.shape
    debug_info["dtype_before_cast"] = str(X_transformed.dtype)

    # Paksa float64
    X_transformed = X_transformed.astype("float64")

    debug_info["dtype_after_cast"] = str(X_transformed.dtype)

    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    debug_info["final_dataframe_shape"] = X_transformed_df.shape

    # ======================
    # SHAP
    # ======================
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X_transformed_df)

        debug_info["shap_type"] = str(type(shap_values))
        debug_info["shap_shape"] = shap_values.values.shape

        return shap_values, debug_info

    except Exception as e:
        debug_info["error"] = str(e)
        return None, debug_info



# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["EDA", "Model Evaluation", "Prediction", "Explainability"]
)

# =========================
# EDA
# =========================
if menu == "EDA":

    st.title("Exploratory Data Analysis")

    metrics = get_basic_metrics(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", metrics["rows"])
    col2.metric("Total Columns", metrics["columns"])
    col3.metric("Revenue Rate (%)", metrics["revenue_rate"])

    st.dataframe(df.head())

    st.pyplot(plot_revenue_distribution(df))

    numeric_cols = df.select_dtypes(include="number").columns
    selected_num = st.selectbox("Select Numerical Feature", numeric_cols)
    st.pyplot(plot_numeric_distribution(df, selected_num))

    st.pyplot(plot_correlation(df))

    categorical_cols = df.select_dtypes(include="object").columns
    selected_cat = st.selectbox("Select Categorical Feature", categorical_cols)
    st.pyplot(plot_categorical_distribution(df, selected_cat))


# =========================
# MODEL EVALUATION
# =========================
elif menu == "Model Evaluation":

    st.title("Model Performance")

    X = df.drop("Revenue", axis=1)
    y = df["Revenue"].astype(int)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    st.pyplot(plot_confusion_matrix(y, y_pred))
    st.pyplot(plot_roc_curve(y, y_proba))


# =========================
# PREDICTION
# =========================
elif menu == "Prediction":

    st.title("Customer Purchase Prediction")

    X = df.drop("Revenue", axis=1)
    input_data = {}

    st.subheader("Input Customer Data")

    for col in X.columns:
        if df[col].dtype == "object":
            input_data[col] = st.selectbox(col, df[col].unique())
        elif df[col].dtype == "bool":
            input_data[col] = st.selectbox(col, [0, 1])
        else:
            input_data[col] = st.number_input(col, float(df[col].mean()))

    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"Likely to Purchase ✅ (Probability: {probability:.2f})")
        else:
            st.error(f"Not Likely to Purchase ❌ (Probability: {probability:.2f})")


elif menu == "Explainability":

    st.title("Model Explainability")

    X = df.drop("Revenue", axis=1)
    X_sample = X.sample(200, random_state=42)

    shap_values, debug_info = compute_shap(X_sample)

    # ======================
    # DEBUG PANEL
    # ======================
    with st.expander("🔍 Debug Info"):
        for k, v in debug_info.items():
            st.write(f"{k}: {v}")

    # Kalau SHAP gagal
    if shap_values is None:
        st.error("SHAP computation failed. Check debug info.")
        st.stop()

    # Kalau multiclass
    if len(shap_values.values.shape) == 3:
        shap_values = shap_values[..., 1]

    # ======================
    # PLOTS
    # ======================
    st.subheader("SHAP Beeswarm")
    fig1 = plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig1)

    st.subheader("Feature Importance")
    fig2 = plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig2)

    st.subheader("Single Prediction Explanation")
    fig3 = plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig3)