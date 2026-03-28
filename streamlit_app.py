"""
Shareable demo app. From project root:  streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import MODEL_PATH
from src.data import load_raw_dataframe


st.set_page_config(page_title="Class modeling project", layout="wide")
st.title("Modeling project demo")
st.caption("Replace the synthetic data and model with your topic. Deploy on Streamlit Community Cloud.")

if not MODEL_PATH.exists():
    st.warning("No trained model yet. In a terminal at the project root, run: `python scripts/train.py`")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
metrics = bundle.get("metrics", {})

with st.expander("Model metrics (on hold-out split)"):
    st.write("Accuracy:", metrics.get("accuracy", "—"))
    if metrics.get("roc_auc") is not None:
        st.write("ROC-AUC:", metrics["roc_auc"])
    st.text(metrics.get("report", ""))

tab1, tab2 = st.tabs(["Sample predictions", "Data peek"])

with tab1:
    st.subheader("Predict from feature values")
    inputs = {}
    cols = st.columns(min(3, len(feature_columns)))
    for i, name in enumerate(feature_columns):
        with cols[i % len(cols)]:
            inputs[name] = st.number_input(name, value=0.0, format="%.4f")
    if st.button("Predict"):
        X = pd.DataFrame([inputs], columns=feature_columns)
        pred = model.predict(X)[0]
        st.success(f"Predicted class: **{pred}**")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            st.write("Class probabilities:", dict(zip(model.classes_, proba)))

with tab2:
    st.subheader("Raw / loaded frame (first rows)")
    df = load_raw_dataframe()
    st.dataframe(df.head(20))
