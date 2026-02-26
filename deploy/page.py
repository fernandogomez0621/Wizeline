import streamlit as st

from modeling.engine import load_model
from deploy.csv_upload import render as render_csv_upload
from deploy.manual_input import render as render_manual_input
from deploy.blind_test import render as render_blind_test


def render():
    st.header("Predict")

    model, config = load_model()
    if model is None:
        st.warning("No trained model found. Go to Modeling and train a model first.")
        return

    st.success(f"Model loaded: **{config['model_name']}** (CV R2: {config['cv_r2']:.4f})")

    tabs = st.tabs(["Upload CSV", "Manual Input", "Blind Test"])

    with tabs[0]:
        render_csv_upload(model, config)
    with tabs[1]:
        render_manual_input(model, config)
    with tabs[2]:
        render_blind_test(model, config)
