import streamlit as st
from sklearn.model_selection import train_test_split

from data.loader import load_training_data, add_interactions, FEATURES
from modeling.first_round import render as render_first_round
from modeling.tuning import render as render_tuning
from modeling.residuals import render as render_residuals
from modeling.importance import render as render_importance


def render():
    st.header("Model Training & Comparison")

    df = load_training_data()
    X = df[FEATURES].copy()
    y = df["target"].copy()
    X_interact, interaction_cols = add_interactions(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_int, X_test_int, _, _ = train_test_split(X_interact, y, test_size=0.2, random_state=42)

    tabs = st.tabs(["First Round", "Hyperparameter Tuning", "Residual Analysis", "Feature Importance"])

    with tabs[0]:
        render_first_round(X_train, X_test, X_train_int, X_test_int, y_train, y_test)

    with tabs[1]:
        render_tuning(X_train, X_test, X_train_int, X_test_int,
                      y_train, y_test, X, X_interact, y)

    with tabs[2]:
        render_residuals(X_test, X_test_int, y_test)

    with tabs[3]:
        render_importance(list(X_interact.columns))
