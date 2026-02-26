import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from modeling.engine import load_model


def render(X_test, X_test_int, y_test):
    st.subheader("Residual Analysis")

    model, config = load_model()
    if model is None:
        st.warning("No trained model found. Run tuning first.")
        return

    uses_interact = config["uses_interactions"]
    Xte = X_test_int if uses_interact else X_test
    y_pred = model.predict(Xte)
    residuals = y_test.values - y_pred

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(x=y_test, y=y_pred, opacity=0.5,
                         labels={"x": "Actual", "y": "Predicted"},
                         title="Predictions vs Actual")
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                 y=[y_test.min(), y_test.max()],
                                 mode="lines", line=dict(dash="dash", color="red"), name="Perfect"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(x=y_pred, y=residuals, opacity=0.5,
                         labels={"x": "Predicted", "y": "Residual"},
                         title="Residuals vs Predictions")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(residuals, bins=25, edgecolor="black", alpha=0.7, color="steelblue", density=True)
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x_range, stats.norm.pdf(x_range, residuals.mean(), residuals.std()), "r-", lw=2)
        ax.set_title("Residual Distribution", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot of Residuals", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{residuals.mean():.4f}")
    c2.metric("Std", f"{residuals.std():.4f}")
    c3.metric("Skewness", f"{stats.skew(residuals):.4f}")
    c4.metric("Kurtosis", f"{stats.kurtosis(residuals):.4f}")
