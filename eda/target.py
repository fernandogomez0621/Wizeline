import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from data.loader import load_training_data


def render():
    df = load_training_data()
    y = df["target"]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(y, bins=30, edgecolor="black", alpha=0.7, color="coral", density=True)
        x_range = np.linspace(y.min(), y.max(), 100)
        ax.plot(x_range, stats.norm.pdf(x_range, y.mean(), y.std()), "k-", lw=2, label="Normal")
        ax.set_title("Target Distribution", fontweight="bold")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        stats.probplot(y, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    shapiro_stat, shapiro_p = stats.shapiro(y)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Skewness", f"{y.skew():.4f}")
    c2.metric("Kurtosis", f"{y.kurtosis():.4f}")
    c3.metric("Shapiro-Wilk", f"{shapiro_stat:.6f}")
    c4.metric("Shapiro p-value", f"{shapiro_p:.4f}")

    if abs(y.skew()) < 0.5:
        st.success("Target is approximately normal -- no transformation needed.")
    else:
        st.warning("Target is skewed -- consider transformation.")
