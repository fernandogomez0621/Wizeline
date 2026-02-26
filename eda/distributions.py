import streamlit as st
import matplotlib.pyplot as plt

from data.loader import load_training_data, FEATURES


def render():
    df = load_training_data()
    all_cols = FEATURES + ["target"]

    st.subheader("Histograms")
    fig, axes = plt.subplots(5, 5, figsize=(20, 16))
    axes = axes.flatten()
    for i, col in enumerate(all_cols):
        ax = axes[i]
        ax.hist(df[col], bins=30, edgecolor="black", alpha=0.7,
                color="steelblue" if col != "target" else "coral")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
    for j in range(len(all_cols), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Boxplots")
    fig, axes = plt.subplots(5, 5, figsize=(20, 16))
    axes = axes.flatten()
    for i, col in enumerate(all_cols):
        ax = axes[i]
        ax.boxplot(df[col], vert=True, patch_artist=True,
                   boxprops=dict(facecolor="steelblue" if col != "target" else "coral", alpha=0.7))
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
    for j in range(len(all_cols), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
