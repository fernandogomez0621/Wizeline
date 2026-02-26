import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data.loader import load_training_data, FEATURES


def render():
    df = load_training_data()
    X = df[FEATURES]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1, 21)), y=pca_full.explained_variance_ratio_, name="Individual"))
        fig.add_trace(go.Scatter(x=list(range(1, 21)), y=cumvar, mode="lines+markers", name="Cumulative"))
        fig.add_hline(y=0.9, line_dash="dash", line_color="gray", annotation_text="90%")
        fig.update_layout(title="Explained Variance per Component",
                          xaxis_title="Component", yaxis_title="Variance", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        n_comp_90 = int(np.argmax(cumvar >= 0.9) + 1)
        st.metric("Components for 90% variance", n_comp_90)
        st.metric("Variance with 5 components", f"{cumvar[4]*100:.1f}%")
        st.metric("Variance with 10 components", f"{cumvar[9]*100:.1f}%")

    st.subheader("PCA 2D Projection")
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": X_pca2[:, 0], "PC2": X_pca2[:, 1], "target": y})
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="target", color_continuous_scale="viridis", opacity=0.6)
    fig.update_layout(
        xaxis_title=f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA 3D Projection")
    pca3 = PCA(n_components=3)
    X_pca3 = pca3.fit_transform(X_scaled)
    pca3_df = pd.DataFrame({"PC1": X_pca3[:, 0], "PC2": X_pca3[:, 1], "PC3": X_pca3[:, 2], "target": y})
    fig = px.scatter_3d(pca3_df, x="PC1", y="PC2", z="PC3", color="target",
                        color_continuous_scale="viridis", opacity=0.5)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
