import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from data.loader import load_training_data, FEATURES


def render():
    df = load_training_data()
    X = df[FEATURES]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("Elbow Method")
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig = px.line(x=list(K_range), y=inertias, markers=True,
                  labels={"x": "k", "y": "Inertia"}, title="Elbow Method")
    st.plotly_chart(fig, use_container_width=True)

    k_selected = st.slider("Select k for visualization", 2, 10, 3)
    km = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X_scaled)

    cluster_df = pd.DataFrame({
        "PC1": X_pca2[:, 0], "PC2": X_pca2[:, 1],
        "Cluster": labels.astype(str), "target": y,
    })
    fig = px.scatter(cluster_df, x="PC1", y="PC2", color="Cluster", opacity=0.5,
                     title=f"K-Means (k={k_selected}) on PCA 2D")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(cluster_df, x="Cluster", y="target", color="Cluster",
                 title="Target Distribution per Cluster")
    st.plotly_chart(fig, use_container_width=True)
