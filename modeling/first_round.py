import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from data.loader import load_training_data, add_interactions, FEATURES
from modeling.engine import get_model_configs, evaluate_model


def render(X_train, X_test, X_train_int, X_test_int, y_train, y_test):
    st.subheader("Model Comparison (Default Parameters)")
    st.markdown("12 models: 6 algorithms x 2 variants (with/without interactions)")

    if st.button("Run First Round", type="primary"):
        configs = get_model_configs()
        results = []
        progress = st.progress(0)
        status = st.empty()

        total = len(configs) * 2
        idx = 0
        for name, cfg in configs.items():
            status.text(f"Training: {name}...")
            model_clone = clone(cfg["model"])
            r, _ = evaluate_model(model_clone, X_train, y_train, X_test, y_test, name)
            results.append(r)
            idx += 1
            progress.progress(idx / total)

            status.text(f"Training: {name} + Interact...")
            model_clone = clone(cfg["model"])
            r, _ = evaluate_model(model_clone, X_train_int, y_train, X_test_int, y_test, f"{name} + Interact")
            results.append(r)
            idx += 1
            progress.progress(idx / total)

        progress.empty()
        status.empty()

        results_df = pd.DataFrame(results).sort_values("CV_R2_mean", ascending=False)
        st.session_state["first_round_results"] = results_df

    if "first_round_results" in st.session_state:
        results_df = st.session_state["first_round_results"]
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        sorted_df = results_df.sort_values("CV_R2_mean", ascending=True)
        colors = ["coral" if "Interact" in m else "steelblue" for m in sorted_df["Model"]]
        fig = go.Figure(go.Bar(
            x=sorted_df["CV_R2_mean"], y=sorted_df["Model"],
            orientation="h", marker_color=colors,
            error_x=dict(type="data", array=sorted_df["CV_R2_std"]),
        ))
        fig.update_layout(title="CV R2 per Model (blue=base, coral=+interactions)",
                          height=500, xaxis_title="CV R2")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(results_df, x="CV_R2_mean", y="Test_R2", text="Model",
                         title="CV R2 vs Test R2 (Overfitting Check)")
        fig.add_trace(go.Scatter(x=[0.4, 0.9], y=[0.4, 0.9], mode="lines",
                                 line=dict(dash="dash", color="red"), name="Ideal"))
        fig.update_traces(textposition="top center", textfont_size=8)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
