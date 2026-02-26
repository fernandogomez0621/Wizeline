import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import gaussian_kde

from data.loader import load_training_data, load_blind_test, add_interactions


def render(model, config):
    st.subheader("Blind Test Prediction")
    st.markdown("Generate predictions for the 200-sample blind test set included with the challenge.")

    if st.button("Predict Blind Test", type="primary", key="blind_predict"):
        blind_df = load_blind_test()

        if config["uses_interactions"]:
            blind_processed, _ = add_interactions(blind_df)
        else:
            blind_processed = blind_df

        predictions = model.predict(blind_processed)
        pred_df = pd.DataFrame({"target_pred": predictions})

        st.dataframe(pred_df, use_container_width=True, hide_index=True, height=300)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Count", len(predictions))
        c2.metric("Mean", f"{predictions.mean():.4f}")
        c3.metric("Median", f"{np.median(predictions):.4f}")
        c4.metric("Std", f"{predictions.std():.4f}")
        c5.metric("Range", f"[{predictions.min():.2f}, {predictions.max():.2f}]")

        # KDE comparison
        st.subheader("Distribution Comparison: Training vs Predictions (KDE)")
        train_target = load_training_data()["target"]

        x_range = np.linspace(
            min(train_target.min(), predictions.min()) - 2,
            max(train_target.max(), predictions.max()) + 2,
            500,
        )
        kde_train = gaussian_kde(train_target)
        kde_pred = gaussian_kde(predictions)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_range, y=kde_train(x_range), mode="lines", name="Training (actual)",
            line=dict(color="steelblue", width=2.5),
            fill="tozeroy", fillcolor="rgba(70,130,180,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=x_range, y=kde_pred(x_range), mode="lines", name="Blind Test (predictions)",
            line=dict(color="coral", width=2.5),
            fill="tozeroy", fillcolor="rgba(255,127,80,0.15)",
        ))
        fig.update_layout(
            title="KDE: Training Actual vs Blind Test Predictions",
            xaxis_title="Target", yaxis_title="Density", height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # KS test
        ks_stat, ks_p = stats.ks_2samp(train_target, predictions)
        c1, c2 = st.columns(2)
        c1.metric("KS Statistic", f"{ks_stat:.4f}")
        c2.metric("KS p-value", f"{ks_p:.4f}")
        if ks_p > 0.05:
            st.success("Distributions are similar (H0 not rejected at alpha = 0.05)")
        else:
            st.warning("Distributions differ -- review predictions")

        # Comparison table
        st.subheader("Statistics Comparison")
        comp_df = pd.DataFrame({
            "Statistic": ["Mean", "Median", "Std", "Min", "Max"],
            "Training": [
                f"{train_target.mean():.4f}", f"{train_target.median():.4f}",
                f"{train_target.std():.4f}", f"{train_target.min():.4f}",
                f"{train_target.max():.4f}",
            ],
            "Predictions": [
                f"{predictions.mean():.4f}", f"{np.median(predictions):.4f}",
                f"{predictions.std():.4f}", f"{predictions.min():.4f}",
                f"{predictions.max():.4f}",
            ],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Download
        csv_output = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download blind_test_predictions.csv",
            data=csv_output,
            file_name="blind_test_predictions.csv",
            mime="text/csv",
        )
