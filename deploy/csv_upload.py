import streamlit as st
import pandas as pd
import plotly.express as px

from data.loader import FEATURES, add_interactions


def render(model, config):
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload a CSV with the 20 feature columns (`feature_0` to `feature_19`).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="predict_csv")

    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        missing = [f for f in FEATURES if f not in input_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return

        input_features = input_df[FEATURES].copy()

        if config["uses_interactions"]:
            input_processed, _ = add_interactions(input_features)
        else:
            input_processed = input_features

        predictions = model.predict(input_processed)
        result_df = input_df.copy()
        result_df["target_pred"] = predictions

        st.subheader("Predictions")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Count", len(predictions))
        col2.metric("Mean", f"{predictions.mean():.4f}")
        col3.metric("Std", f"{predictions.std():.4f}")
        col4.metric("Range", f"[{predictions.min():.2f}, {predictions.max():.2f}]")

        fig = px.histogram(x=predictions, nbins=30, title="Prediction Distribution",
                           labels={"x": "Predicted Target"})
        st.plotly_chart(fig, use_container_width=True)

        csv_output = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions CSV",
            data=csv_output,
            file_name="predictions.csv",
            mime="text/csv",
        )
