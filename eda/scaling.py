import streamlit as st
import pandas as pd
import plotly.express as px

from data.loader import load_training_data, FEATURES


def render():
    df = load_training_data()
    X = df[FEATURES]

    range_df = pd.DataFrame({
        "Feature": FEATURES,
        "Min": X.min().values,
        "Max": X.max().values,
        "Range": (X.max() - X.min()).values,
        "Mean": X.mean().values,
        "Std": X.std().values,
    }).sort_values("Range", ascending=False)

    fig = px.bar(range_df, x="Feature", y="Range", color="Range",
                 color_continuous_scale="Viridis",
                 title="Range of Each Feature (Max - Min)")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(range_df.round(4), use_container_width=True, hide_index=True)

    ratio = range_df["Range"].max() / range_df["Range"].min()
    st.info(f"Scale ratio: {ratio:.0f}x -- Standardization recommended for linear/distance-based models.")
