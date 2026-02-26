import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_selection import mutual_info_regression

from data.loader import load_training_data, FEATURES


def render():
    df = load_training_data()
    X = df[FEATURES]
    y = df["target"]

    mi = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({"Feature": FEATURES, "MI": mi}).sort_values("MI", ascending=False)

    fig = px.bar(mi_df, x="MI", y="Feature", orientation="h",
                 color="MI", color_continuous_scale="Teal",
                 title="Mutual Information with Target")
    fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Features with MI > 0", int((mi_df["MI"] > 0).sum()))
    col2.metric("Features with MI = 0", int((mi_df["MI"] == 0).sum()))

    st.dataframe(mi_df.round(4), use_container_width=True, hide_index=True)
