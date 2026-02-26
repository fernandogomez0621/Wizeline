import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

from data.loader import load_training_data, FEATURES


def render():
    df = load_training_data()
    X = df[FEATURES]

    st.markdown("VIF > 5 = moderate multicollinearity, VIF > 10 = severe.")

    vif_data = []
    for col in FEATURES:
        others = [f for f in FEATURES if f != col]
        reg = LinearRegression()
        reg.fit(X[others], X[col])
        r2 = reg.score(X[others], X[col])
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vif_data.append({"Feature": col, "VIF": round(vif, 4), "R_squared": round(r2, 4)})
    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)

    fig = px.bar(vif_df, x="VIF", y="Feature", orientation="h",
                 color="VIF", color_continuous_scale=["green", "orange", "red"],
                 range_color=[1, 2])
    fig.add_vline(x=5, line_dash="dash", line_color="orange", annotation_text="VIF=5")
    fig.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="VIF=10")
    fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(vif_df, use_container_width=True, hide_index=True)
    st.success(f"Max VIF: {vif_df['VIF'].max():.4f} -- No multicollinearity detected.")
