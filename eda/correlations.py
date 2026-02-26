import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from data.loader import load_training_data, FEATURES


def render():
    df = load_training_data()

    corr_type = st.radio("Correlation type", ["Pearson", "Spearman"], horizontal=True)
    corr = df.corr(method=corr_type.lower())

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})
    ax.set_title(f"{corr_type} Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader(f"{corr_type} Correlation with Target")
    corr_target = corr["target"].drop("target").sort_values(key=abs, ascending=False)
    fig = px.bar(
        x=corr_target.values, y=corr_target.index,
        orientation="h", labels={"x": f"{corr_type} r", "y": "Feature"},
        color=corr_target.values, color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
    )
    fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scatter Plots: Top Features vs Target")
    top_n = st.slider("Number of top features", 3, 10, 6)
    top_feats = corr_target.head(top_n).index.tolist()
    cols_per_row = 3
    for row_start in range(0, len(top_feats), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, feat in enumerate(top_feats[row_start:row_start + cols_per_row]):
            with cols[idx]:
                fig = px.scatter(df, x=feat, y="target", opacity=0.4,
                                 trendline="ols", title=f"{feat} (r={corr['target'][feat]:.3f})")
                fig.update_layout(height=300, margin=dict(t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
