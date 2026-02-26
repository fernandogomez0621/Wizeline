import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats

from data.loader import load_training_data, FEATURES


def render():
    df = load_training_data()
    y = df["target"]

    corr_pearson = df.corr(method="pearson")
    corr_target_abs = corr_pearson["target"].drop("target").abs().sort_values(ascending=False)
    top5 = corr_target_abs.head(5).index.tolist()

    interaction_data = []
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            f1, f2 = top5[i], top5[j]
            interaction = df[f1] * df[f2]
            r_p = interaction.corr(y)
            r_s, _ = stats.spearmanr(interaction, y)
            interaction_data.append({
                "Interaction": f"{f1} x {f2}",
                "Pearson": round(r_p, 4),
                "Spearman": round(r_s, 4),
            })
    int_df = pd.DataFrame(interaction_data).sort_values("Pearson", key=abs, ascending=False)

    fig = px.bar(int_df, x="Pearson", y="Interaction", orientation="h",
                 color="Pearson", color_continuous_scale="Blues",
                 title="Interaction Correlation with Target")
    fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(int_df, use_container_width=True, hide_index=True)
    best = int_df.iloc[0]
    st.info(f"Best interaction: {best['Interaction']} (r = {best['Pearson']}), "
            f"better than any single feature (best individual: r = {corr_target_abs.iloc[0]:.4f})")
