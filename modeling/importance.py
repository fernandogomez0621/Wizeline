import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data.loader import FEATURES, add_interactions
from modeling.engine import load_model


def render(X_interact_cols):
    st.subheader("Feature Importance")

    model, config = load_model()
    if model is None:
        st.warning("No trained model found. Run tuning first.")
        return

    uses_interact = config["uses_interactions"]
    feat_names = X_interact_cols if uses_interact else FEATURES

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        inner = model.named_steps.get("model", None)
        if inner and hasattr(inner, "coef_"):
            importances = np.abs(inner.coef_)
        elif inner and hasattr(inner, "feature_importances_"):
            importances = inner.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)

    if importances is not None:
        imp_df = pd.DataFrame({
            "Feature": feat_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False)

        colors = ["coral" if "_x_" in f else "teal" for f in imp_df["Feature"]]
        fig = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"],
            orientation="h", marker_color=colors,
        ))
        fig.update_layout(
            title=f"Feature Importance -- {config['model_name']}",
            height=600, yaxis=dict(autorange="reversed"),
            xaxis_title="Importance",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(imp_df.round(4), use_container_width=True, hide_index=True)
    else:
        st.info("Cannot extract feature importance from this model type.")
