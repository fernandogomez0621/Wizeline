import streamlit as st

from eda.overview import render as render_overview
from eda.distributions import render as render_distributions
from eda.correlations import render as render_correlations
from eda.pca import render as render_pca
from eda.vif import render as render_vif
from eda.clustering import render as render_clustering
from eda.interactions import render as render_interactions
from eda.mutual_info import render as render_mutual_info
from eda.target import render as render_target
from eda.scaling import render as render_scaling


def render():
    st.header("Exploratory Data Analysis")

    tabs = st.tabs([
        "Overview",
        "Distributions",
        "Correlations",
        "PCA",
        "VIF",
        "Clustering",
        "Interactions",
        "Mutual Info",
        "Target",
        "Scaling",
    ])

    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_distributions()
    with tabs[2]:
        render_correlations()
    with tabs[3]:
        render_pca()
    with tabs[4]:
        render_vif()
    with tabs[5]:
        render_clustering()
    with tabs[6]:
        render_interactions()
    with tabs[7]:
        render_mutual_info()
    with tabs[8]:
        render_target()
    with tabs[9]:
        render_scaling()
