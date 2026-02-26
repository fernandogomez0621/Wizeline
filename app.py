import streamlit as st

st.set_page_config(
    page_title="Wizeline ML Challenge",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Wizeline ML Challenge")
st.sidebar.markdown("**Andres Fernando Gomez Rojas**")
st.sidebar.markdown("Data Scientist")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["EDA", "Modeling", "Predict"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Pipeline:**
    1. **EDA** -- Explore the data
    2. **Modeling** -- Train & compare models
    3. **Predict** -- Generate predictions
    """
)

if page == "EDA":
    from eda.page import render
    render()
elif page == "Modeling":
    from modeling.page import render
    render()
elif page == "Predict":
    from deploy.page import render
    render()
