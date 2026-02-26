import streamlit as st
import pandas as pd

from data.loader import load_training_data, FEATURES


def render():
    df = load_training_data()
    y = df["target"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", df.shape[0])
    col2.metric("Features", len(FEATURES))
    col3.metric("Null Values", int(df.isnull().sum().sum()))
    col4.metric("Target Mean", f"{y.mean():.2f}")

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T.round(4), use_container_width=True)

    st.subheader("Data Types")
    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.values,
        "Non-Null": df.notnull().sum().values,
        "Unique": df.nunique().values,
    })
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
