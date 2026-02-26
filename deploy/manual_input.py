import streamlit as st
import pandas as pd

from data.loader import load_training_data, FEATURES, add_interactions


def render(model, config):
    st.subheader("Single Prediction")
    st.markdown("Enter values for each feature to get a prediction.")

    df_train = load_training_data()

    cols_per_row = 4
    values = {}
    for row_start in range(0, len(FEATURES), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, feat in enumerate(FEATURES[row_start:row_start + cols_per_row]):
            with cols[idx]:
                feat_min = float(df_train[feat].min())
                feat_max = float(df_train[feat].max())
                feat_mean = float(df_train[feat].mean())
                values[feat] = st.number_input(
                    feat,
                    min_value=feat_min * 0.5,
                    max_value=feat_max * 1.5,
                    value=feat_mean,
                    format="%.4f",
                    key=f"manual_{feat}",
                )

    if st.button("Predict", type="primary", key="manual_predict"):
        input_row = pd.DataFrame([values])

        if config["uses_interactions"]:
            input_processed, _ = add_interactions(input_row)
        else:
            input_processed = input_row

        prediction = model.predict(input_processed)[0]
        st.metric("Predicted Target", f"{prediction:.4f}")

        target_mean = df_train["target"].mean()
        target_std = df_train["target"].std()
        z_score = (prediction - target_mean) / target_std

        if abs(z_score) < 1:
            st.success(f"Within 1 std of training mean (z = {z_score:.2f})")
        elif abs(z_score) < 2:
            st.info(f"Within 2 std of training mean (z = {z_score:.2f})")
        else:
            st.warning(f"Outside 2 std of training mean (z = {z_score:.2f}) -- may be extrapolation")
