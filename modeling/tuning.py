import streamlit as st
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from data.loader import add_interactions, FEATURES, TOP_INTERACT
from modeling.engine import (
    get_model_configs, evaluate_model, tune_model,
    save_model, get_cv,
)


def render(X_train, X_test, X_train_int, X_test_int, y_train, y_test, X_full, X_full_int, y_full):
    st.subheader("Hyperparameter Tuning (Top 3 Models)")

    if "first_round_results" not in st.session_state:
        st.warning("Run the first round first.")
        return

    results_df = st.session_state["first_round_results"]
    top3 = results_df.head(3)["Model"].tolist()
    st.write("Top 3 models:", top3)

    n_iter = st.slider("RandomizedSearchCV iterations (for tree models)", 10, 100, 50)

    if st.button("Run Hyperparameter Tuning", type="primary"):
        configs = get_model_configs()
        tuning_results = []
        tuned_models_dict = {}
        progress = st.progress(0)

        for i, model_name in enumerate(top3):
            st.text(f"Tuning: {model_name}...")
            uses_interact = "Interact" in model_name
            base_name = model_name.replace(" + Interact", "")

            cfg = configs[base_name]
            Xtr = X_train_int if uses_interact else X_train
            Xte = X_test_int if uses_interact else X_test

            if cfg["grid"]:
                model_clone = clone(cfg["model"])
                r, m = tune_model(
                    model_clone, cfg["grid"],
                    Xtr, y_train, Xte, y_test,
                    model_name,
                    use_random=cfg["use_random"],
                    n_iter=n_iter,
                )
                tuning_results.append(r)
                tuned_models_dict[model_name] = m
            else:
                model_clone = clone(cfg["model"])
                r, m = evaluate_model(model_clone, Xtr, y_train, Xte, y_test, model_name)
                tuning_results.append({
                    "Model": model_name, "Best_CV_R2": r["CV_R2_mean"],
                    "Test_R2": r["Test_R2"], "Test_RMSE": r["Test_RMSE"],
                    "Test_MAE": r["Test_MAE"], "Best_Params": "N/A",
                    "Time_s": r["Time_s"],
                })
                tuned_models_dict[model_name] = m

            progress.progress((i + 1) / len(top3))

        progress.empty()
        tuning_df = pd.DataFrame(tuning_results).sort_values("Test_R2", ascending=False)
        st.session_state["tuning_results"] = tuning_df
        st.session_state["tuned_models"] = tuned_models_dict

        # Save best model
        best_name = tuning_df.iloc[0]["Model"]
        best_model = tuned_models_dict[best_name]
        uses_interact = "Interact" in best_name

        Xf = X_full_int if uses_interact else X_full
        best_model.fit(Xf, y_full)

        cv_scores = cross_val_score(best_model, Xf, y_full, cv=get_cv(), scoring="r2")
        best_model.fit(Xf, y_full)

        config_dict = {
            "model_name": best_name,
            "uses_interactions": uses_interact,
            "interaction_features": TOP_INTERACT,
            "feature_columns": FEATURES,
            "best_params": tuning_df.iloc[0].get("Best_Params", {}),
            "cv_r2": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
        }
        save_model(best_model, config_dict)
        st.session_state["best_model_name"] = best_name
        st.session_state["best_model"] = best_model
        st.session_state["best_config"] = config_dict

    if "tuning_results" in st.session_state:
        tuning_df = st.session_state["tuning_results"]
        st.dataframe(tuning_df, use_container_width=True, hide_index=True)

        best_name = tuning_df.iloc[0]["Model"]
        st.success(f"Best model: {best_name} (Test R2 = {tuning_df.iloc[0]['Test_R2']})")

        if isinstance(tuning_df.iloc[0].get("Best_Params"), dict):
            st.json(tuning_df.iloc[0]["Best_Params"])
