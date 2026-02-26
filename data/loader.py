import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

TOP_INTERACT = ["feature_2", "feature_13", "feature_9", "feature_11"]

FEATURES = [f"feature_{i}" for i in range(20)]


def load_training_data():
    path = os.path.join(DATA_DIR, "training_data.csv")
    df = pd.read_csv(path)
    return df


def load_blind_test():
    path = os.path.join(DATA_DIR, "blind_test_data.csv")
    df = pd.read_csv(path)
    return df


def add_interactions(df_input, feature_pairs=None):
    if feature_pairs is None:
        feature_pairs = TOP_INTERACT
    df_out = df_input.copy()
    interaction_cols = []
    for i in range(len(feature_pairs)):
        for j in range(i + 1, len(feature_pairs)):
            f1, f2 = feature_pairs[i], feature_pairs[j]
            col_name = f"{f1}_x_{f2}"
            df_out[col_name] = df_out[f1] * df_out[f2]
            interaction_cols.append(col_name)
    return df_out, interaction_cols


def get_interaction_columns():
    cols = []
    for i in range(len(TOP_INTERACT)):
        for j in range(i + 1, len(TOP_INTERACT)):
            cols.append(f"{TOP_INTERACT[i]}_x_{TOP_INTERACT[j]}")
    return cols
