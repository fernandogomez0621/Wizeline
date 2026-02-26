import numpy as np
import pandas as pd
import joblib
import os
import time
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def get_cv():
    return KFold(n_splits=5, shuffle=True, random_state=42)


def get_model_configs():
    return {
        "Linear Regression": {
            "model": LinearRegression(),
            "grid": {},
            "use_random": False,
        },
        "Ridge": {
            "model": Pipeline([("scaler", StandardScaler()), ("model", Ridge(max_iter=10000))]),
            "grid": {"model__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]},
            "use_random": False,
        },
        "Lasso": {
            "model": Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=10000))]),
            "grid": {"model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]},
            "use_random": False,
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "grid": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [None, 10, 15, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
            "use_random": True,
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=42, verbosity=0),
            "grid": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
            },
            "use_random": True,
        },
        "SVR (RBF)": {
            "model": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            "grid": {
                "model__C": [0.1, 1, 10, 50, 100],
                "model__epsilon": [0.01, 0.05, 0.1, 0.5],
                "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            },
            "use_random": False,
        },
    }


def evaluate_model(model, X_tr, y_tr, X_te, y_te, model_name):
    cv = get_cv()
    start = time.time()

    cv_r2 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="r2")
    cv_rmse = -cross_val_score(model, X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    test_r2 = r2_score(y_te, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    test_mae = mean_absolute_error(y_te, y_pred)

    elapsed = time.time() - start

    return {
        "Model": model_name,
        "CV_R2_mean": round(cv_r2.mean(), 4),
        "CV_R2_std": round(cv_r2.std(), 4),
        "CV_RMSE_mean": round(cv_rmse.mean(), 4),
        "Test_R2": round(test_r2, 4),
        "Test_RMSE": round(test_rmse, 4),
        "Test_MAE": round(test_mae, 4),
        "Time_s": round(elapsed, 1),
    }, model


def tune_model(model, param_dist, X_tr, y_tr, X_te, y_te, model_name, use_random=False, n_iter=50):
    cv = get_cv()

    if use_random:
        searcher = RandomizedSearchCV(
            model, param_dist, n_iter=n_iter, cv=cv, scoring="r2",
            n_jobs=-1, verbose=0, random_state=42, return_train_score=True,
        )
    else:
        searcher = GridSearchCV(
            model, param_dist, cv=cv, scoring="r2",
            n_jobs=-1, verbose=0, return_train_score=True,
        )

    start = time.time()
    searcher.fit(X_tr, y_tr)
    elapsed = time.time() - start

    y_pred = searcher.predict(X_te)
    test_r2 = r2_score(y_te, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    test_mae = mean_absolute_error(y_te, y_pred)

    return {
        "Model": model_name,
        "Best_CV_R2": round(searcher.best_score_, 4),
        "Test_R2": round(test_r2, 4),
        "Test_RMSE": round(test_rmse, 4),
        "Test_MAE": round(test_mae, 4),
        "Best_Params": searcher.best_params_,
        "Time_s": round(elapsed, 1),
    }, searcher.best_estimator_


def save_model(model, config_dict, filename="final_model.pkl"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, model_path)
    config_path = os.path.join(MODELS_DIR, "model_config.pkl")
    joblib.dump(config_dict, config_path)
    return model_path, config_path


def load_model(filename="final_model.pkl"):
    model_path = os.path.join(MODELS_DIR, filename)
    config_path = os.path.join(MODELS_DIR, "model_config.pkl")
    if os.path.exists(model_path) and os.path.exists(config_path):
        model = joblib.load(model_path)
        config = joblib.load(config_path)
        return model, config
    return None, None
