# Wizeline ML Challenge - Streamlit App

## Structure

```
streamlit_app/
    app.py                      # Entry point
    requirements.txt
    data/
        __init__.py
        loader.py               # Data loading, feature engineering, constants
        training_data.csv
        blind_test_data.csv
    eda/
        __init__.py
        page.py                 # Orchestrator (tabs)
        overview.py             # Dataset overview & descriptive stats
        distributions.py        # Histograms & boxplots
        correlations.py         # Pearson, Spearman, scatter plots
        pca.py                  # PCA 2D, 3D, explained variance
        vif.py                  # Variance Inflation Factor
        clustering.py           # K-Means, elbow method
        interactions.py         # Feature products vs target
        mutual_info.py          # Mutual Information
        target.py               # Target normality (Shapiro-Wilk, Q-Q)
        scaling.py              # Feature ranges & scaling analysis
    modeling/
        __init__.py
        page.py                 # Orchestrator (tabs)
        engine.py               # Training, tuning, save/load model
        first_round.py          # 12-model comparison (default params)
        tuning.py               # RandomizedSearchCV top 3
        residuals.py            # Residual analysis (4 plots + stats)
        importance.py           # Feature importance chart
    deploy/
        __init__.py
        page.py                 # Orchestrator (tabs)
        csv_upload.py           # Batch prediction from CSV
        manual_input.py         # Single prediction (20 inputs)
        blind_test.py           # Blind test + KDE + KS test
    models/                     # Generated after training
        final_model.pkl
        model_config.pkl
```

## Setup

```bash
source envwizeline/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
