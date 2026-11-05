"""Valutazione dei modelli: cross-validation, metriche sul test set e intervallo di previsione."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_validate

from .models import BASELINE

N_SPLITS = 10


def make_cv(random_state=42, n_splits=N_SPLITS):
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_models(models, X, y, cv=None, random_state=42):
    """Confronta i modelli con una cross-validation sul training set.

    Riporta anche l'R² sui dati di training di ogni fold: un R² di training molto più alto di
    quello di validazione indica overfitting.
    """
    cv = make_cv(random_state) if cv is None else cv
    scoring = {"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"}
    rows = []
    for name, model in models.items():
        scores = cross_validate(clone(model), X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
        rows.append({
            "modello": name,
            "R2": scores["test_R2"].mean(),
            "R2 std": scores["test_R2"].std(),
            "RMSE": -scores["test_RMSE"].mean(),
            "RMSE std": scores["test_RMSE"].std(),
            "MAE": -scores["test_MAE"].mean(),
            "R2 training": scores["train_R2"].mean(),
        })
    return pd.DataFrame(rows).set_index("modello")


def select_model(results):
    """Sceglie il modello con la regola di una deviazione standard.

    Tra i modelli (esclusa la baseline) il cui RMSE medio è entro una deviazione standard dal
    migliore, sceglie il primo in ordine di complessità, cioè nell'ordine in cui sono elencati.
    Restituisce il nome del modello migliore e di quello scelto.
    """
    candidates = results.drop(index=BASELINE, errors="ignore")
    best = candidates["RMSE"].idxmin()
    threshold = candidates.loc[best, "RMSE"] + candidates.loc[best, "RMSE std"]
    chosen = candidates.index[candidates["RMSE"] <= threshold][0]
    return best, chosen


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def prediction_interval(model, X, y, level=0.9, cv=None, random_state=42):
    """Stima l'intervallo di previsione dai residui in cross-validation sul training set.

    Restituisce gli scarti (inferiore e superiore) da sommare alla previsione: con level=0.9
    circa il 90% dei valori reali cade in [previsione + inferiore, previsione + superiore].
    """
    cv = make_cv(random_state) if cv is None else cv
    residuals = np.asarray(y) - cross_val_predict(clone(model), X, y, cv=cv, n_jobs=-1)
    tail = (1 - level) / 2
    return {
        "livello": level,
        "inferiore": float(np.quantile(residuals, tail)),
        "superiore": float(np.quantile(residuals, 1 - tail)),
    }


def interval_coverage(y_true, y_pred, interval):
    """Quota di valori reali che cadono nell'intervallo di previsione."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    inside = (y_true >= y_pred + interval["inferiore"]) & (y_true <= y_pred + interval["superiore"])
    return float(inside.mean())
