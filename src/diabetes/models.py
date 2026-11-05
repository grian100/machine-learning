"""Modelli di regressione da confrontare.

Ogni modello è una Pipeline completa (pre-processing + regressore): la standardizzazione viene
fittata insieme al modello, così non introduce data leakage in cross-validation e viene
esportata insieme al modello.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ALPHAS = np.logspace(-3, 3, 61)
BASELINE = "Baseline (media)"


def _linear(regressor, degree=1):
    steps = [("scaler", StandardScaler())]
    if degree > 1:
        steps.append(("poly", PolynomialFeatures(degree, include_bias=False)))
    steps.append(("model", regressor))
    return Pipeline(steps)


def build_models(random_state=42):
    """Modelli candidati, dal più semplice al più complesso."""
    return {
        # Predice sempre la media del training set: ogni modello utile deve batterlo
        BASELINE: Pipeline([("model", DummyRegressor(strategy="mean"))]),
        "Regressione lineare": _linear(LinearRegression()),
        # Regolarizzazione L2: alpha scelto in cross-validation interna
        "Ridge": _linear(RidgeCV(alphas=ALPHAS)),
        # Regolarizzazione L1: azzera i coefficienti delle variabili poco utili (selezione delle variabili)
        "Lasso": _linear(LassoCV(alphas=ALPHAS, cv=5, max_iter=50_000)),
        "ElasticNet": _linear(
            ElasticNetCV(alphas=ALPHAS, l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=50_000)
        ),
        # Polinomiale senza regolarizzazione: con molte interazioni va in overfitting
        "Polinomiale grado 2": _linear(LinearRegression(), degree=2),
        "Polinomiale grado 3": _linear(LinearRegression(), degree=3),
        # Polinomiale con regolarizzazione Ridge
        "Polinomiale grado 2 + Ridge": _linear(RidgeCV(alphas=ALPHAS), degree=2),
        # Modelli non lineari ad alberi (non richiedono standardizzazione)
        "Random forest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=5, random_state=random_state
            )),
        ]),
        "Gradient boosting": Pipeline([
            ("model", GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.03, max_depth=2, subsample=0.8,
                random_state=random_state,
            )),
        ]),
    }


def linear_coefficients(pipeline, feature_names):
    """Coefficienti di un modello lineare fittato, sulle variabili standardizzate.

    Essendo calcolati su variabili con media 0 e deviazione standard 1, sono confrontabili tra loro:
    indicano di quanto cambia la previsione aumentando la variabile di una deviazione standard.
    """
    coef = pipeline.named_steps["model"].coef_
    return pd.Series(coef, index=feature_names, name="coefficiente").sort_values(key=abs, ascending=False)
