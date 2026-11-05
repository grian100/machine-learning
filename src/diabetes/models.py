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
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ALPHAS = np.logspace(-3, 3, 61)
BASELINE = "Baseline (media)"
TUNED_GB = "Gradient boosting (GridSearchCV)"

# Griglia degli iperparametri del gradient boosting: 2 x 2 x 3 x 2 = 24 combinazioni.
# È volutamente contenuta: dentro la cross-validation di confronto ogni combinazione viene
# addestrata 5 x 10 = 50 volte. Una prima ricerca più ampia (54 combinazioni, anche con
# n_estimators=600 e learning_rate=0.01) non aveva dato risultati migliori.
GB_PARAM_GRID = {
    "n_estimators": [100, 300],
    "learning_rate": [0.03, 0.1],
    "max_depth": [1, 2, 3],
    "min_samples_leaf": [5, 20],
}


def _linear(regressor, degree=1):
    steps = [("scaler", StandardScaler())]
    if degree > 1:
        steps.append(("poly", PolynomialFeatures(degree, include_bias=False)))
    steps.append(("model", regressor))
    return Pipeline(steps)


def tuned_gradient_boosting(random_state=42, param_grid=None, cv=5):
    """Gradient boosting con iperparametri scelti da GridSearchCV.

    La ricerca usa una cross-validation interna sui soli dati passati al fit: dentro la
    cross-validation di confronto tra modelli diventa una nested cross-validation, quindi
    il punteggio stimato non è ottimistico.
    """
    search = GridSearchCV(
        GradientBoostingRegressor(subsample=0.8, random_state=random_state),
        param_grid=GB_PARAM_GRID if param_grid is None else param_grid,
        cv=KFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return Pipeline([("model", search)])


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
        TUNED_GB: tuned_gradient_boosting(random_state),
    }


def linear_coefficients(pipeline, feature_names):
    """Coefficienti di un modello lineare fittato, sulle variabili standardizzate.

    Essendo calcolati su variabili con media 0 e deviazione standard 1, sono confrontabili tra loro:
    indicano di quanto cambia la previsione aumentando la variabile di una deviazione standard.
    """
    coef = pipeline.named_steps["model"].coef_
    return pd.Series(coef, index=feature_names, name="coefficiente").sort_values(key=abs, ascending=False)


def grid_search_table(pipeline):
    """Risultati della GridSearchCV di una pipeline fittata, dalla combinazione migliore alla peggiore."""
    search = pipeline.named_steps["model"]
    results = pd.DataFrame(search.cv_results_)
    params = pd.json_normalize(results["params"])
    table = params.assign(RMSE=-results["mean_test_score"], **{"RMSE std": results["std_test_score"]})
    return table.sort_values("RMSE").reset_index(drop=True)
