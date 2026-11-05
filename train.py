"""Confronta i modelli in cross-validation, valuta quello scelto sul test set ed esporta la pipeline.

Passaggi:
1. split train/test (80/20, riproducibile);
2. cross-validation a 10 fold di tutti i modelli sul training set;
3. scelta del modello con la regola di una deviazione standard;
4. valutazione sul test set, mai usato prima, e stima dell'intervallo di previsione;
5. riaddestramento su tutti i dati ed esportazione in models/ (pipeline completa + metadati).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from sklearn.base import clone

from diabetes import (
    BASELINE,
    RANDOM_STATE,
    build_metadata,
    build_models,
    cross_validate_models,
    interval_coverage,
    linear_coefficients,
    load_dataset,
    prediction_interval,
    regression_metrics,
    save_model,
    select_model,
    split_dataset,
)

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.3f}".format)
    REPORTS_DIR.mkdir(exist_ok=True)

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print(f"Training set: {len(X_train)} pazienti, test set: {len(X_test)} pazienti\n")

    models = build_models(RANDOM_STATE)
    results = cross_validate_models(models, X_train, y_train, random_state=RANDOM_STATE)
    print("Cross-validation a 10 fold sul training set:")
    print(results, "\n")
    results.to_csv(REPORTS_DIR / "confronto_modelli.csv")

    best, chosen = select_model(results)
    print(f"Modello con RMSE più basso: {best}")
    print(f"Modello scelto (il più semplice entro una deviazione standard dal migliore): {chosen}\n")

    lasso = clone(models["Lasso"]).fit(X_train, y_train)
    coefficients = linear_coefficients(lasso, X_train.columns)
    print(f"Coefficienti del Lasso (alpha = {lasso.named_steps['model'].alpha_:.3f}) sulle variabili standardizzate:")
    print(coefficients.to_string(), "\n")
    removed = list(coefficients[coefficients == 0].index)
    print(f"Variabili escluse dal Lasso: {removed or 'nessuna'}\n")

    model = clone(models[chosen]).fit(X_train, y_train)
    test_metrics = regression_metrics(y_test, model.predict(X_test))
    baseline_metrics = regression_metrics(y_test, clone(models[BASELINE]).fit(X_train, y_train).predict(X_test))
    print(pd.DataFrame({chosen: test_metrics, BASELINE: baseline_metrics}).T.to_string(), "\n")

    interval = prediction_interval(models[chosen], X_train, y_train, level=0.9, random_state=RANDOM_STATE)
    coverage = interval_coverage(y_test, model.predict(X_test), interval)
    print(f"Intervallo di previsione al 90%: previsione da {interval['inferiore']:+.1f} a {interval['superiore']:+.1f}")
    print(f"Quota di pazienti del test set dentro l'intervallo: {coverage:.1%}\n")

    # Il modello da esportare viene riaddestrato su tutti i dati disponibili;
    # le metriche riportate restano quelle stimate sul test set
    final_model = clone(models[chosen]).fit(X, y)
    metadata = build_metadata(
        chosen, X, test_metrics, {**interval, "copertura_test": coverage},
        cv_results=results.loc[chosen].to_dict(),
    )
    path = save_model(final_model, metadata, MODELS_DIR)
    print(f"Modello esportato in {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
