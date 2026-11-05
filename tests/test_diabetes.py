import math

import numpy as np
import pytest
from sklearn.base import clone

from diabetes import (
    BASELINE,
    FEATURES,
    build_metadata,
    build_models,
    cross_validate_models,
    interval_coverage,
    linear_coefficients,
    load_dataset,
    load_model,
    predict_with_interval,
    prediction_interval,
    regression_metrics,
    save_model,
    select_model,
    split_dataset,
)
from diabetes.evaluation import make_cv


@pytest.fixture(scope="module")
def data():
    X, y = load_dataset()
    return split_dataset(X, y)


@pytest.fixture(scope="module")
def results(data):
    X_train, _, y_train, _ = data
    models = {k: v for k, v in build_models().items() if k not in ("Random forest", "Gradient boosting")}
    return cross_validate_models(models, X_train, y_train, cv=make_cv(n_splits=5))


def test_load_dataset():
    X, y = load_dataset()
    assert X.shape == (442, 10)
    assert list(X.columns) == list(FEATURES)
    assert set(X["sex"].unique()) == {0, 1}
    assert not X.isna().any().any()
    assert len(y) == 442


def test_split_is_reproducible():
    X, y = load_dataset()
    first, second = split_dataset(X, y), split_dataset(X, y)
    assert first[0].index.equals(second[0].index)
    assert len(first[1]) == math.ceil(0.2 * len(X))


def test_all_models_fit_and_predict(data):
    X_train, X_test, y_train, _ = data
    for name, model in build_models().items():
        prediction = clone(model).fit(X_train, y_train).predict(X_test)
        assert prediction.shape == (len(X_test),), name
        assert np.isfinite(prediction).all(), name


def test_linear_models_beat_baseline(results):
    assert results.loc[BASELINE, "R2"] < 0.05
    assert results.loc["Regressione lineare", "R2"] > 0.4
    assert results.loc["Regressione lineare", "RMSE"] < results.loc[BASELINE, "RMSE"]


def test_unregularized_polynomial_overfits(results):
    poly = results.loc["Polinomiale grado 3"]
    # Sui dati di training sembra ottimo, in validazione è peggiore della baseline
    assert poly["R2 training"] > 0.8
    assert poly["R2"] < results.loc[BASELINE, "R2"]


def test_select_model(results):
    best, chosen = select_model(results)
    assert BASELINE not in (best, chosen)
    assert results.loc[chosen, "RMSE"] <= results.loc[best, "RMSE"] + results.loc[best, "RMSE std"]
    assert list(results.index).index(chosen) <= list(results.index).index(best)


def test_lasso_coefficients(data):
    X_train, _, y_train, _ = data
    lasso = clone(build_models()["Lasso"]).fit(X_train, y_train)
    coefficients = linear_coefficients(lasso, X_train.columns)
    assert set(coefficients.index) == set(X_train.columns)
    assert coefficients.index[0] in ("bmi", "s5")


def test_prediction_interval(data):
    X_train, X_test, y_train, y_test = data
    model = build_models()["Regressione lineare"]
    interval = prediction_interval(model, X_train, y_train, level=0.9, cv=make_cv(n_splits=5))
    assert interval["inferiore"] < 0 < interval["superiore"]
    coverage = interval_coverage(y_test, clone(model).fit(X_train, y_train).predict(X_test), interval)
    assert 0.8 <= coverage <= 1


def test_export_roundtrip(data, tmp_path):
    X_train, X_test, y_train, y_test = data
    model = clone(build_models()["Ridge"]).fit(X_train, y_train)
    metrics = regression_metrics(y_test, model.predict(X_test))
    interval = {"livello": 0.9, "inferiore": -90.0, "superiore": 90.0}
    save_model(model, build_metadata("Ridge", X_train, metrics, interval), tmp_path)

    loaded, metadata = load_model(tmp_path)
    assert metadata["variabili"] == list(X_train.columns)
    # Il modello esportato include la standardizzazione: accetta i dati clinici originali
    np.testing.assert_allclose(loaded.predict(X_test), model.predict(X_test))

    result = predict_with_interval(loaded, metadata, X_test.head(3))
    assert list(result.columns) == ["previsione", "minimo", "massimo", "fuori_intervallo_training"]
    np.testing.assert_allclose(result["massimo"] - result["minimo"], 180)
    assert (result["fuori_intervallo_training"] == "").all()


def test_predict_checks_input(data, tmp_path):
    X_train, X_test, y_train, y_test = data
    model = clone(build_models()["Ridge"]).fit(X_train, y_train)
    interval = {"livello": 0.9, "inferiore": -90.0, "superiore": 90.0}
    save_model(model, build_metadata("Ridge", X_train, {"RMSE": 1.0}, interval), tmp_path)
    loaded, metadata = load_model(tmp_path)

    with pytest.raises(ValueError, match="bmi"):
        predict_with_interval(loaded, metadata, X_test.drop(columns="bmi"))

    # Colonne in ordine diverso e un valore fuori dall'intervallo di training
    patient = X_test.head(1)[list(reversed(X_test.columns))].assign(bmi=80.0)
    result = predict_with_interval(loaded, metadata, patient)
    assert result["fuori_intervallo_training"].iloc[0] == "bmi"
