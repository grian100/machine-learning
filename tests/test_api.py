import numpy as np
import pytest
from fastapi.testclient import TestClient

from diabetes import load_dataset, load_model
from diabetes.api import DEFAULT_MODEL_DIR, MAX_BATCH_SIZE, create_app

PATIENT = {"age": 60, "sex": 1, "bmi": 32.5, "bp": 110, "s1": 210, "s2": 130, "s3": 38, "s4": 5.5, "s5": 5.3, "s6": 105}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(DEFAULT_MODEL_DIR, api_key="")) as client:
        yield client


@pytest.fixture(scope="module")
def model():
    return load_model(DEFAULT_MODEL_DIR)


def test_health(client):
    assert client.get("/health").json() == {"stato": "ok"}


def test_model_info(client, model):
    info = client.get("/model").json()
    _, metadata = model
    assert info["modello"] == metadata["modello"]
    assert info["variabili"] == metadata["variabili"]
    assert "cross_validation" not in info


def test_predict_matches_model(client, model):
    loaded, metadata = model
    X, _ = load_dataset()
    patient = X.iloc[0].to_dict()
    patient["sex"] = int(patient["sex"])
    body = client.post("/predict", json=patient).json()

    expected = loaded.predict(X.iloc[[0]])[0]
    interval = metadata["intervallo_previsione"]
    assert body["previsione"] == pytest.approx(expected)
    assert body["minimo"] == pytest.approx(expected + interval["inferiore"])
    assert body["massimo"] == pytest.approx(expected + interval["superiore"])
    assert body["livello_intervallo"] == interval["livello"]
    assert body["fuori_intervallo_training"] == []


def test_predict_flags_values_outside_training_range(client):
    body = client.post("/predict", json={**PATIENT, "bmi": 45, "age": 85}).json()
    assert sorted(body["fuori_intervallo_training"]) == ["age", "bmi"]


@pytest.mark.parametrize("change", [
    {"bmi": 5},           # valore impossibile
    {"sex": 2},           # codifica non valida
    {"age": "sessanta"},  # tipo sbagliato
])
def test_predict_rejects_invalid_values(client, change):
    response = client.post("/predict", json={**PATIENT, **change})
    assert response.status_code == 422


def test_predict_rejects_missing_fields(client):
    patient = {k: v for k, v in PATIENT.items() if k != "s5"}
    response = client.post("/predict", json=patient)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "s5"]


def test_predict_batch(client, model):
    loaded, _ = model
    X, _ = load_dataset()
    patients = X.head(5).astype({"sex": int}).to_dict(orient="records")
    body = client.post("/predict/batch", json={"pazienti": patients}).json()
    predictions = [p["previsione"] for p in body["previsioni"]]
    np.testing.assert_allclose(predictions, loaded.predict(X.head(5)))


def test_predict_batch_limits(client):
    assert client.post("/predict/batch", json={"pazienti": []}).status_code == 422
    too_many = {"pazienti": [PATIENT] * (MAX_BATCH_SIZE + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_api_key():
    with TestClient(create_app(DEFAULT_MODEL_DIR, api_key="segreta")) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json=PATIENT).status_code == 401
        assert client.post("/predict", json=PATIENT, headers={"X-API-Key": "sbagliata"}).status_code == 401
        assert client.post("/predict", json=PATIENT, headers={"X-API-Key": "segreta"}).status_code == 200
        assert client.get("/model").status_code == 401


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/model", "/predict", "/predict/batch"} <= set(paths)
