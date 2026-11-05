"""Esportazione e caricamento del modello, con i metadati necessari per usarlo in sicurezza.

Il modello viene salvato con joblib (il formato consigliato da scikit-learn) insieme a un file
JSON di metadati: variabili attese, intervalli dei valori visti in training, metriche e versioni.
Attenzione: come pickle, joblib può eseguire codice al caricamento, quindi vanno caricati solo
file di provenienza fidata.
"""

import json
import platform
import warnings
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

MODEL_FILE = "diabetes_model.joblib"
METADATA_FILE = "metadata.json"


def build_metadata(model_name, X_train, test_metrics, interval, cv_results=None):
    return {
        "modello": model_name,
        "data_addestramento": date.today().isoformat(),
        "variabili": list(X_train.columns),
        "intervalli_training": {
            col: [float(X_train[col].min()), float(X_train[col].max())] for col in X_train.columns
        },
        "metriche_test": {k: float(v) for k, v in test_metrics.items()},
        "intervallo_previsione": interval,
        "cross_validation": cv_results,
        "versioni": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def save_model(model, metadata, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / MODEL_FILE)
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return directory / MODEL_FILE


def load_model(directory):
    directory = Path(directory)
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    trained_with = metadata["versioni"]["scikit-learn"]
    if trained_with != sklearn.__version__:
        warnings.warn(
            f"Il modello è stato addestrato con scikit-learn {trained_with}, "
            f"ma è installata la versione {sklearn.__version__}: i risultati potrebbero non essere affidabili."
        )
    model = joblib.load(directory / MODEL_FILE)
    return model, metadata


def check_patients(patients, metadata):
    """Verifica le colonne e restituisce, per ogni paziente, le variabili fuori dai valori visti in training."""
    expected = metadata["variabili"]
    missing = [col for col in expected if col not in patients.columns]
    if missing:
        raise ValueError(f"Mancano le variabili: {missing}")
    patients = patients[expected]
    out_of_range = {}
    for idx, row in patients.iterrows():
        flagged = [
            col for col, (low, high) in metadata["intervalli_training"].items()
            if not low <= row[col] <= high
        ]
        if flagged:
            out_of_range[idx] = flagged
    return patients, out_of_range


def predict_with_interval(model, metadata, patients):
    """Previsione della progressione con l'intervallo di previsione stimato in training."""
    patients, out_of_range = check_patients(patients, metadata)
    interval = metadata["intervallo_previsione"]
    prediction = model.predict(patients)
    result = pd.DataFrame({
        "previsione": prediction,
        "minimo": prediction + interval["inferiore"],
        "massimo": prediction + interval["superiore"],
    }, index=patients.index)
    result["fuori_intervallo_training"] = [", ".join(out_of_range.get(i, [])) for i in patients.index]
    return result
