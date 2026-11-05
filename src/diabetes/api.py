"""API REST per le previsioni della progressione del diabete.

Avvio:
    uvicorn diabetes.api:app --app-dir src

Documentazione interattiva su http://localhost:8000/docs.

Configurazione tramite variabili d'ambiente:
- DIABETES_MODEL_DIR: cartella con diabetes_model.joblib e metadata.json (default: models/)
- DIABETES_API_KEY: se impostata, ogni richiesta (tranne /health) deve avere l'header
  X-API-Key con questo valore
"""

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from .export import load_model, predict_with_interval

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MAX_BATCH_SIZE = 1000

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class Patient(BaseModel):
    """Dati clinici di un paziente, con i valori originali (non scalati) del dataset.

    I limiti servono a scartare valori impossibili o errori di inserimento; i valori
    plausibili ma fuori da quelli visti in addestramento vengono accettati e segnalati.
    """

    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 60, "sex": 1, "bmi": 32.5, "bp": 110, "s1": 210,
        "s2": 130, "s3": 38, "s4": 5.5, "s5": 5.3, "s6": 105,
    }})

    age: float = Field(ge=0, le=120, description="età del paziente (anni)")
    sex: Literal[0, 1] = Field(description="sesso, codificato 0/1 come nel dataset di addestramento")
    bmi: float = Field(gt=10, lt=80, description="indice di massa corporea")
    bp: float = Field(gt=30, lt=250, description="pressione sanguigna media")
    s1: float = Field(gt=0, lt=1000, description="colesterolo sierico totale")
    s2: float = Field(gt=0, lt=1000, description="lipoproteine a bassa densità (LDL)")
    s3: float = Field(gt=0, lt=500, description="lipoproteine ad alta densità (HDL)")
    s4: float = Field(gt=0, lt=50, description="rapporto colesterolo totale / HDL")
    s5: float = Field(gt=0, lt=15, description="logaritmo dei trigliceridi sierici")
    s6: float = Field(gt=0, lt=1000, description="livello di glicemia")


class PatientBatch(BaseModel):
    pazienti: list[Patient] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    previsione: float = Field(description="progressione della malattia prevista a un anno")
    minimo: float = Field(description="limite inferiore dell'intervallo di previsione")
    massimo: float = Field(description="limite superiore dell'intervallo di previsione")
    livello_intervallo: float = Field(description="probabilità stimata che il valore reale cada nell'intervallo")
    fuori_intervallo_training: list[str] = Field(
        description="variabili con valori fuori da quelli visti in addestramento: la previsione è meno affidabile"
    )


class BatchPrediction(BaseModel):
    previsioni: list[Prediction]


class ModelInfo(BaseModel):
    modello: str
    data_addestramento: str
    variabili: list[str]
    intervalli_training: dict[str, list[float]]
    metriche_test: dict[str, float]
    intervallo_previsione: dict[str, float]
    versioni: dict[str, str]


def _model_dir():
    return Path(os.environ.get("DIABETES_MODEL_DIR", DEFAULT_MODEL_DIR))


def create_app(model_dir=None, api_key=None):
    """Crea l'applicazione. Senza argomenti legge la configurazione dalle variabili d'ambiente."""

    @asynccontextmanager
    async def lifespan(app):
        # Il modello viene caricato una sola volta, all'avvio
        app.state.model, app.state.metadata = load_model(model_dir or _model_dir())
        yield

    app = FastAPI(
        title="Previsione della progressione del diabete",
        description=(
            "Previsione della progressione del diabete a un anno a partire da 10 variabili cliniche. "
            "Ogni previsione include un intervallo di previsione. Il modello è uno strumento di "
            "supporto alla valutazione clinica, non un sostituto."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    expected_key = api_key if api_key is not None else os.environ.get("DIABETES_API_KEY")

    def check_api_key(key: str | None = Security(api_key_header)):
        if expected_key and not (key and secrets.compare_digest(key, expected_key)):
            raise HTTPException(status_code=401, detail="API key mancante o non valida")

    def predict(request: Request, patients: list[Patient]) -> list[Prediction]:
        metadata = request.app.state.metadata
        data = pd.DataFrame([p.model_dump() for p in patients])
        result = predict_with_interval(request.app.state.model, metadata, data)
        level = metadata["intervallo_previsione"]["livello"]
        return [
            Prediction(
                previsione=row.previsione,
                minimo=row.minimo,
                massimo=row.massimo,
                livello_intervallo=level,
                fuori_intervallo_training=[c for c in row.fuori_intervallo_training.split(", ") if c],
            )
            for row in result.itertuples()
        ]

    @app.get("/health", tags=["servizio"])
    def health():
        """Verifica che il servizio sia attivo."""
        return {"stato": "ok"}

    @app.get("/model", response_model=ModelInfo, tags=["modello"], dependencies=[Depends(check_api_key)])
    def model_info(request: Request):
        """Informazioni sul modello: variabili attese, metriche, intervallo di previsione e versioni."""
        return request.app.state.metadata

    @app.post("/predict", response_model=Prediction, tags=["previsioni"], dependencies=[Depends(check_api_key)])
    def predict_one(patient: Patient, request: Request):
        """Previsione per un singolo paziente."""
        return predict(request, [patient])[0]

    @app.post("/predict/batch", response_model=BatchPrediction, tags=["previsioni"],
              dependencies=[Depends(check_api_key)])
    def predict_batch(batch: PatientBatch, request: Request):
        """Previsione per più pazienti in una sola richiesta (al massimo 1000)."""
        return {"previsioni": predict(request, batch.pazienti)}

    return app


app = create_app()
