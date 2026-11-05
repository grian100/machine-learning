"""Caricamento del Diabetes dataset di scikit-learn e suddivisione in training e test set."""

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

TARGET = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURES = {
    "age": "età del paziente (anni)",
    "sex": "sesso (variabile binaria 0/1, vedi nota)",
    "bmi": "indice di massa corporea",
    "bp": "pressione sanguigna media",
    "s1": "colesterolo sierico totale (tc)",
    "s2": "lipoproteine a bassa densità (ldl)",
    "s3": "lipoproteine ad alta densità (hdl)",
    "s4": "rapporto colesterolo totale / HDL (tch)",
    "s5": "logaritmo dei trigliceridi sierici (ltg)",
    "s6": "livello di glicemia (glu)",
}

# Nel dataset originale 'sex' vale 1 o 2, ma la documentazione non indica quale valore
# corrisponda a quale sesso: viene ricodificata come variabile binaria 0/1 senza etichette.
SEX_ENCODING = {1.0: 0, 2.0: 1}


def load_dataset():
    """Restituisce le feature (valori originali, non scalati) e il target come DataFrame/Series."""
    diabetes = load_diabetes(scaled=False, as_frame=True)
    X = diabetes.data.copy()
    X["sex"] = X["sex"].map(SEX_ENCODING).astype(int)
    y = diabetes.target.rename(TARGET)
    return X, y


def split_dataset(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split train/test riproducibile: il test set viene usato solo per la valutazione finale."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
