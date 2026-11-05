"""Previsione della progressione del diabete con il modello esportato da train.py.

Esempi:
    python predict.py                      # previsione su 5 pazienti di esempio del dataset
    python predict.py --csv pazienti.csv   # un paziente per riga, con le colonne age, sex, bmi, bp, s1-s6
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from diabetes import load_dataset, load_model, predict_with_interval

MODELS_DIR = ROOT / "models"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, help="file CSV con i dati dei pazienti")
    args = parser.parse_args()

    if not (MODELS_DIR / "diabetes_model.joblib").exists():
        sys.exit("Modello non trovato: eseguire prima 'python train.py'.")
    model, metadata = load_model(MODELS_DIR)

    if args.csv:
        patients = pd.read_csv(args.csv)
    else:
        X, y = load_dataset()
        patients = X.sample(5, random_state=0)

    result = predict_with_interval(model, metadata, patients)
    level = metadata["intervallo_previsione"]["livello"]
    print(f"Modello: {metadata['modello']} (addestrato il {metadata['data_addestramento']})")
    print(f"Errore medio atteso (RMSE sul test set): {metadata['metriche_test']['RMSE']:.1f}")
    print(f"Intervallo di previsione al {level:.0%}\n")
    if not args.csv:
        result.insert(0, "valore reale", y.loc[patients.index])
    pd.set_option("display.float_format", "{:.1f}".format)
    print(result.to_string())
    if result["fuori_intervallo_training"].str.len().any():
        print("\nAttenzione: alcuni valori sono fuori dall'intervallo visto in addestramento, "
              "la previsione per quei pazienti è meno affidabile.")


if __name__ == "__main__":
    main()
