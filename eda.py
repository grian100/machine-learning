"""Analisi esplorativa sul training set: statistiche, scatterplot, lineplot per età e heatmap.

I grafici vengono salvati nella cartella reports/. L'analisi usa solo il training set,
perché serve a fare scelte sul modello: il test set resta riservato alla valutazione finale.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from diabetes import FEATURES, TARGET, load_dataset, split_dataset

REPORTS_DIR = ROOT / "reports"


def scatterplots(df):
    features = [f for f in FEATURES if f != "sex"]
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for ax, feature in zip(axs.flat, features):
        ax.scatter(df[feature], df[TARGET], s=6, alpha=0.6, color="#1f4e79")
        slope, intercept = np.polyfit(df[feature], df[TARGET], 1)
        x = np.linspace(df[feature].min(), df[feature].max(), 50)
        ax.plot(x, slope * x + intercept, color="#c0504d")
        ax.set_title(f"{feature} (r = {df[feature].corr(df[TARGET]):.2f})")
        ax.set_xlabel(FEATURES[feature])
        ax.set_ylabel("progressione")
    fig.suptitle("Relazione tra le variabili e la progressione del diabete (training set)")
    fig.tight_layout()
    return fig


def lineplots(df):
    # Le età vengono raggruppate per decennio: con un valore per ogni età le curve sarebbero solo rumore
    # I pazienti sotto i 20 anni sono pochissimi e vengono accorpati al decennio 20-29
    df = df.assign(
        decennio=(df["age"] // 10 * 10).clip(lower=20).astype(int),
        gruppo=df["sex"].map({0: "sex = 0", 1: "sex = 1"}),
    )
    features = ["bmi", "bp", "s1", "s3", "s5", TARGET]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feature in zip(axs.flat, features):
        sns.lineplot(ax=ax, data=df, x="decennio", y=feature, hue="gruppo", style="gruppo",
                     markers=True, palette=["#1f4e79", "#c0504d"], errorbar=("ci", 95))
        ax.set_xlabel("età (decennio)")
        ax.set_ylabel(FEATURES.get(feature, "progressione"))
    fig.suptitle("Andamento per età e sesso (media e intervallo di confidenza al 95%, training set)")
    fig.tight_layout()
    return fig


def heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlazione tra le variabili (training set)")
    fig.tight_layout()
    return fig


def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.2f}".format)
    REPORTS_DIR.mkdir(exist_ok=True)

    X, y = load_dataset()
    print(f"Pazienti: {len(X)}, variabili: {X.shape[1]}, valori mancanti: {X.isna().sum().sum()}\n")
    for feature, description in FEATURES.items():
        print(f"  {feature:4s} {description}")

    X_train, _, y_train, _ = split_dataset(X, y)
    df = X_train.assign(**{TARGET: y_train})
    print(f"\nStatistiche descrittive (training set):\n{df.describe().T}\n")
    print(f"Valori distinti per variabile:\n{df.nunique()}\n")
    print(f"Età: da {df['age'].min():.0f} a {df['age'].max():.0f} anni")
    print(f"Correlazione con la progressione:\n{df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)}\n")

    plt.switch_backend("Agg")
    for name, plot in [("scatterplot", scatterplots), ("lineplot", lineplots), ("heatmap", heatmap)]:
        fig = plot(df)
        fig.savefig(REPORTS_DIR / f"{name}.png", dpi=110)
        plt.close(fig)
    print(f"Grafici salvati in {REPORTS_DIR.name}/")


if __name__ == "__main__":
    main()
