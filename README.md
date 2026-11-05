# Previsione della Progressione del Diabete in Pazienti a Rischio
Il progetto sviluppa un modello di regressione che, a partire dai dati clinici dei pazienti, prevede la progressione della malattia a un anno. Il modello è pensato come strumento di supporto per medici e specialisti, per personalizzare i piani terapeutici e monitorare meglio l'evoluzione del diabete nei pazienti a rischio.

# Servizi e librerie utilizzati
- <strong>Python==3.12.11</strong>
- <strong>Scikit-learn==1.7.2</strong> è una libreria open source di apprendimento automatico per Python: contiene algoritmi di regressione, classificazione e clustering, oltre a strumenti per il pre-processing e la valutazione dei modelli
- <strong>Pandas==2.3.3</strong> fornisce strutture dati veloci e flessibili per l'analisi di dati "relazionali" o "etichettati"
- <strong>Numpy==2.3.3</strong> offre funzioni matematiche complete, generatori di numeri casuali, routine di algebra lineare e molto altro
- <strong>Matplotlib==3.10.6</strong> è una libreria completa per creare visualizzazioni statiche, animate e interattive
- <strong>Seaborn==0.13.2</strong> è una libreria di visualizzazione basata su matplotlib, con un'interfaccia di alto livello per grafici statistici
- <strong>Joblib</strong> per l'esportazione del modello, <strong>pytest</strong> per i test e <strong>JupyterLab</strong> per il notebook dei risultati

Il codice può essere eseguito in locale o su <strong>Google Colaboratory</strong> [https://colab.google/].

# Obiettivi del progetto
L'obiettivo è sviluppare un modello di regressione che, utilizzando i dati clinici disponibili, preveda la progressione della malattia per ciascun paziente. La soluzione sarà integrata nei sistemi di gestione sanitaria per fornire previsioni personalizzate ai medici, aiutandoli a prendere decisioni più informate. I vantaggi attesi riguardano:
  - <strong>Miglioramento della cura personalizzata</strong>: i medici possono prevedere la progressione del diabete e adattare i piani di trattamento in anticipo.
  - <strong>Riduzione dei costi sanitari</strong>: prevedendo l'evoluzione della malattia è possibile evitare complicazioni a lungo termine che richiederebbero interventi più costosi.
  - <strong>Integrazione in piattaforme sanitarie</strong>: il modello è esportato come pipeline completa, pronta per essere integrata in una piattaforma di monitoraggio dei pazienti.
  - <strong>Strumento di supporto decisionale</strong>: ogni previsione è accompagnata da un intervallo che ne indica l'incertezza.

# Descrizione del dataset
Il dataset è il Diabetes dataset fornito da scikit-learn: 442 pazienti, 10 variabili cliniche misurate all'inizio dello studio e un target che misura la progressione della malattia un anno dopo (valori da 25 a 346). I dati vengono caricati con i valori originali, non scalati (`load_diabetes(scaled=False)`).

| Variabile | Descrizione |
|---|---|
| age | età del paziente (anni) |
| sex | sesso (variabile binaria 0/1, vedi nota) |
| bmi | indice di massa corporea |
| bp | pressione sanguigna media |
| s1 | colesterolo sierico totale |
| s2 | lipoproteine a bassa densità (LDL) |
| s3 | lipoproteine ad alta densità (HDL) |
| s4 | rapporto tra colesterolo totale e HDL |
| s5 | logaritmo dei trigliceridi sierici |
| s6 | livello di glicemia |

- Non ci sono valori mancanti.
- Nel dataset originale `sex` vale 1 o 2, ma la documentazione <strong>non indica a quale sesso corrisponda ciascun valore</strong>: viene ricodificata come variabile binaria 0/1 (1 → 0, 2 → 1) senza attribuire etichette.

# Repository
```
machine-learning/
├── src/diabetes/
│   ├── data.py          # caricamento del dataset e split train/test riproducibile
│   ├── models.py        # modelli candidati, ognuno come Pipeline (standardizzazione + regressore)
│   ├── evaluation.py    # cross-validation, scelta del modello, metriche, intervallo di previsione
│   └── export.py        # esportazione e caricamento del modello con i metadati, previsioni
├── tests/
│   └── test_diabetes.py
├── notebooks/
│   └── risultati.ipynb  # notebook con tutti i risultati, grafici e conclusioni
├── models/
│   ├── diabetes_model.joblib  # modello esportato (pipeline completa)
│   └── metadata.json          # variabili attese, intervalli di training, metriche, versioni
├── eda.py               # analisi esplorativa: scatterplot, lineplot, heatmap (in reports/)
├── train.py             # confronto dei modelli, valutazione ed esportazione (in models/)
├── predict.py           # previsione su nuovi pazienti con il modello esportato
├── requirements.txt
└── pytest.ini
```

# Utilizzo
```bash
pip install -r requirements.txt
python eda.py                          # analisi esplorativa, grafici in reports/
python train.py                        # addestramento, valutazione ed esportazione in models/
python predict.py                      # previsione su 5 pazienti di esempio
python predict.py --csv pazienti.csv   # previsione su un file con le colonne age, sex, bmi, bp, s1-s6
pytest                                 # test
jupyter lab notebooks/risultati.ipynb  # notebook con i risultati
```

Il notebook [`notebooks/risultati.ipynb`](notebooks/risultati.ipynb) è salvato già eseguito: grafici e tabelle si possono consultare direttamente su GitHub.

# Workflow
1. <strong>Caricamento del dataset e split</strong>: il dataset viene diviso subito in training set (80%) e test set (20%) con un seme fisso. Tutte le scelte (analisi esplorativa, selezione delle variabili, scelta del modello) usano solo il training set; il test set serve solo per la valutazione finale, così da evitare il data leakage.

2. <strong>Analisi esplorativa (EDA)</strong>: scatterplot delle variabili rispetto al target con retta di tendenza, andamento per età e sesso, heatmap delle correlazioni. Le variabili più legate alla progressione sono <strong>BMI</strong> (r = 0.60), <strong>s5</strong> (trigliceridi, r = 0.55) e <strong>BP</strong> (r = 0.44); `s1` e `s2` sono quasi ridondanti (r = 0.89).

3. <strong>Pre-processing</strong>: non ci sono valori mancanti né variabili categoriche da codificare (`sex` è già binaria). Le variabili vengono standardizzate dentro la pipeline di ogni modello lineare: lo scaler è fittato solo sui dati di training di ogni fold ed esportato insieme al modello.

4. <strong>Confronto dei modelli</strong>: cross-validation a 10 fold sul training set di dieci modelli, dalla baseline (che predice sempre la media) a regressione lineare, Ridge, Lasso, ElasticNet, polinomiali, random forest e gradient boosting. Gli iperparametri di Ridge, Lasso ed ElasticNet sono scelti con una cross-validation interna.

5. <strong>Selezione delle variabili</strong>: la <strong>regressione Lasso</strong>, con penalità scelta in cross-validation, azzera i coefficienti delle variabili meno utili.

6. <strong>Valutazione</strong>: il modello scelto viene valutato sul test set con <strong>R²</strong>, <strong>RMSE</strong> e <strong>MAE</strong>, confrontandolo con la baseline. L'intervallo di previsione al 90% è stimato dai residui in cross-validation e verificato sul test set.

7. <strong>Esportazione</strong>: il modello viene riaddestrato su tutti i dati ed esportato con <strong>joblib</strong> come pipeline completa (standardizzazione compresa), insieme a un file di metadati con variabili attese, intervalli dei valori visti in addestramento, metriche, intervallo di previsione e versioni delle librerie.

# Risultati

### Confronto dei modelli (cross-validation a 10 fold sul training set)

| Modello | R² validazione | RMSE | R² training |
|---|---|---|---|
| Baseline (media) | −0.03 | 77.8 | 0.00 |
| <strong>Regressione lineare</strong> | <strong>0.48</strong> | <strong>55.2</strong> | 0.53 |
| Ridge | 0.48 | 55.3 | 0.53 |
| Lasso | 0.48 | 55.3 | 0.52 |
| ElasticNet | 0.48 | 55.4 | 0.52 |
| Polinomiale grado 2 | 0.36 | 61.1 | 0.62 |
| Polinomiale grado 3 | −130.8 | 714.1 | 0.92 |
| Polinomiale grado 2 + Ridge | 0.44 | 57.0 | 0.56 |
| Random forest | 0.44 | 57.2 | 0.76 |
| Gradient boosting | 0.45 | 56.7 | 0.69 |

- I modelli lineari (con o senza regolarizzazione) sono equivalenti e non vanno in overfitting.
- I <strong>polinomiali senza regolarizzazione</strong> sembrano migliori sui dati di training (fino a R² 0.92 con il grado 3), ma su dati nuovi sono peggiori: è overfitting. Valutarli sugli stessi dati usati per addestrarli porterebbe a scegliere il modello sbagliato.
- Random forest e gradient boosting non migliorano i modelli lineari: con 442 pazienti la relazione è descritta bene da un modello lineare.

Il modello viene scelto con la <strong>regola di una deviazione standard</strong>: tra i modelli il cui RMSE è entro una deviazione standard dal migliore, il più semplice. Il modello scelto è la <strong>regressione lineare</strong>.

### Selezione delle variabili (Lasso)
Con la penalità scelta in cross-validation, il Lasso conserva `bmi`, `s5`, `bp` e, con un peso minimo, `s6` (effetto positivo) e `s3`, `sex`, `s1` (effetto negativo), ed esclude `age`, `s2` e `s4`, la cui informazione è già contenuta nelle variabili correlate. `sex` ha correlazione quasi nulla con il target, ma a parità delle altre variabili ha un effetto rilevante: per questo non va esclusa a priori.

### Valutazione sul test set

| | R² | RMSE | MAE |
|---|---|---|---|
| Regressione lineare | 0.45 | 53.9 | 42.8 |
| Baseline (media) | −0.01 | 73.2 | 64.0 |

L'intervallo di previsione al 90% va da −89 a +92 punti attorno alla previsione e contiene il 93% dei pazienti del test set.

### Limiti
Le 10 variabili cliniche spiegano poco meno della metà della variabilità della progressione (R² circa 0.45–0.48), quindi l'incertezza delle singole previsioni è elevata. Il modello distingue in modo affidabile i pazienti a rischio basso, medio o alto, ma non prevede il valore esatto: va usato come <strong>supporto</strong> alla valutazione clinica, non come sostituto. `predict.py` segnala inoltre i pazienti con valori fuori dall'intervallo visto in addestramento, per i quali la previsione è meno affidabile.

Il modello esportato è incluso nel repository in `models/`, pronto per l'integrazione: `predict.py` lo usa direttamente, senza bisogno di rieseguire `train.py`. Va caricato con la stessa versione di scikit-learn indicata in `metadata.json` (1.7.2): con versioni diverse `load_model` mostra un avviso, perché il formato interno dei modelli può cambiare tra una versione e l'altra. Rieseguendo `train.py` il modello viene ricreato e sovrascritto.

Il file del modello va caricato solo da fonti fidate: come pickle, joblib può eseguire codice al caricamento.
