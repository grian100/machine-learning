# Previsione della Progressione del Diabete in Pazienti a Rischio
Il progetto si sviluppa cercando di implementare un modello di regressione predittiva che, utilizzando dati clinici dei pazienti, il quale possa fornire previsioni accurate sulla progressione della malattia. Questo modello sarà utilizzato da medici e specialisti per personalizzare i piani terapeutici e monitorare meglio l’evoluzione del diabete nei pazienti a rischio.

# Servizi e librerie utilizzati
- Ambiente di sviluppo <strong>Python==3.12.11</strong>
- <strong>Scikit-learn==1.7.2</strong> è una libreria open source di apprendimento automatico per il linguaggio di programmazione Python. Contiene algoritmi di classificazione, regressione e clustering (raggruppamento) e macchine a vettori di supporto, regressione logistica, classificatore bayesiano, k-mean e DBSCAN
- <strong>Pandas==2.3.3</strong>  è un pacchetto Python che fornisce strutture dati veloci, flessibili ed espressive, progettate per rendere semplice e intuitivo l'utilizzo di dati "relazionali" o "etichettati". Il suo obiettivo è quello di essere il componente fondamentale di alto livello per l'analisi pratica dei dati in Python
- <strong>Numpy==2.3.3</strong> offre funzioni matematiche complete, generatori di numeri casuali, routine di algebra lineare, trasformate di Fourier e molto altro
- <strong>Matplotlib==3.10.6</strong> è una libreria completa per creare visualizzazioni statiche, animate e interattive in Python
- <strong>Seaborn==0.13.2</strong> è una libreria Python per la visualizzazione di dati basata su matplotlib . Fornisce un'interfaccia di alto livello per disegnare grafici statistici accattivanti e informativi

# Obiettivi del progetto
L'obiettivo è di sviluppare un modello di regressione che utilizzando i dati clinici disponibili, riesca a prevedere la progressione della malattia per ciascun paziente. La soluzione sarà integrata nei sistemi aziendali di gestione sanitaria per fornire previsioni personalizzate e dati predittivi ai medici, aiutandoli a prendere decisioni più informate. I vantaggi attesi riguardano:
  - <strong>Miglioramento della cura personalizzata</strong>: I medici saranno in grado di prevedere la progressione del diabete nei pazienti e adattare i piani di trattamento in anticipo.
  - <strong>Riduzione dei costi sanitari</strong>: Prevedendo con maggiore accuratezza l'evoluzione della malattia, è possibile evitare complicazioni a lungo termine che richiederebbero interventi più costosi.
  - <strong>Integrazione in piattaforme sanitarie</strong>: Il modello sarà parte integrante di una piattaforma di monitoraggio dei pazienti, migliorando la gestione a lungo termine e l'efficacia dei trattamenti.
  - <strong>Strumento di supporto decisionale</strong>: Fornire previsioni precise permette ai medici di prendere decisioni più informate, basate su dati clinici storici e attuali.

# Descrizione del dataset
Il dataset scelto per questo progetto è il Diabetes dataset fornito da scikit-learn, che contiene informazioni cliniche su pazienti affetti da diabete e un target che rappresenta la progressione della malattia. Le variabili indipendenti includono parametri come:
  - Age: età del paziente
  - Sex: genere
  - BMI: indice di massa corporea
  - BP: pressione sanguigna media
  - S1: colesterolo sierico totale
  - S2: lipoproteine a bassa densità
  - S3: lipoproteine ad alta densità
  - S4: rapporto tra colesterolo totale e HDL
  - S5: trigliceridi
  - S6: livello di glicemia
    
Il target è una misura quantitativa che riflette la progressione del diabete.

# Passaggi per la realizzazione del Modello
  - Caricamento del Dataset: il dataset sarà caricato utilizzando la funzione load_diabetes della libreria scikit-learn. Questo permetterà di accedere rapidamente ai dati necessari per il training del modello
  - Analisi Esplorativa dei Dati (EDA): verrà eseguita una prima esplorazione delle variabili per comprendere meglio le correlazioni tra di esse e il target. Verranno utilizzate visualizzazioni come scatter plot e heatmap per identificare eventuali pattern e relazioni tra variabili come BMI, BP, e S5, che potrebbero avere una maggiore influenza sulla progressione del diabete
  - Pulizia e Pre-processing dei Dati: Verranno affrontati eventuali valori mancanti o anomalie nei dati. Le variabili numeriche saranno standardizzate per garantire che abbiano la stessa scala, mentre le variabili categoriche, come il genere, verranno codificate usando tecniche di encoding
  - Selezione delle Variabili: la selezione delle variabili è cruciale per migliorare la precisione del modello e ridurre la complessità computazionale. Tecniche come l'analisi della correlazione e la Regressione Lasso saranno utilizzate per identificare le variabili più influenti sulla progressione del diabete
  - Creazione del Modello di Regressione: una volta selezionate le variabili più rilevanti, si procederà alla creazione del modello di regressione
  - Valutazione del Modello: il modello finale sarà valutato utilizzando metriche di regressione come il Mean Squared Error (MSE) e il R-squared (R²). Le prestazioni del modello saranno confrontate con modelli di base per assicurarsi che il sistema offra un significativo miglioramento
  - Esportazione del Modello: una volta ottenuto il modello ottimale, verrà esportato utilizzando pickle per essere integrato nella piattaforma sanitaria di MedPredict, pronta per l'utilizzo in un contesto clinico reale

# Repository
MACHINE-LEARNIG_PROJECT/
  - README.md
  - src/
    - caricamento del dataset
    - analisi esplorativa
      - scatterplot/
        - scatterplot-field.png #prime realzioni tra le feauters e il target
      - lineplot/
        - lineplot-field.png #confronto tra le feauters e l'età dei pazienti
    - pulizia e pre-processing
    - analisi delle variabili
       -  heatmap/
          - heatmap-features.png #mappa di calore per visualizzare la correlazone numerica delle variabili
    - creazione del modello di regressione
    - valutazione del modello
    - esportazione del modello
