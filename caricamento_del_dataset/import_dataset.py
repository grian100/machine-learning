#Il dataset sarà caricato utilizzando la funzione load_diabetes della libreria scikit-learn. Questo permetterà di accedere rapidamente ai dati necessari per il training del modello.
from .intro import pd as pd
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(scaled=False)
diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diabetes_df['target'] = diabetes.target
diabetes_df.head()
print("Nomi delle features:\n",diabetes_df.columns)
print("\n")
diabetes_df.info(verbose=True)
print("\n")
print("Valori nulli:\n",diabetes_df.isna().sum())

#numero righe e colonne del dataset
diabetes_df.shape


#il dataset load_diabetes è costituito da 11 features in totale (compreso il valore target);
#delle 11 features rimanenti, una di esse è di risposta (features "Response").
#il dataset è composto da 442 osservazioni totali per ognuna delle 11 features del dataset e non sono presenti valori nulli all'interno del dataset.
#tutte le features sono delle variabili numeriche decimali