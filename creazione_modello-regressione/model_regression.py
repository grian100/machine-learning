#una volta selezionate le variabili più rilevanti, si procederà alla creazione del modello di regressione usando il Logistic Regression. 
#la regressione logistica in sintesi stima la probabilità che si verifichi un evento, sulla base di un determinato set di dati di variabili indipendenti.
from ..selezione_variabili.selection_feauters import df_corr,verifica_modello
from ..caricamento_del_dataset.import_dataset import diabetes_df

#creiamo il modello

X = df_corr # Features per il modello
y = diabetes_df['target'] # Target variabile
y = [int(label) for label in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
print(X_train)

lr=Ridge(alpha=1.)
lr.fit(X_train, y_train)

print("per il set di traning il modello Logistic restituiesce questi valori:")
verifica_modello(lr,(X_train, y_train))

print("per il set di test il modello Logistic restituiesce questi valori:")
verifica_modello(lr,(X_test, y_test))