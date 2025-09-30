#la selezione delle variabili è cruciale per migliorare la precisione del modello e ridurre la complessità computazionale. 
#tecniche come l'analisi della correlazione e la Regressione Lasso saranno utilizzate per identificare le variabili più influenti sulla progressione del diabete.

from ..caricamento_del_dataset.intro import pd as pd
from ..caricamento_del_dataset.intro import plt,sns
from ..caricamento_del_dataset.import_dataset import diabetes_df
#heatmap_data = pipeline e selezione delle features
heatmap_data = pd.DataFrame(pipeline.fit_transform(Xpip,Ypip), columns=Xpip.columns)

features = heatmap_data.select_dtypes(exclude=['category']).columns
features


#Valutazione della matrice di correlazione tra le features
colorMap = heatmap_data[features].corr()
colorMap

#Creazione della mappa di calore per visualizzare la correlazone numerica delle variabili
plt.figure(figsize=(14, 10), dpi=80)
sns.heatmap(colorMap, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap di correlazione tra le features Diabetes')
plt.show()


df_corr = pd.DataFrame()
df_corr = heatmap_data[['age','bmi','bp','s1','s2','s3','s4','s5','s6']]
df_corr.head()

#a questo punto valutiamo tramite una funzione il modello misurando l'errore quadratico medio e l'R² eseguendo la regressione Lasso
def verifica_modello(model, dataset):

    X, y = dataset

    y_pred = model.predict(X)

    print(f"MSE: {mean_squared_error(y, y_pred):.3f}")
    print(f"R2: {r2_score(y, y_pred):.3f}")


X = df_corr
y = diabetes_df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state=0)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


#la Lasso Regression (Least Absolute Shrinkage and Selection Operator) è una tecnica di regolarizzazione che applica una penalità per prevenire 
#l'overfitting e migliorare l'accuratezza dei modelli statistici

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

print("per il set di traning il modello Lasso restituiesce questi valori:")
verifica_modello(model, (X_train, y_train))


print("per il set di test il modello Lasso restituiesce questi valori:")
verifica_modello(model, (X_test, y_test))