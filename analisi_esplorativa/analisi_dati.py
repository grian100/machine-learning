from ..caricamento_del_dataset.import_dataset import diabetes_df
from ..caricamento_del_dataset.intro import plt,np,pd,sns

#una prima analisi viene fatta con il modulo describe che oltre alle medie fornisce valori min e max delle variabili
diabetes_df.describe()


#Un secondo approccio consente attraverso la creazione di grafici come scatterplot e heatmap (della librearia matplotlib) il confronto tra le varibili in modo da valutare 
# le relazioni dalle stesse a livello generale visivo

fig, axs = plt.subplots(3, 3, figsize=(15, 8))
fig.suptitle('Scatterplot tra le features del dataset Diabetes')

#escludiamo il target dalle feautures
features = diabetes_df.columns[:-1]

for i in range(3):
    for j in range(3):
        n = j + i * 3
        if n < len(features):
            feature = features[n]
            # Converte i campi a numeri, "errors='coerce'"" replica valori invalidi con NaN
            x_data = pd.to_numeric(diabetes_df[feature], errors='coerce')
            y_data = diabetes_df['target']

            # Filtriamo i valori NaN
            mask = ~np.isnan(x_data)
            x_data = x_data[mask]
            y_data = y_data[mask]

            axs[i, j].scatter(x_data, y_data, s=3, color='navy')

            # Sono realizzati i grafici dei campi con i valori aggiungendo una linea di tendenza della nuvola di punti
            if len(x_data) > 1 and len(y_data) > 1:
                slope, intercept = np.polyfit(x_data, y_data, 1)
                line = slope * x_data + intercept
                axs[i, j].plot(x_data, line, color='red')

            axs[i, j].set_xlabel(feature)
            axs[i, j].set_ylabel('target')
        else:
            axs[i, j].axis('off')

plt.tight_layout()
plt.show()


#volendo quantificare i valori unici all'interno di ciascuna features del dataset utilizzaimo il metodo nunique() di Pandas che ci consente di avere il dettaglio numerico:
diabetes_df.nunique()

#Si detona inoltre che nel dataset la voce 'sex', che dovrebbe caratterizzare il sesso maschile o femminile, viene indicato con una tipologia numerica (float). 
# Si procede quindi ai fini di una corretta analsi a trasformre tale feature, mappandola con la lettere 'F' (Femmina) al posto del valore 1.0 e con la lettera 'M' (Maschio) 
# al posto del valore 2.0, il metodo utilizzato è map()

dict = {1.0 : 'F', 2.0 : 'M'}
diabetes_df['sex'] = diabetes_df['sex'].map(dict)
diabetes_df.head()

#il metodo info() ci conferma l'avvenuta modifica del campo sex in un object
diabetes_df.info()

#dovendo valutare l'impatto del diabete sulle classi di età, verifichiamo come sono stratificati i valori della feature age, con il metodo .unique() della libreria di Pandas, 
#che ci conferma come sia analizzato un ampio spettro di età va da 19 ai 79 anni
for column in ["age"]:
  age_order=diabetes_df[column].unique()[:58]

  age_order = [int(age) for age in age_order]
  age_order.sort()
  print(age_order)
  
#a questo punto valutiamo nel dettaglio come in base all'eta anagrafica (age) ed al sesso (sex) si distribuisco le varie features fornite, 
#attraverso andamenti grafici in cui sull'asse delle x viene indicata l'età e sulle ordinate la feature oggetto di indagine

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Lineplot tra le features del dataset Diabetes')

#escludiamo il target dalle feautures ed anche l'asse delle x formato dal campo 'age' e la tipologia indicata dal campo 'sex'
features = diabetes_df.columns[2:-1]

for i in range(2):
    for j in range(3):
        n = j + i * 3
        if  n < len(features):
                feature = features[n]
                sns.lineplot(ax=axs[i, j], data=diabetes_df, x="age", y=feature, hue="sex",style="sex", palette="Accent")
                axs[i, j].set_xlabel('age')
                axs[i, j].set_ylabel(feature)
        else:
            axs[i, j].axis('off')

plt.tight_layout()
plt.show()