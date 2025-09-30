#con qusta metodologia si valutano se ci sono eventuali valori mancanti o anomalie nei dati. Le variabili numeriche saranno standardizzate 
#in moda da avere la medesima scala di confronto, mentre le variabili categoriche verranno codificate usando tecniche di encoding.

#per stabilire se ci sono dei valori mancanti tra le features possiamo agire con il modulo isna() ed eseguire una somma dei valori mancanti 
#in tal caso le colonne del databases non hanno valori mancanti infatti tutte ritornano una somma pari a zero
from ..caricamento_del_dataset.import_dataset import diabetes_df

# numero valori mancanti per ogni colonna
diabetes_df.isna().sum()


#procediamo a costruire una pipeline agendo in modo separato tra le variabili numeriche e categoriche ottenendo la trasformazione voluta ovvero il ColumnTransform, 
#stabilendo le strategie da adottare in caso di valori mancanti e quale tecnica di encoding utilizzare in caso di varibili categoriche

#con la standardizzazione si crea una distribuzione normale, ovvero una distribuzione con media 0 e deviazione standard 1, 
# questo aspetto è molto importante nella costrizione del modello in quanto porta tutte le features (escludendo le categoriche) a dei valori confrontabili

#consideriamo solo le variabili numeriche
numerical = diabetes_df.select_dtypes(exclude=['object','category','boolean']).columns

#consideriamo anche le variabili categoriche
categorical = diabetes_df.select_dtypes(include=['object','category','boolean']).columns

#stabiliamo le variabili su cui operare la Pipeline e il target
Xpip = diabetes_df.drop(columns=['target'])
Ypip = diabetes_df['target']

#Costruiamo la pipeline che ci ritorna la 'ColumnTransformer' ovvero le operazioni che saranno eseguite sulle differenti features
ct = ColumnTransformer(transformers=[
    (
        'numeriche',
        Pipeline([
            ('missing',SimpleImputer(strategy='mean')),
            #('scaler', StandardScaler()),
            ]),
        make_column_selector(dtype_exclude=['object','category','bool'])

    ),
    (
        'categoriche',
        Pipeline([
            ('missing',SimpleImputer(strategy='most_frequent')),
            ('encoder',OrdinalEncoder(categories=[['F','M']]))
            ]),
        make_column_selector(dtype_include=['object','category','bool'])
    )
])
ct

pipeline = Pipeline([
    ('column_transformer',ct),
])
pipeline.fit_transform(Xpip,Ypip)