#una volta ottenuto è valuatato il il modello ottimale, questo verrà esportato utilizzando Pickle per essere integrato nella piattaforma sanitaria 
#pronta per l'utilizzo in un contesto clinico reale
from ..selezione_variabili.selection_feauters import model
import pickle

#genera il file pickle in cui salvare il modello
filename = 'finalized_model_diabetes.pkl'
pickle.dump(model, open(filename, 'wb'))