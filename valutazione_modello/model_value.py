#una volta creato il modello finale lo confronteremo anche con altri metodi di regressione, ad esempio la Lineare semplice o quella Polinomiale, 
#calcolando sia il Mean Squared Error (MSE) che R-squared (R²). Questo per valutare le prestazioni e assicurarsi di eventuali miglioramenti.
from ..selezione_variabili.selection_feauters import df_corr,verifica_modello

#Regressione lineare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

lr = LinearRegression()
lr.fit(X_train, y_train)

print(lr.coef_)
print(lr.intercept_)

y_pred=lr.predict(X_train)
print(f"l'errore assoluto medio vale {mean_absolute_error( y_train,y_pred)}")

print(f"l'errore quadratico medio vale {mean_squared_error(y_train, y_pred)}")

print(f"R quadro vale {r2_score(y_train, y_pred)}")


#Regressione Polinomiale
#La Regressione Polinomiale con grado fino a 4 con bias

X=df_corr

for i in range(1,5):

    poly = PolynomialFeatures(i)
    X_poly = poly.fit_transform(X)

    lr = LinearRegression()
    lr.fit(X_poly, y)

    print(f"Polinomio di grado {i} con bias")
    verifica_modello(lr,(X_poly,y))
    
    
    #La Regressione Polinomiale con grado fino a 4 senza bias

for i in range(1,5):

    poly = PolynomialFeatures(i,include_bias=False)
    X_poly = poly.fit_transform(X)

    lr = LinearRegression()
    lr.fit(X_poly, y)

    print(f"Polinomio di grado {i} senza bias")
    verifica_modello(lr,(X_poly,y))