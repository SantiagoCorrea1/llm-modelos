import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import median_absolute_error

def evaluar_regresion_robusta(df, columnas, columna_objetivo, test_size):
    X = df[columnas].to_numpy()
    y = df[columna_objetivo].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    model = TheilSenRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return median_absolute_error(y_test, y_pred)
