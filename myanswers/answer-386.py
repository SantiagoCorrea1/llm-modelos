import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import median_absolute_error

def evaluar_regresion_robusta(df, columnas, target_col, test_size):
    X = df[columnas].to_numpy()
    y = df[target_col].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    model = TheilSenRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return median_absolute_error(y_test, y_pred)