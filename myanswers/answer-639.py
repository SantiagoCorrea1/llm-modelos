import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def rankear_por_informacion_mutua(df, target_col):
    df_num = df.select_dtypes(include=[np.number])
    X = df_num.drop(columns=[target_col])
    y = df_num[target_col].to_numpy()
    
    scores = mutual_info_regression(X, y, random_state=42)
    orden = np.argsort(scores)[::-1]
    
    return np.array(X.columns[orden], dtype=str)