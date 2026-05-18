import numpy as np
from sklearn.feature_selection import VarianceThreshold

def eliminar_baja_varianza(X_matrix, umbral):
    selector = VarianceThreshold(threshold=umbral)
    return selector.fit_transform(X_matrix)