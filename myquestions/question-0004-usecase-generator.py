import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def generar_caso_de_uso_regresion_lineal():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función regresion_lineal(X_train, y_train, X_test, y_test).
    """

    # --- Configuración aleatoria ---
    n_samples = random.randint(50, 130)
    n_features = random.randint(1, 5)
    n_informative = random.randint(1, n_features)
    noise = random.uniform(5.0, 30.0)
    random_state = random.randint(0, 999)
    test_size = random.choice([0.2, 0.25, 0.3])

    # --- Generar datos sintéticos de regresión ---
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- INPUT ---
    input_data = {
        'X_train': X_train.copy(),
        'y_train': y_train.copy(),
        'X_test': X_test.copy(),
        'y_test': y_test.copy()
    }

    # --- OUTPUT esperado (Ground Truth) ---
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    mse = float(mean_squared_error(y_test, predicciones))
    r2 = float(r2_score(y_test, predicciones))

    output_data = (predicciones, mse, r2)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_regresion_lineal()
    predicciones, mse, r2 = salida_esperada
    print("=== INPUT ===")
    print(f"X_train shape: {entrada['X_train'].shape}")
    print(f"X_test shape:  {entrada['X_test'].shape}")
    print("\n=== OUTPUT ESPERADO ===")
    print(f"Predicciones (primeras 5): {predicciones[:5]}")
    print(f"MSE: {mse:.4f}")
    print(f"R²:  {r2:.4f}")