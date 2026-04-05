import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def generar_caso_de_uso_clasificar_datos():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función clasificar_datos(X_train, y_train, X_test).
    """

    # --- Configuración aleatoria ---
    n_samples = random.randint(60, 150)
    n_features = random.randint(2, 5)
    n_informative = random.randint(2, max(2, n_features))
    random_state = random.randint(0, 999)
    test_size = random.choice([0.2, 0.25, 0.3])

    # --- Generar datos sintéticos de clasificación binaria ---
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_informative, n_features),
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- INPUT ---
    input_data = {
        'X_train': X_train.copy(),
        'y_train': y_train.copy(),
        'X_test': X_test.copy()
    }

    # --- OUTPUT esperado (Ground Truth) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = LogisticRegression(random_state=42, max_iter=1000)
    modelo.fit(X_train_scaled, y_train)

    predicciones = modelo.predict(X_test_scaled)
    accuracy_train = float(accuracy_score(y_train, modelo.predict(X_train_scaled)))

    output_data = (predicciones, accuracy_train)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_clasificar_datos()
    predicciones, accuracy_train = salida_esperada
    print("=== INPUT ===")
    print(f"X_train shape: {entrada['X_train'].shape}")
    print(f"X_test shape:  {entrada['X_test'].shape}")
    print(f"y_train únicos: {set(entrada['y_train'])}")
    print("\n=== OUTPUT ESPERADO ===")
    print(f"Predicciones: {predicciones}")
    print(f"Accuracy en entrenamiento: {accuracy_train:.4f}")