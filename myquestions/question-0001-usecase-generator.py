import pandas as pd
import numpy as np
import random


def generar_caso_de_uso_calcular_totales():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función calcular_totales(df, categoria).
    """

    # --- Configuración aleatoria ---
    categorias_posibles = ['Electrónica', 'Ropa', 'Alimentos', 'Hogar', 'Deportes']
    n_categorias = random.randint(2, 4)
    categorias = random.sample(categorias_posibles, k=n_categorias)
    categoria_filtro = random.choice(categorias)

    n_rows = random.randint(10, 30)

    productos = [f'Producto_{random.randint(1, 20)}' for _ in range(n_rows)]
    cats = [random.choice(categorias) for _ in range(n_rows)]
    cantidades = [random.randint(1, 50) for _ in range(n_rows)]
    precios = [round(random.uniform(5.0, 300.0), 2) for _ in range(n_rows)]

    df = pd.DataFrame({
        'producto': productos,
        'categoria': cats,
        'cantidad': cantidades,
        'precio_unitario': precios
    })

    # Garantizar al menos 3 filas de la categoría filtro
    count_filtro = (df['categoria'] == categoria_filtro).sum()
    if count_filtro < 3:
        extras = []
        for _ in range(3 - count_filtro):
            extras.append({
                'producto': f'Producto_{random.randint(1, 20)}',
                'categoria': categoria_filtro,
                'cantidad': random.randint(1, 50),
                'precio_unitario': round(random.uniform(5.0, 300.0), 2)
            })
        df = pd.concat([df, pd.DataFrame(extras)], ignore_index=True)

    # --- INPUT ---
    input_data = {
        'df': df.copy(),
        'categoria': categoria_filtro
    }

    # --- OUTPUT esperado (Ground Truth) ---
    resultado = df[df['categoria'] == categoria_filtro].copy()
    resultado['total'] = resultado['cantidad'] * resultado['precio_unitario']
    resultado = resultado[['producto', 'cantidad', 'precio_unitario', 'total']].reset_index(drop=True)

    output_data = resultado

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_calcular_totales()
    print("=== INPUT ===")
    print(f"Categoría a filtrar: '{entrada['categoria']}'")
    print("DataFrame:")
    print(entrada['df'])
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)