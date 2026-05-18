import pandas as pd
import numpy as np
import random


def generar_caso_de_uso_estadisticas_por_cliente():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función estadisticas_por_cliente(df).
    """

    # --- Configuración aleatoria ---
    n_clientes = random.randint(3, 7)
    nombres = random.sample(
        ['Ana', 'Luis', 'Carlos', 'María', 'Pedro', 'Laura', 'Jorge', 'Sofía', 'Andrés', 'Valentina'],
        k=n_clientes
    )
    n_rows = random.randint(n_clientes * 2, n_clientes * 5)

    clientes = [random.choice(nombres) for _ in range(n_rows)]
    montos = [round(random.uniform(10.0, 5000.0), 2) for _ in range(n_rows)]
    num_trans = [random.randint(1, 20) for _ in range(n_rows)]

    df = pd.DataFrame({
        'cliente': clientes,
        'monto': montos,
        'num_transacciones': num_trans
    })

    # Garantizar al menos 2 filas por cliente
    for nombre in nombres:
        if (df['cliente'] == nombre).sum() < 2:
            df = pd.concat([df, pd.DataFrame([{
                'cliente': nombre,
                'monto': round(random.uniform(10.0, 5000.0), 2),
                'num_transacciones': random.randint(1, 20)
            }])], ignore_index=True)

    # --- INPUT ---
    input_data = {'df': df.copy()}

    # --- OUTPUT esperado (Ground Truth) ---
    resultado = df.groupby('cliente').agg(
        monto_promedio=('monto', 'mean'),
        monto_maximo=('monto', 'max'),
        total_transacciones=('num_transacciones', 'sum')
    )
    resultado['monto_promedio'] = resultado['monto_promedio'].round(2)
    resultado['monto_maximo'] = resultado['monto_maximo'].round(2)
    resultado = resultado.sort_values('total_transacciones', ascending=False)

    output_data = resultado

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_estadisticas_por_cliente()
    print("=== INPUT ===")
    print(entrada['df'])
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
