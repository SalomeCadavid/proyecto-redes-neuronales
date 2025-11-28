import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def generar_dataset():
# COMPLETAR: Generar 50 registros sintéticos
    np.random.seed(42)
    n_registros = 50
    temperatura = np.random.randint(15, 41, n_registros)
    promocion = np.random.randint(0, 2, n_registros)
    fin_semana = np.random.randint(0, 2, n_registros)
    # Fórmula base: ventas = 100 + 8*temperatura + 50*promocion + 30*fin_semana + ruido
    ventas = 100 + 8*temperatura + 50*promocion + 30*fin_semana +
    np.random.normal(0, 20, n_registros)
    dataset = pd.DataFrame({
    'Temperatura': temperatura,
    'Promocion': promocion,
    'Fin_de_Semana': fin_semana,
    'Ventas': ventas
    })
    return dataset
# EJECUTAR Y GUARDAR

df = generar_dataset()
df.to_csv('data/dataset_sintetico.csv', index=False)
print('Dataset generado con', len(df), 'registros')