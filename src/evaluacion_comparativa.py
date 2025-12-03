import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # backend para guardar imágenes siempre
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


def evaluar_modelos():

    # -------------------- PARTE BOLÍVAR --------------------
    # Cargar datos
    df = pd.read_csv('data/dataset_sintetico.csv')

    # Cargar modelos entrenados
    modelo_simple = tf.keras.models.load_model('modelos/modelo_simple.h5')
    modelo_multiple = tf.keras.models.load_model('modelos/modelo_multiple.h5')

    # Predicción modelo simple (usa solo Temperatura)
    temp = df['Temperatura'].values.reshape(-1, 1)

    # Predicción SIN normalizar (modelo simple entrenado así)
    pred_simple = modelo_simple.predict(temp)

    # Métricas modelo simple
    mse_simple = mean_squared_error(df['Ventas'], pred_simple)
    mae_simple = mean_absolute_error(df['Ventas'], pred_simple)



    # -------------------- PARTE EL DORO --------------------
    # Features del modelo múltiple
    X_mult = df[['Temperatura', 'Promocion', 'Fin_de_Semana']].values

    # Normalización (IMPORTANTE: debe ser el MISMO proceso que en el entrenamiento)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Ajustar scaler con TODO el dataset original
    scaler_x.fit(X_mult)
    scaler_y.fit(df['Ventas'].values.reshape(-1, 1))

    # Normalizar features
    X_scaled = scaler_x.transform(X_mult)

    # Predicción
    pred_mult_scaled = modelo_multiple.predict(X_scaled)

    # Desnormalizar predicción para obtener valores reales de ventas
    pred_mult = scaler_y.inverse_transform(pred_mult_scaled)

    # Métricas modelo múltiple
    mse_mult = mean_squared_error(df['Ventas'], pred_mult)
    mae_mult = mean_absolute_error(df['Ventas'], pred_mult)



    # -------------------- PARTE ISAAC --------------------
    # 1. Gráfico modelo simple
    plt.figure(figsize=(8, 5))
    plt.scatter(df['Ventas'], pred_simple, alpha=0.7)
    plt.xlabel("Ventas reales")
    plt.ylabel("Predicción modelo simple")
    plt.title("Comparación Modelo Simple")
    plt.savefig("resultados/graficos/comparacion_simple.png")
    plt.close()

    # 2. Gráfico modelo múltiple
    plt.figure(figsize=(8, 5))
    plt.scatter(df['Ventas'], pred_mult, alpha=0.7)
    plt.xlabel("Ventas reales")
    plt.ylabel("Predicción modelo múltiple")
    plt.title("Comparación Modelo Múltiple")
    plt.savefig("resultados/graficos/comparacion_multiple.png")
    plt.close()

    # Comparación ambos modelos
    plt.figure(figsize=(8,5))
    plt.plot(df.index, df['Ventas'], label="Ventas reales")
    plt.plot(df.index, pred_simple, label="Predicción Simple")
    plt.plot(df.index, pred_mult, label="Predicción Múltiple")
    plt.legend()
    plt.title("Comparativa Modelos Simple vs Múltiple")
    plt.savefig("resultados/graficos/comparativa_modelos.png")
    plt.close()



    # Retornar métricas
    return {
        "Modelo_Simple": {"MSE": mse_simple, "MAE": mae_simple},
        "Modelo_Multiple": {"MSE": mse_mult, "MAE": mae_mult}
    }


resultados = evaluar_modelos()
print("Resultados comparativos:")
print(resultados)


if __name__ == "__main__":
    evaluar_modelos()
    print("Ejecución completada.")



