import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def modelo_ventas_simple():
    # -------------------- PARTE INTEGRANTE 1 --------------------

    # Cargar dataset
    df = pd.read_csv('data/dataset_sintetico.csv')
    
        # Preparar datos (solo temperatura)
    X = df['Temperatura']
    y = df['Ventas']

    # Crear modelo (1 neurona)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )
    
    # Entrenamiento
    history = model.fit(X, y, epochs=500, verbose=0)
    
    # Guardar modelo
    model.save('modelos/modelo_simple.h5')
    
    
    # Graficar pérdida
    plt.plot(history.history['loss'])
    plt.title('Pérdida del Modelo Simple')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.savefig('resultados/graficos/loss_modelo_simple.png')
    plt.close()
    
    # Predicciones de prueba
    temperaturas_test = [15, 20, 30, 40]
    predicciones = model.predict(temperaturas_test)
    print("Predicciones para temperaturas:", temperaturas_test)
    print(predicciones.flatten())

    return model, history, X, y

