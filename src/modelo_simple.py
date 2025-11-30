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
