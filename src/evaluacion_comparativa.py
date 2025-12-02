import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


def evaluar_modelos():
    # -------------------- PARTE BOLÍVAR --------------------
    # Cargar datosssssssssssss
    df = pd.read_csv('data/dataset_sintetico.csv')

    # Cargar modelos entrenados
    modelo_simple = tf.keras.models.load_model('modelos/modelo_simple.h5')
    modelo_multiple = tf.keras.models.load_model('modelos/modelo_multiple.h5')

    # Predicción modelo simple (usa solo Temperatura)
    pred_simple = modelo_simple.predict(df['Temperatura'])

    # Métricas modelo simple
    mse_simple = mean_squared_error(df['Ventas'], pred_simple)
    mae_simple = mean_absolute_error(df['Ventas'], pred_simple)
        
    # -------------------- PARTE EL DORO --------------------
    # Features del modelo múltiple
    X_mult = df[['Temperatura', 'Promocion', 'Fin_de_Semana']]

    # Normalización
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X_mult)
    y_scaled = scaler_y.fit_transform(df['Ventas'].values.reshape(-1, 1))
    
    # Predicción con el modelo múltiple
    pred_mult_scaled = modelo_multiple.predict(X_scaled)

    # Desnormalizar
    pred_mult = scaler_y.inverse_transform(pred_mult_scaled)
