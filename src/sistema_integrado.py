import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Cargar modelos
modelo_simple = tf.keras.models.load_model("modelos/modelo_simple.h5")
modelo_multiple = tf.keras.models.load_model("modelos/modelo_multiple.h5")

# Load scalers
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# Ajustar scaler con dataset real
df = pd.read_csv("data/dataset_sintetico.csv")
scaler_x.fit(df[["Temperatura", "Promocion", "Fin_de_Semana"]])
scaler_y.fit(df[["Ventas"]])

def crear_dashboard():
    st.title("Sistema Integrado de Predicción de Ventas")

    # Inputs del usuario
    temp = st.slider("Temperatura (°C)", 15, 40, 25)
    promocion = st.selectbox("¿Hay promoción?", [0, 1])
    fin_semana = st.selectbox("¿Es fin de semana?", [0, 1])