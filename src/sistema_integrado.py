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
    
    if st.button("Predecir Ventas"):
        # ---- PREDICCIÓN MODELO SIMPLE ----
        pred_simple = modelo_simple.predict([[temp]]).flatten()[0]

        # ---- PREDICCIÓN MODELO MÚLTIPLE ----
        entrada_multi = scaler_x.transform([[temp, promocion, fin_semana]])
        pred_multi = modelo_multiple.predict(entrada_multi).flatten()[0]
        pred_multi = scaler_y.inverse_transform([[pred_multi]])[0][0]

        st.write(f"### Predicción Modelo Simple: **${pred_simple:,.2f}**")
        st.write(f"### Predicción Modelo Múltiple: **${pred_multi:,.2f}**")
        
        # ----- COMPARATIVA VISUAL -----
        st.subheader("Comparativa de resultados entre modelos")
        
        fig, ax = plt.subplots()
        modelos = ["Simple", "Múltiple"]
        valores = [pred_simple, pred_multi]
        ax.bar(modelos, valores)
        ax.set_ylabel("Ventas Predichas")
        ax.set_title("Comparación de Modelos")
        
        st.pyplot(fig)