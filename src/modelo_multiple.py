import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def modelo_ventas_multiple():

    # -------------------- PARTE PABLITO --------------------
    # Cargar dataset
    df = pd.read_csv('data/dataset_sintetico.csv')

    # Seleccionar features y target
    X = df[['Temperatura', 'Promocion', 'Fin_de_Semana']]
    y = df['Ventas']