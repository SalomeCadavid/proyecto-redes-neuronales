import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def modelo_ventas_simple():
    # -------------------- PARTE INTEGRANTE 1 --------------------

    # Cargar dataset
    df = pd.read_csv('data/dataset_sintetico.csv')

