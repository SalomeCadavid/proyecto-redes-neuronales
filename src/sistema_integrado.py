import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Cargar modelos
modelo_simple = tf.keras.models.load_model("modelos/modelo_simple.h5")
modelo_multiple = tf.keras.models.load_model("modelos/modelo_multiple.h5")