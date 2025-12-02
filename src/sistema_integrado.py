# sistema_integrado.py

import numpy as np
from modelo_simple import ModeloSimple
from modelo_multiple import ModeloMultiple

class SistemaIntegrado:
    def __init__(self):
        self.modelo_simple = ModeloSimple()
        self.modelo_multiple = ModeloMultiple()
