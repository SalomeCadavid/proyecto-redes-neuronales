# Manual Técnico del Proyecto – Sistema Predictivo de Ventas con Redes Neuronales

## 1. Descripción General del Proyecto

Este proyecto implementa un sistema completo de predicción de ventas basado en redes neuronales. El sistema integra cinco módulos principales, cada uno desarrollado por un equipo: generación de datos, modelo simple, modelo múltiple, evaluación comparativa y dashboard integrado.

Estructura del proyecto:

```
Proyecto_Redes_Neuronales/
├── README.md
├── requirements.txt
├── data/
│   └── dataset_sintetico.csv
├── src/
│   ├── modelo_simple.py
│   ├── modelo_multiple.py
│   ├── evaluacion_comparativa.py
│   └── sistema_integrado.py
├── notebooks/
│   ├── analisis_exploratorio.ipynb
│   └── experimentos_modelos.ipynb
├── docs/
│   ├── presentaciones/
│   └── manual_tecnico.md
└── resultados/
    ├── graficos/
    └── metricas/
```

---

## 2. Dataset Sintético (Equipo 1)

El dataset contiene 50 registros generados artificialmente con variables:

* Temperatura (15-40 °C)
* Promoción (0 o 1)
* Fin de semana (0 o 1)
* Ventas (calculadas con fórmula base + ruido)

Archivo generado: `/data/dataset_sintetico.csv`

El análisis exploratorio incluye:

* Gráficos de dispersión
* Histogramas
* Mapa de calor de correlaciones
* Documentación de relaciones entre temperatura y ventas

---

## 3. Modelo de Predicción Simple (Equipo 2)

Archivo: `/src/modelo_simple.py`

Características:

* Red neuronal con 1 neurona (modelo lineal)
* Entrada: Temperatura
* Optimizer: Adam (LR=0.1)
* Pérdida: MSE
* Entrenamiento: 500 épocas

Salida generada:

* `modelos/modelo_simple.h5`
* Gráfica de pérdida opcional

El modelo permite predecir ventas en función de una única variable.

---

## 4. Modelo de Predicción Múltiple (Equipo 3)

Archivo: `/src/modelo_multiple.py`

Características:

* Red neuronal profunda con 3 capas ocultas (100 neuronas cada una)
* Entradas: Temperatura + Promoción + Fin de semana
* Normalización: MinMaxScaler
* División train/test 80/20
* Validación: validation_split=0.2
* Épocas: 100

Salida generada:

* `modelos/modelo_multiple.h5`
* Métricas y curvas de pérdida

Este modelo es más robusto y captura relaciones no lineales.

---

## 5. Evaluación Comparativa de Modelos (Equipo 4)

Archivo: `/src/evaluacion_comparativa.py`

Funciones incluidas:

* Carga de modelos simple y múltiple
* Predicciones sobre el dataset completo
* Cálculo de métricas:

  * MSE
  * MAE
* Generación de gráficas:

  * Predicción vs Real (simple)
  * Predicción vs Real (múltiple)
  * Comparativa simple vs múltiple

Los resultados se guardan en:

* `/resultados/graficos/`
* `/resultados/metricas/`

---

## 6. Sistema Integrado (Equipo 5)

Archivo: `/src/sistema_integrado.py`

Se implementó un dashboard interactivo en **Streamlit** que permite:

* Ingresar temperatura, promoción y fin de semana
* Realizar predicción con ambos modelos
* Mostrar tabla de resultados
* Gráfica comparativa

Características técnicas:

* Carga dinámica de modelos h5
* Normalización de entrada para modelo múltiple
* Visualización con Matplotlib

---

## 7. Requisitos del Sistema

Archivo: `requirements.txt`

Dependencias principales:

* Python 3.10+
* pandas
* numpy
* matplotlib
* seaborn
* tensorflow
* scikit-learn
* streamlit

---

## 8. Ejecución del Proyecto

### 1. Crear entorno virtual

```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. Instalar dependencias

```
pip install -r requirements.txt
```

### 3. Generar dataset (solo una vez)

```
python src/dataset_sintetico.py
```

### 4. Entrenar modelos

```
python src/modelo_simple.py
python src/modelo_multiple.py
```

### 5. Evaluar modelos

```
python src/evaluacion_comparativa.py
```

### 6. Ejecutar dashboard

```
streamlit run src/sistema_integrado.py
```

---

## 9. Análisis General del Sistema

* El modelo simple funciona como referencia base.
* El modelo múltiple demuestra mayor precisión al usar más variables relevantes.
* La comparación gráfica evidencia el comportamiento real vs predicho.
* El dashboard integra todo el sistema de forma práctica y visual.

---

## 10. Conclusiones Técnicas

* La arquitectura modular permite que cada equipo trabaje de forma independiente.
* TensorFlow y Scikit-Learn permiten crear modelos reproducibles.
* La normalización es clave para mejorar el rendimiento del modelo múltiple.
* Streamlit facilita una interfaz funcional sin necesidad de frameworks pesados.

---

## 11. Mejoras Potenciales

* Agregar regularización al modelo múltiple (Dropout / L2)
* Aumentar tamaño del dataset
* Implementar división temporal en caso de series de tiempo
* Guardar métricas en formato JSON
* Exportar dashboard como aplicación web completa

