import pandas as pd
import numpy as np

def predecir_estudiante(rf_pipeline, datos_estudiante):
    """
    Realiza la predicción del rendimiento de un estudiante.
    
    Parámetros
    ----------
    rf_pipeline : Pipeline
        Pipeline entrenado de Random Forest (incluye preprocesador y clasificador).
    datos_estudiante : dict
        Diccionario con las 30 características del estudiante (sin G1, G2, G3).
        
    Retorna
    -------
    prediccion : int
        1 si aprueba (G3 >= 10), 0 si reprueba.
    probabilidad_aprobacion : float
        Probabilidad (0.0 a 1.0) de que el estudiante apruebe.
    """
    # Convertir el diccionario en un DataFrame de una sola fila
    df_input = pd.DataFrame([datos_estudiante])
    
    # El pipeline de sklearn se encarga automáticamente del preprocesamiento
    # (StandardScaler para numéricas y OneHotEncoder para categóricas)
    prediccion = rf_pipeline.predict(df_input)[0]
    probabilidades = rf_pipeline.predict_proba(df_input)[0]
    probabilidad_aprobacion = probabilidades[1]
    
    return int(prediccion), float(probabilidad_aprobacion)
