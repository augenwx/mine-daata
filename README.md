#  Sistema de Predicción de Rendimiento Académico (Matemáticas y Portugués)

Este sistema interactivo e inteligente está diseñado para analizar, segmentar y predecir el éxito académico de los estudiantes en las materias de **Matemáticas** y **Portugués**, basándose en minería de datos y modelos predictivos de Machine Learning. 

El proyecto refactoriza y lleva a producción la lógica desarrollada en el notebook `prediccion_mat_por_unificado .ipynb` a través de una aplicación web moderna e intuitiva con **Streamlit**.

## Capturas de Pantalla del Sistema

A continuación se muestra una vista general de las diferentes secciones del sistema interactivo (dashboard, segmentación, métricas del modelo y el predictor en tiempo real simplificado):

![Capturas del Sistema](src/Diseño%20sin%20título.png)

---

##  ¿De qué trata el sistema?

El sistema procesa dos conjuntos de datos escolares (`student-mat.csv` y `student-por.csv`) que capturan el perfil sociodemográfico, familiar, de salud y los hábitos cotidianos de los alumnos. El pipeline realiza el análisis completo a través de cuatro grandes áreas:

1. **Dashboard y ETL (Extracción, Transformación y Carga):**
   - Limpieza automática de duplicados exactos e imputación de valores nulos (mediana para numéricos, moda para categóricos).
   - Prevención de fuga de datos (**Data Leakage**) mediante la exclusión automática de las notas parciales ($G1$, $G2$) y finales ($G3$).
   - Visualización del balance y tasa de aprobación de cada asignatura activa.

2. **Segmentación de Estudiantes (Clustering):**
   - Agrupamiento en **3 perfiles de comportamiento** mediante **K-Means (K=3)** a partir de 8 hábitos de vida (tiempo de estudio, salidas con amigos, consumo de alcohol, inasistencias, etc.).
   - Visualización espacial interactiva mediante reducción de dimensionalidad en **2D PCA**.
   - Identificación de los 3 perfiles promedio de estudiantes: *Estudiantes Responsables y de Bajo Riesgo*, *Estudiantes Sociales / Recreativos* y *Estudiantes Desconectados / En Riesgo*.

3. **Evaluación de Modelos Predictivos (Clasificación):**
   - Comparación de métricas (*Accuracy, F1-Score, AUC*) entre un clasificador base (**Dummy**), un **Árbol de Decisión** y un **Random Forest**.
   - Curvas **ROC interactiva** y **Matriz de Confusión** en un mapa de calor dinámico para evaluar los aciertos y fallas del modelo.
   - Gráfico de **Importancia de Variables (Feature Importances)** que revela los factores de mayor impacto en el rendimiento escolar.

4. **🔮 Predictor Escolar Inteligente:**
   - Permite ingresar el perfil de un estudiante y calcular instantáneamente la probabilidad de que este apruebe o repruebe la materia activa.
   - Cuenta con dos modalidades de entrada de datos:
     * **Modo Simplificado (Recomendado para Producción):** Requiere únicamente **12 atributos clave** (los cuales representan más del 90% del impacto predictivo), rellenando de forma transparente en el fondo las 18 variables secundarias.
     * **Modo Completo:** Habilita el ingreso exhaustivo de los 30 parámetros sociodemográficos del dataset original.
   - Muestra un medidor gráfico de tipo **Gauge** interactivo y entrega recomendaciones pedagógicas a medida de forma semántica (en verde si aprueba, en rojo con alertas si está en riesgo de reprobación).

---

##  Estructura del Proyecto

```text
├── student-mat.csv      # Dataset original de Matemáticas (delimitador ;)
├── student-por.csv      # Dataset original de Portugués (delimitador ;)
├── requirements.txt      # Dependencias y librerías del proyecto
├── app.py                # Aplicación web interactiva en Streamlit (UI y dashboard)
├── README.md             # Este archivo explicativo de uso
└── src/                  # Carpeta que contiene los módulos de código Python
    ├── __init__.py       # Hace de la carpeta src un módulo ejecutable
    ├── model_pipeline.py # Código modular de carga, ETL, clustering y modelos
    └── predictor.py      # Módulo especializado en la inferencia y predicción en tiempo real
```

---

##  ¿Cómo se usa?

Sigue estos sencillos pasos para instalar y ejecutar el sistema en tu máquina local:

### 1. Requisitos Previos
Asegúrate de tener instalado **Python 3.10 o superior** en tu sistema. Puedes comprobarlo ejecutando en tu terminal:
```bash
python --version
```

### 2. Clonación o Preparación de Archivos
Asegúrate de que los archivos `student-mat.csv` y `student-por.csv` se encuentren en la misma carpeta raíz que los scripts del sistema.

### 3. Instalación de Dependencias
Abre una terminal de PowerShell o CMD en la carpeta raíz del proyecto e instala las librerías necesarias con el siguiente comando:
```powershell
pip install -r requirements.txt
```
*(Las dependencias principales son: `streamlit`, `pandas`, `scikit-learn`, `plotly`, `matplotlib` y `seaborn`)*.

### 4. Ejecución del Servidor Web
Una vez finalizada la instalación de dependencias, inicia la aplicación web interactiva de Streamlit ejecutando:
```powershell
streamlit run app.py
```

### 5. Navegación
Al ejecutar el comando, tu consola desplegará las direcciones de red locales y **se abrirá automáticamente una pestaña en tu navegador web predeterminado** (usualmente en la dirección `http://localhost:8501`).

* **Omitir correo electrónico:** Si en el primer arranque de la consola Streamlit te solicita un correo electrónico con el texto `Email: `, simplemente **presiona la tecla Enter** en tu teclado (dejando el campo en blanco) para saltar el registro e iniciar el sistema.
* **Uso del Dashboard:** Utiliza las pestañas en la parte superior para navegar por las fases del análisis y usa el menú desplegable en la **Barra Lateral Izquierda** para alternar los modelos y gráficos dinámicamente entre **Matemáticas** y **Portugués**.
