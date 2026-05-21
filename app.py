import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from src.model_pipeline import cargar_datos, analizar_asignatura
from src.predictor import predecir_estudiante

# Configuración premium de la página de Streamlit
st.set_page_config(
    page_title="Sistema de Predicción de Rendimiento Académico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para mejorar el diseño visual y dar aspecto premium
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .card-title {
        font-size: 14px;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: 600;
    }
    .card-value {
        font-size: 24px;
        font-weight: 700;
        color: #1F2937;
    }
    .success-box {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        font-size: 18px;
        font-weight: 600;
    }
    .danger-box {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #EF4444;
        font-size: 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CARGA Y CACHÉ DE DATOS
# -------------------------------------------------------------
@st.cache_resource
def obtener_analisis_completo():
    """
    Carga y entrena el pipeline para ambas asignaturas una sola vez por sesión
    y almacena en caché los resultados para máxima rapidez.
    """
    try:
        df_mat, df_por = cargar_datos(".")
        resultados_mat = analizar_asignatura(df_mat, "Matematicas")
        resultados_por = analizar_asignatura(df_por, "Portugues")
        return resultados_mat, resultados_por
    except Exception as e:
        st.error(f"Error al inicializar el pipeline de datos: {e}")
        return None, None

# Cargar los datos y pipelines entrenados
res_mat, res_por = obtener_analisis_completo()

if res_mat is None or res_por is None:
    st.stop()

# -------------------------------------------------------------
# BARRA LATERAL (CONTROLES)
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/clouds/200/000000/education.png", width=150)
st.sidebar.title("Configuración")
st.sidebar.markdown("Seleccione los parámetros globales para visualizar y simular:")

# Selector de asignatura
asignatura_sel = st.sidebar.selectbox(
    "Asignatura a Analizar:",
    options=["Matemáticas", "Portugués"],
    index=0
)

# Definir qué dataset usar en las pestañas en base a la selección
res_activo = res_mat if asignatura_sel == "Matemáticas" else res_por

st.sidebar.markdown("---")
st.sidebar.markdown("**Datos del Dataset Activo:**")
st.sidebar.markdown(f"- **Registros Originales:** {res_activo['shape_original'][0]}")
st.sidebar.markdown(f"- **Duplicados Eliminados:** {res_activo['duplicados_eliminados']}")
st.sidebar.markdown(f"- **Tasa de Aprobación:** {res_activo['balance'].get(1, 0) * 100:.1f}%")
st.sidebar.markdown(f"- **Tasa de Reprobación:** {res_activo['balance'].get(0, 0) * 100:.1f}%")

# -------------------------------------------------------------
# CABECERA DE LA APP
# -------------------------------------------------------------
st.markdown("<div class='main-title'>🎓 Sistema de Predicción de Rendimiento Académico</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Minería de Datos y Modelado Inteligente para Matemáticas y Portugués</div>", unsafe_allow_html=True)

# Pestañas principales
tab_data, tab_cluster, tab_model, tab_predict = st.tabs([
    "📊 Dashboard y ETL",
    "🎯 Segmentación (Clustering)",
    "📈 Modelos y Rendimiento",
    "🔮 Predictor en Tiempo Real"
])

# =============================================================
# TAB 1: DASHBOARD Y ETL
# =============================================================
with tab_data:
    st.header("Análisis Exploratorio y Limpieza de Datos (ETL)")
    st.markdown("""
    En esta sección se puede observar el estado de los datos crudos cargados desde los archivos CSV originales, 
    así como la distribución del balance de clases, que determina el porcentaje de aprobación y reprobación académica 
    ($G3 \\ge 10$ significa Aprobado).
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Balance de la Variable Objetivo (Aprobados vs Reprobados)")
        df_bal = pd.DataFrame({
            'Estado': ['Aprobado (G3 >= 10)', 'Reprobado (G3 < 10)'],
            'Proporción': [res_activo['balance'].get(1, 0), res_activo['balance'].get(0, 0)]
        })
        fig_pie = px.pie(
            df_bal, 
            names='Estado', 
            values='Proporción',
            color='Estado',
            color_discrete_map={'Aprobado (G3 >= 10)': '#10B981', 'Reprobado (G3 < 10)': '#EF4444'},
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=30, b=30, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("Estadísticas del Proceso ETL")
        st.markdown(f"""
        Para asegurar la calidad del modelo y evitar el **data leakage** (fuga de datos), se aplicaron las siguientes transformaciones automatizadas:
        
        * **Eliminación de Duplicados:** Se detectaron y eliminaron **{res_activo['duplicados_eliminados']}** duplicados exactos.
        * **Imputación de Valores Nulos:** Se identificaron **{res_activo['nulos_iniciales']}** valores nulos en el dataset crudo. El sistema realiza automáticamente la imputación utilizando la **mediana** para variables numéricas y la **moda** para categóricas.
        * **Remoción de Data Leakage:** Se excluyeron del entrenamiento las notas de periodos parciales ($G1$ y $G2$), así como la nota final ($G3$), ya que su presencia infla artificialmente las métricas del modelo al estar directamente correlacionadas con la aprobación.
        """)
        
    st.subheader("Muestra de los Datos Procesados")
    st.dataframe(res_activo['df_limpio'].head(10), use_container_width=True)

# =============================================================
# TAB 2: SEGMENTACION Y PERFILES (CLUSTERING)
# =============================================================
with tab_cluster:
    st.header("Segmentación de Estudiantes (Clustering)")
    st.markdown("""
    Utilizando algoritmos de aprendizaje no supervisado, agrupamos a los estudiantes en **3 perfiles o segmentos** basados en sus hábitos y entorno: 
    *tiempo de estudio, tiempo de viaje, fallas previas, tiempo libre, salidas con amigos, consumo de alcohol (semana y fin de semana) e inasistencias*.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Comparativa de Silhouette Scores (K=3)")
        st.markdown(f"""
        El coeficiente Silhouette mide qué tan bien agrupados están los clústeres (-1 a 1):
        * **K-Means (k=3):** `{res_activo['sil_km']:.3f}`
        * **Jerárquico (k=3):** `{res_activo['sil_hc']:.3f}`
        
        Dado que K-Means presenta un coeficiente de Silhouette superior, este algoritmo fue el seleccionado para estructurar los perfiles.
        """)
        
        # Gráfico del silhouette por K
        df_sil = pd.DataFrame({
            'Número de Clústeres (K)': [2, 3, 4, 5, 6],
            'Silhouette Score': res_activo['sil_scores']
        })
        fig_sil = px.line(
            df_sil, 
            x='Número de Clústeres (K)', 
            y='Silhouette Score',
            markers=True,
            title="Evolución de Silhouette por número de Clústeres (K-Means)"
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        
    with col2:
        st.subheader("Visualización 2D mediante PCA")
        st.markdown("Reducción de las 8 variables predictoras de hábitos a 2 componentes principales para proyectar los grupos:")
        
        fig_pca = px.scatter(
            res_activo['df_pca'], 
            x='Componente 1', 
            y='Componente 2', 
            color=res_activo['df_pca']['Perfil'].astype(str),
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={'color': 'Perfil del Estudiante'},
            title="Espacio de Clústeres Proyectado en PCA"
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
    st.subheader("Perfiles Promedio de Estudiantes (Hábitos de Vida)")
    st.markdown("""
    Analizando los centros de cada clúster, podemos catalogar y perfilar a los estudiantes del dataset en 3 comportamientos clave:
    """)
    
    # Mostrar la tabla de perfiles promedio
    df_perfiles_display = res_activo['resumen_perfiles'].copy()
    df_perfiles_display.index = [f"Perfil {i}" for i in df_perfiles_display.index]
    st.table(df_perfiles_display)
    
    st.markdown("""
    **Interpretación General:**
    * **Perfil 0 - Estudiantes Responsables y de Bajo Riesgo:** Alto tiempo de estudio, mínimo consumo de alcohol, casi ninguna inasistencia y fallas previas en cero.
    * **Perfil 1 - Estudiantes Altamente Desconectados / En Riesgo:** Alto nivel de reprobaciones previas (failures), tiempo de viaje escolar elevado y gran número de inasistencias.
    * **Perfil 2 - Estudiantes Sociales / Recreativos:** Alto tiempo libre, salidas muy frecuentes con amigos, y consumos elevados de alcohol (tanto Dalc como Walc).
    """)

# =============================================================
# TAB 3: MODELOS Y RENDIMIENTO (CLASIFICACION)
# =============================================================
with tab_model:
    st.header("Evaluación Comparativa de Modelos Predictivos")
    st.markdown("""
    Comparamos tres clasificadores para predecir si el estudiante aprueba o no: **Dummy (Línea base), Árbol de Decisión y Random Forest**. 
    Los modelos fueron evaluados sobre un conjunto de prueba del 20% que nunca vieron durante el entrenamiento.
    """)
    
    col_metrics, col_roc = st.columns([1, 1])
    
    with col_metrics:
        st.subheader("Tabla Comparativa de Resultados")
        st.dataframe(res_activo['df_resultados'], use_container_width=True)
        
        st.subheader("Métricas Detalladas de Clasificación (Random Forest)")
        m_rf = res_activo['metricas_rf']
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Accuracy", f"{m_rf['accuracy']*100:.1f}%")
            st.metric("Recall (Sensibilidad)", f"{m_rf['recall']*100:.1f}%")
        with col_m2:
            st.metric("F1-Score", f"{m_rf['f1']*100:.1f}%")
            st.metric("Especificidad", f"{m_rf['especificidad']*100:.1f}%")
        with col_m3:
            st.metric("Precision", f"{m_rf['precision']*100:.1f}%")
            
    with col_roc:
        st.subheader("Curvas ROC (Árbol vs Random Forest)")
        fig_roc = go.Figure()
        
        # Graficar Árbol de Decisión
        dt_roc = res_activo['roc_curves_data']['Arbol de Decision']
        fig_roc.add_trace(go.Scatter(
            x=dt_roc['fpr'], y=dt_roc['tpr'],
            mode='lines',
            name=f"Árbol de Decisión (AUC = {dt_roc['auc']:.3f})",
            line=dict(color='#F59E0B', width=2)
        ))
        
        # Graficar Random Forest
        rf_roc = res_activo['roc_curves_data']['Random Forest']
        fig_roc.add_trace(go.Scatter(
            x=rf_roc['fpr'], y=rf_roc['tpr'],
            mode='lines',
            name=f"Random Forest (AUC = {rf_roc['auc']:.3f})",
            line=dict(color='#10B981', width=3)
        ))
        
        # Línea de referencia
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name="Línea Base Aleatoria (AUC = 0.500)",
            line=dict(color='#9CA3AF', dash='dash')
        ))
        
        fig_roc.update_layout(
            xaxis_title="Tasa de Falsos Positivos (FPR)",
            yaxis_title="Tasa de Verdaderos Positivos (TPR)",
            margin=dict(t=30, b=30, l=10, r=10)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    col_cm, col_feat = st.columns([1, 1.2])
    
    with col_cm:
        st.subheader("Matriz de Confusión (Random Forest)")
        cm = res_activo['cm_rf']
        
        # Heatmap interactivo de la matriz
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicción del Modelo", y="Valor Real de Clase"),
            x=['Reprueba (0)', 'Aprueba (1)'],
            y=['Reprueba (0)', 'Aprueba (1)'],
            color_continuous_scale="BuGn"
        )
        fig_cm.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=30, l=10, r=10))
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown(f"""
        * **Verdaderos Positivos (VP): {m_rf['vp']}** estudiantes aprobados predichos correctamente.
        * **Verdaderos Negativos (VN): {m_rf['vn']}** estudiantes reprobados (en riesgo) predichos correctamente.
        * **Falsos Positivos (FP): {m_rf['fp']}** estudiantes que el modelo predijo que aprobarían, pero reprobaron.
        * **Falsos Negativos (FN): {m_rf['fn']}** estudiantes que el modelo predijo que reprobarían, pero terminaron aprobando.
        """)
        
    with col_feat:
        st.subheader("Importancia de Variables en Random Forest")
        st.markdown("Top 15 características más relevantes para determinar el éxito escolar:")
        
        df_imp_top = res_activo['df_importances'].head(15)
        fig_imp = px.bar(
            df_imp_top,
            x='Importancia',
            y='Caracteristica',
            orientation='h',
            color='Importancia',
            color_continuous_scale="Viridis",
            title="Importancia Relativa de Variables"
        )
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, margin=dict(t=30, b=30, l=10, r=10))
        st.plotly_chart(fig_imp, use_container_width=True)

# =============================================================
# TAB 4: PREDICTOR ESCOLAR EN TIEMPO REAL
# =============================================================
with tab_predict:
    st.header(f"🔮 Simulador y Predictor en Tiempo Real — {asignatura_sel}")
    st.markdown("""
    Ingrese los datos del alumno para obtener la predicción del éxito escolar junto con la probabilidad calculada por el modelo de **Random Forest**.
    """)
    
    # Selector de modo del predictor
    modo_predictor = st.radio(
        "Seleccione el modo del predictor:",
        options=["🌟 Simplificado (Recomendado)", "🧪 Completo"],
        horizontal=True,
        help="El modo simplificado solo requiere los 12 atributos más importantes (que representan más del 90% del impacto predictivo), rellenando automáticamente el resto con valores típicos."
    )
    
    # Formulario dinámico según el modo seleccionado
    with st.form("formulario_prediccion"):
        if modo_predictor == "🌟 Simplificado (Recomendado)":
            st.markdown("### 📝 Formulario Simplificado (Solo variables de alto impacto)")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("##### 👤 Información General")
                age = st.slider("Edad:", min_value=15, max_value=22, value=17)
                health = st.slider("Estado de salud actual:", min_value=1, max_value=5, value=5, help="1: Muy malo, 5: Excelente")
                absences = st.number_input("Inasistencias acumuladas a clase:", min_value=0, max_value=93, value=2)
                
            with c2:
                st.markdown("##### 📚 Hábitos de Estudio")
                studytime = st.select_slider("Tiempo de estudio semanal:", options=[1, 2, 3, 4], format_func=lambda x: ["<2 horas", "2-5 horas", "5-10 horas", ">10 horas"][x-1])
                freetime = st.slider("Tiempo libre post-escuela:", min_value=1, max_value=5, value=3, help="1: Muy bajo, 5: Muy alto")
                goout = st.slider("Frecuencia de salidas con amigos:", min_value=1, max_value=5, value=3, help="1: Muy baja, 5: Muy alta")
                
            with c3:
                st.markdown("##### 👥 Entorno Familiar y Alcohol")
                failures = st.selectbox("Materias reprobadas anteriormente:", options=[0, 1, 2, 3])
                Medu = st.select_slider("Educación de la Madre:", options=[0, 1, 2, 3, 4], format_func=lambda x: ["Ninguno", "Primaria básica", "Primaria superior", "Secundaria", "Ed. superior"][x])
                Fedu = st.select_slider("Educación del Padre:", options=[0, 1, 2, 3, 4], format_func=lambda x: ["Ninguno", "Primaria básica", "Primaria superior", "Secundaria", "Ed. superior"][x])
                famrel = st.slider("Calidad de relación familiar:", min_value=1, max_value=5, value=4, help="1: Muy mala, 5: Excelente")
                Dalc = st.slider("Consumo de alcohol entre semana:", min_value=1, max_value=5, value=1)
                Walc = st.slider("Consumo de alcohol en fin de semana:", min_value=1, max_value=5, value=1)
                
            # Variables por defecto para completar las 30 requeridas en la versión simplificada
            datos_estudiante = {
                'school': 'GP',
                'sex': 'F',
                'age': age,
                'address': 'U',
                'famsize': 'GT3',
                'Pstatus': 'T',
                'Medu': Medu,
                'Fedu': Fedu,
                'Mjob': 'other',
                'Fjob': 'other',
                'reason': 'course',
                'guardian': 'mother',
                'traveltime': 1,
                'studytime': studytime,
                'failures': failures,
                'schoolsup': 'no',
                'famsup': 'yes',
                'paid': 'no',
                'activities': 'yes',
                'nursery': 'yes',
                'higher': 'yes',
                'internet': 'yes',
                'romantic': 'no',
                'famrel': famrel,
                'freetime': freetime,
                'goout': goout,
                'Dalc': Dalc,
                'Walc': Walc,
                'health': health,
                'absences': absences
            }
            
        else:
            st.markdown("### 🧪 Formulario Completo (Todas las 30 características)")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.subheader("🏠 Información Familiar y Social")
                school = st.selectbox("Escuela del estudiante:", options=["GP", "MS"], help="GP: Gabriel Pereira, MS: Mousinho da Silveira")
                sex = st.selectbox("Género:", options=["F", "M"], format_func=lambda x: "Femenino" if x == "F" else "Masculino")
                age = st.slider("Edad:", min_value=15, max_value=22, value=17)
                address = st.selectbox("Zona de residencia:", options=["U", "R"], format_func=lambda x: "Urbano" if x == "U" else "Rural")
                famsize = st.selectbox("Tamaño de la familia:", options=["GT3", "LE3"], format_func=lambda x: "Más de 3 miembros" if x == "GT3" else "3 o menos miembros")
                Pstatus = st.selectbox("Estatus de cohabitación de padres:", options=["T", "A"], format_func=lambda x: "Viven juntos" if x == "T" else "Separados")
                Medu = st.select_slider("Nivel educativo de la Madre:", options=[0, 1, 2, 3, 4], format_func=lambda x: ["Ninguno", "Primaria básica", "Primaria superior", "Secundaria", "Educación superior"][x])
                Fedu = st.select_slider("Nivel educativo del Padre:", options=[0, 1, 2, 3, 4], format_func=lambda x: ["Ninguno", "Primaria básica", "Primaria superior", "Secundaria", "Educación superior"][x])
                Mjob = st.selectbox("Trabajo de la Madre:", options=["other", "services", "at_home", "teacher", "health"])
                Fjob = st.selectbox("Trabajo del Padre:", options=["other", "services", "at_home", "teacher", "health"])
                
            with c2:
                st.subheader("📚 Hábitos y Soporte de Estudio")
                reason = st.selectbox("Razón de elección de la escuela:", options=["course", "home", "reputation", "other"])
                guardian = st.selectbox("Tutor legal del estudiante:", options=["mother", "father", "other"])
                traveltime = st.select_slider("Tiempo de viaje escolar:", options=[1, 2, 3, 4], format_func=lambda x: ["<15 min", "15-30 min", "30-60 min", ">60 min"][x-1])
                studytime = st.select_slider("Tiempo de estudio semanal:", options=[1, 2, 3, 4], format_func=lambda x: ["<2 horas", "2-5 horas", "5-10 horas", ">10 horas"][x-1])
                failures = st.selectbox("Fallas previas (Materias reprobadas):", options=[0, 1, 2, 3], help="Cantidad de materias reprobadas en años anteriores")
                schoolsup = st.selectbox("¿Recibe soporte educativo de la escuela?", options=["no", "yes"])
                famsup = st.selectbox("¿Recibe soporte educativo de la familia?", options=["no", "yes"])
                paid = st.selectbox("¿Recibe clases particulares pagadas?", options=["no", "yes"])
                activities = st.selectbox("¿Realiza actividades extracurriculares?", options=["no", "yes"])
                nursery = st.selectbox("¿Asistió a guardería?", options=["no", "yes"])
                
            with c3:
                st.subheader("🍻 Consumo de Alcohol y Salud")
                higher = st.selectbox("¿Desea cursar educación superior?", options=["yes", "no"])
                internet = st.selectbox("¿Tiene acceso a internet en casa?", options=["yes", "no"])
                romantic = st.selectbox("¿Tiene pareja sentimental actualmente?", options=["no", "yes"])
                famrel = st.slider("Calidad de relación familiar:", min_value=1, max_value=5, value=4, help="1: Muy mala, 5: Excelente")
                freetime = st.slider("Tiempo libre post-escuela:", min_value=1, max_value=5, value=3, help="1: Muy bajo, 5: Muy alto")
                goout = st.slider("Frecuencia de salidas con amigos:", min_value=1, max_value=5, value=3, help="1: Muy baja, 5: Muy alta")
                Dalc = st.slider("Consumo de alcohol entre semana:", min_value=1, max_value=5, value=1, help="1: Muy bajo, 5: Muy alto")
                Walc = st.slider("Consumo de alcohol en fin de semana:", min_value=1, max_value=5, value=1, help="1: Muy bajo, 5: Muy alto")
                health = st.slider("Estado de salud actual:", min_value=1, max_value=5, value=5, help="1: Muy malo, 5: Excelente")
                absences = st.number_input("Inasistencias acumuladas:", min_value=0, max_value=93, value=2)
                
            # Recolectar datos en la versión completa
            datos_estudiante = {
                'school': school,
                'sex': sex,
                'age': age,
                'address': address,
                'famsize': famsize,
                'Pstatus': Pstatus,
                'Medu': Medu,
                'Fedu': Fedu,
                'Mjob': Mjob,
                'Fjob': Fjob,
                'reason': reason,
                'guardian': guardian,
                'traveltime': traveltime,
                'studytime': studytime,
                'failures': failures,
                'schoolsup': schoolsup,
                'famsup': famsup,
                'paid': paid,
                'activities': activities,
                'nursery': nursery,
                'higher': higher,
                'internet': internet,
                'romantic': romantic,
                'famrel': famrel,
                'freetime': freetime,
                'goout': goout,
                'Dalc': Dalc,
                'Walc': Walc,
                'health': health,
                'absences': absences
            }
            
        # Botón de envío del formulario
        submit_btn = st.form_submit_button("🔮 Predecir Rendimiento", use_container_width=True)
        
    # Procesar la predicción cuando se hace clic en el botón
    if submit_btn:
        # Obtener pipeline de Random Forest activo
        rf_pipeline = res_activo['modelos']['Random Forest']
        
        # Ejecutar inferencia
        pred, prob = predecir_estudiante(rf_pipeline, datos_estudiante)
        
        # Mostrar el resultado al usuario
        st.subheader("Resultado del Análisis Predictivo")
        
        col_res1, col_res2 = st.columns([1.5, 1])
        
        with col_res1:
            if pred == 1:
                st.markdown(f"""
                <div class='success-box'>
                    ✅ EL ESTUDIANTE APROBARÁ LA ASIGNATURA DE {asignatura_sel.upper()}<br>
                    <span style='font-size: 14px; font-weight: normal; color: #047857;'>
                        Probabilidad estimada de aprobación: {prob*100:.2f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='danger-box'>
                    ⚠️ EL ESTUDIANTE ESTÁ EN RIESGO DE REPROBAR {asignatura_sel.upper()}<br>
                    <span style='font-size: 14px; font-weight: normal; color: #B91C1C;'>
                        Probabilidad de aprobación estimada: {prob*100:.2f}% (Riesgo alto de reprobación)
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("### Recomendaciones Académicas sugeridas:")
            if pred == 1:
                st.markdown("""
                * **Mantener hábitos:** El estudiante posee un perfil estable con buena distribución de tiempo de estudio.
                * **Soporte continuo:** Continuar motivando al alumno y mantener el acceso a internet para investigación autodidacta.
                """)
            else:
                st.markdown("""
                * **Tutoría prioritaria:** Programar sesiones de refuerzo académico personalizadas inmediatamente.
                * **Monitoreo de asistencia:** Controlar activamente sus faltas, ya que las inasistencias elevadas tienen un impacto negativo directo en el rendimiento.
                * **Diálogo familiar:** Involucrar a los padres y al tutor para coordinar una estrategia de estudio semanal mínima de 5 horas.
                """)
                
        with col_res2:
            # Gráfico de indicador gauge para la probabilidad
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilidad de Aprobación", 'font': {'size': 18}},
                number = {'suffix': "%", 'font': {'size': 32}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#10B981" if pred == 1 else "#EF4444"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#FEE2E2'},
                        {'range': [50, 80], 'color': '#FEF3C7'},
                        {'range': [80, 100], 'color': '#D1FAE5'}
                    ]
                }
            ))
            fig_gauge.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
