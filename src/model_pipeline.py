import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

def cargar_datos(directorio_base="."):
    """
    Carga los archivos student-mat.csv y student-por.csv del directorio especificado.
    """
    ruta_mat = os.path.join(directorio_base, "student-mat.csv")
    ruta_por = os.path.join(directorio_base, "student-por.csv")
    
    if not os.path.exists(ruta_mat) or not os.path.exists(ruta_por):
        raise FileNotFoundError(f"No se encontraron los archivos CSV en {directorio_base}")
        
    df_mat_raw = pd.read_csv(ruta_mat, sep=';')
    df_por_raw = pd.read_csv(ruta_por, sep=';')
    
    return df_mat_raw, df_por_raw

def analizar_asignatura(df_raw, nombre):
    """
    Ejecuta todo el pipeline (ETL, Clustering, Partición, Modelado) para una asignatura dada.
    Retorna un diccionario completo con todos los objetos, métricas y dataframes resultantes.
    """
    resultados = {'nombre': nombre}
    
    # ─────────────────────────────────────────────────────────────
    # FASE 1: ETL Y LIMPIEZA
    # ─────────────────────────────────────────────────────────────
    df = df_raw.copy()
    resultados['shape_original'] = df.shape
    
    # Eliminar duplicados exactos
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates()
    resultados['duplicados_eliminados'] = n_dup
    
    # Imputación de nulos (por robustez del pipeline)
    n_nulos = df.isnull().sum().sum()
    resultados['nulos_iniciales'] = n_nulos
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    # Variable objetivo: 1=Aprueba (G3>=10), 0=Reprueba
    df['target_campana'] = (df['G3'] >= 10).astype(int)
    balance = df['target_campana'].value_counts(normalize=True).round(3)
    resultados['df_limpio'] = df
    resultados['balance'] = balance
    
    # ─────────────────────────────────────────────────────────────
    # FASE 2: SEGMENTACION — CLUSTERING
    # ─────────────────────────────────────────────────────────────
    cols_comp = ['studytime', 'traveltime', 'failures',
                 'freetime', 'goout', 'Dalc', 'Walc', 'absences']
    X_cluster = df[cols_comp]
    scaler_c = StandardScaler()
    X_scaled = scaler_c.fit_transform(X_cluster)
    
    # Silhouette para K=2..6
    sil_scores = []
    for k in range(2, 7):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, lbl))
        
    # K-Means K=3
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_km = km.fit_predict(X_scaled)
    sil_km = silhouette_score(X_scaled, labels_km)
    
    # Jerárquico K=3
    hc = AgglomerativeClustering(n_clusters=3)
    labels_hc = hc.fit_predict(X_scaled)
    sil_hc = silhouette_score(X_scaled, labels_hc)
    
    # Perfiles obtenidos con K-Means
    df_perf = X_cluster.copy()
    df_perf['Perfil'] = labels_km
    resumen_perf = df_perf.groupby('Perfil').mean().round(2)
    
    # PCA para visualización 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=['Componente 1', 'Componente 2'])
    df_pca['Perfil'] = labels_km
    
    resultados.update({
        'scaler_c': scaler_c,
        'km_model': km,
        'sil_scores': sil_scores,
        'sil_km': sil_km,
        'sil_hc': sil_hc,
        'labels_km': labels_km,
        'labels_hc': labels_hc,
        'resumen_perfiles': resumen_perf,
        'df_pca': df_pca,
        'pca_transformer': pca,
        'cols_comp': cols_comp
    })
    
    # ─────────────────────────────────────────────────────────────
    # FASE 3: PARTICION Y PREPROCESAMIENTO
    # ─────────────────────────────────────────────────────────────
    # Se eliminan G1, G2, G3 para evitar data leakage
    X = df.drop(columns=['G1', 'G2', 'G3', 'target_campana'])
    y = df['target_campana']
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocesador con transformadores de columnas
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])
    
    # Partición 80/20 estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    resultados.update({
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'preprocessor': preprocessor,
        'num_cols': num_cols, 'cat_cols': cat_cols
    })
    
    # ─────────────────────────────────────────────────────────────
    # FASE 4: CLASIFICACION
    # ─────────────────────────────────────────────────────────────
    modelos = {
        'Baseline (Dummy)': Pipeline([
            ('prep', preprocessor),
            ('clf', DummyClassifier(strategy='most_frequent'))
        ]),
        'Arbol de Decision': Pipeline([
            ('prep', preprocessor),
            ('clf', DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'))
        ]),
        'Random Forest': Pipeline([
            ('prep', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'))
        ])
    }
    
    res_modelos = []
    roc_curves_data = {}
    
    for nombre_m, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if nombre_m != 'Baseline (Dummy)' else 0.500
        
        res_modelos.append({
            'Modelo': nombre_m,
            'Accuracy': round(acc, 3),
            'F1-Score': round(f1, 3),
            'AUC': round(auc, 3)
        })
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves_data[nombre_m] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
        
    df_res = pd.DataFrame(res_modelos)
    
    # Matriz de confusión para Random Forest
    rf_pipeline = modelos['Random Forest']
    y_pred_rf = rf_pipeline.predict(X_test)
    y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    tn, fp, fn, tp = cm_rf.ravel()
    
    # Métricas detalladas para Random Forest
    precision_rf = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_rf = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity_rf = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_rf = f1_score(y_test, y_pred_rf)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    # Feature Importances de Random Forest
    rf_clf = rf_pipeline.named_steps['clf']
    cat_encoder = rf_pipeline.named_steps['prep'].named_transformers_['cat']
    cat_feat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    all_feat_names = num_cols + cat_feat_names
    importances = rf_clf.feature_importances_
    
    df_importances = pd.DataFrame({
        'Caracteristica': all_feat_names,
        'Importancia': importances
    }).sort_values(by='Importancia', ascending=False)
    
    # Matriz de confusión para Árbol de Decisión
    dt_pipeline = modelos['Arbol de Decision']
    y_pred_dt = dt_pipeline.predict(X_test)
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    
    resultados.update({
        'modelos': modelos,
        'df_resultados': df_res,
        'roc_curves_data': roc_curves_data,
        'cm_rf': cm_rf,
        'cm_dt': cm_dt,
        'metricas_rf': {
            'vp': tp, 'vn': tn, 'fp': fp, 'fn': fn,
            'accuracy': round(accuracy_rf, 3),
            'precision': round(precision_rf, 3),
            'recall': round(recall_rf, 3),
            'especificidad': round(specificity_rf, 3),
            'f1': round(f1_rf, 3)
        },
        'df_importances': df_importances
    })
    
    return resultados
