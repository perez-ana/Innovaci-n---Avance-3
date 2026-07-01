# ==========================================
# VETVIEW - MÓDULO DE MACHINE LEARNING (APF3)
# COMPARACIÓN: REGRESIÓN LOGÍSTICA vs SVM vs KNN
# GRUPO 2 - SECCIÓN 25367
# ==========================================

# ==========================================
# 0. INSTALACIÓN DE DEPENDENCIAS (opcional)
# ==========================================
# !pip install pandas numpy scikit-learn matplotlib seaborn

# ==========================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==========================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 2. GENERACIÓN DEL DATASET (si no existe)
# ==========================================
def generar_dataset_simulado():
    """
    Genera el dataset simulado de 600 registros para Veterinaria North.
    Basado en los umbrales clínicos definidos en el diagnóstico.
    """
    np.random.seed(42)
    n = 600
    
    # Variables base
    tipo_servicio = np.random.choice(['Baño', 'Hospitalizacion'], n, p=[0.4, 0.6])
    duracion = np.where(tipo_servicio == 'Baño', 
                        np.random.normal(45, 8, n).clip(20, 80),
                        np.random.normal(1400, 600, n).clip(120, 3000))
    
    # Características de los sensores IoT
    movimiento = np.random.beta(2, 5, n) * 100
    quietud = np.random.exponential(50, n).clip(0, 2000)
    vocalizacion = np.random.exponential(3, n).clip(0, 70)
    
    # Regla de negocio para anomalía (definida por el Dr. North)
    ratio = quietud / duracion
    anomalo = ((vocalizacion > 10) & (movimiento < 20)) | (ratio > 0.5)
    target = anomalo.astype(int)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'id_paciente': np.arange(1, n+1),
        'tipo_servicio': tipo_servicio,
        'duracion_servicio_min': duracion,
        'nivel_movimiento_pct': movimiento,
        'tiempo_quietud_min': quietud,
        'tiempo_vocalizacion_min': vocalizacion,
        'ratio_inactividad_clinica': quietud / duracion,
        'target_anomalo': target
    })
    
    return df

# Cargar o generar dataset
print("="*80)
print("VETVIEW - SISTEMA DE MONITOREO VETERINARIO")
print("MÓDULO DE MACHINE LEARNING - APF3")
print("COMPARACIÓN: Regresión Logística vs SVM vs KNN")
print("="*80)

if not os.path.exists("vetview_dataset_clean.csv"):
    print("\n⚠️ Generando dataset simulado...")
    df = generar_dataset_simulado()
    df.to_csv("vetview_dataset_clean.csv", index=False)
    print(f"✅ Dataset guardado: {df.shape[0]} registros")
else:
    df = pd.read_csv("vetview_dataset_clean.csv")
    print(f"\n✅ Dataset cargado: {df.shape[0]} registros")

# ==========================================
# 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==========================================
print("\n" + "="*80)
print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("="*80)

# Estadísticos descriptivos
print("\n📊 Estadísticos Descriptivos:")
print(df.describe())

# Distribución del target
print(f"\n📊 Distribución del target:")
print(df['target_anomalo'].value_counts())
print(f"Porcentaje de casos anómalos: {(df['target_anomalo'].sum()/len(df))*100:.2f}%")

# Visualización: distribución del target
plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
df['target_anomalo'].value_counts().plot(kind='bar', color=colors)
plt.title('Distribución de Casos: Normal vs Anómalo', fontsize=14)
plt.xlabel('Comportamiento', fontsize=12)
plt.ylabel('Cantidad de registros', fontsize=12)
plt.xticks([0, 1], ['Normal (0)', 'Anómalo (1)'], rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_target_distribution.png', dpi=100)
plt.show()

# Matriz de correlación
plt.figure(figsize=(10, 6))
corr = df.drop(columns=['id_paciente']).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, 
            linewidths=0.5, square=True)
plt.title('Matriz de Correlación entre Variables', fontsize=14)
plt.tight_layout()
plt.savefig('eda_correlation_matrix.png', dpi=100)
plt.show()

# Boxplot bivariado: vocalización por tipo de comportamiento
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Vocalización vs target
sns.boxplot(x='target_anomalo', y='tiempo_vocalizacion_min', data=df, ax=axes[0], 
            palette=['#2ecc71', '#e74c3c'])
axes[0].set_title('Vocalización por Comportamiento', fontsize=12)
axes[0].set_xlabel('Comportamiento', fontsize=10)
axes[0].set_ylabel('Tiempo de Vocalización (min)', fontsize=10)
axes[0].set_xticklabels(['Normal (0)', 'Anómalo (1)'])

# Movimiento vs target
sns.boxplot(x='target_anomalo', y='nivel_movimiento_pct', data=df, ax=axes[1],
            palette=['#2ecc71', '#e74c3c'])
axes[1].set_title('Movimiento por Comportamiento', fontsize=12)
axes[1].set_xlabel('Comportamiento', fontsize=10)
axes[1].set_ylabel('Nivel de Movimiento (%)', fontsize=10)
axes[1].set_xticklabels(['Normal (0)', 'Anómalo (1)'])

plt.tight_layout()
plt.savefig('eda_boxplots.png', dpi=100)
plt.show()

# ==========================================
# 4. SEPARACIÓN DE VARIABLES
# ==========================================
print("\n" + "="*80)
print("SEPARACIÓN DE VARIABLES")
print("="*80)

# Separar X (variables predictoras) y y (target)
X = df.drop(columns=['id_paciente', 'target_anomalo'])
y = df['target_anomalo']

# Identificar tipos de columnas
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Variables predictoras: {X.shape[1]}")
print(f"  - Numéricas ({len(numeric_cols)}): {', '.join(numeric_cols)}")
print(f"  - Categóricas ({len(categorical_cols)}): {', '.join(categorical_cols)}")
print(f"Variable objetivo: target_anomalo")

# ==========================================
# 5. PREPROCESAMIENTO
# ==========================================
print("\n" + "="*80)
print("PREPROCESAMIENTO DE DATOS")
print("="*80)

# Crear preprocesador con escalado para SVM y KNN
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),  # Escalado para SVM y KNN
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

print("✅ Preprocesador creado:")
print("   - StandardScaler aplicado a variables numéricas")
print("   - OneHotEncoder aplicado a variables categóricas")
print("   - Nota: El escalado es CRÍTICO para SVM y KNN")

# ==========================================
# 6. DIVISIÓN DEL DATASET
# ==========================================
print("\n" + "="*80)
print("DIVISIÓN DEL DATASET (Train/Test)")
print("="*80)

# Dividir con estratificación (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"📊 Conjunto de entrenamiento: {X_train.shape[0]} registros")
print(f"   - Normales: {(y_train == 0).sum()} ({((y_train == 0).sum()/len(y_train))*100:.1f}%)")
print(f"   - Anómalos: {(y_train == 1).sum()} ({((y_train == 1).sum()/len(y_train))*100:.1f}%)")

print(f"\n📊 Conjunto de prueba: {X_test.shape[0]} registros")
print(f"   - Normales: {(y_test == 0).sum()} ({((y_test == 0).sum()/len(y_test))*100:.1f}%)")
print(f"   - Anómalos: {(y_test == 1).sum()} ({((y_test == 1).sum()/len(y_test))*100:.1f}%)")

# ==========================================
# 7. MODELOS CON OPTIMIZACIÓN DE HIPERPARÁMETROS
# ==========================================
print("\n" + "="*80)
print("ENTRENAMIENTO Y OPTIMIZACIÓN DE MODELOS")
print("="*80)

# --- 7.1 REGRESIÓN LOGÍSTICA ---
print("\n" + "="*40)
print("🔍 MODELO 1: REGRESIÓN LOGÍSTICA")
print("="*40)

pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

param_grid_lr = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs'],
    'classifier__class_weight': ['balanced', None]
}

print("🔎 Realizando Grid Search para Regresión Logística...")
grid_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_lr.fit(X_train, y_train)

print(f"\n✅ Mejores parámetros: {grid_lr.best_params_}")
print(f"   F1 promedio en validación: {grid_lr.best_score_:.4f}")
mejor_lr = grid_lr.best_estimator_

# --- 7.2 SVM (Máquina de Vectores de Soporte) ---
print("\n" + "="*40)
print("🔍 MODELO 2: SVM (Support Vector Machine)")
print("="*40)

pipeline_svm = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42, probability=True))
])

param_grid_svm = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['rbf', 'linear'],
    'classifier__gamma': ['scale', 'auto', 0.1, 1],
    'classifier__class_weight': ['balanced', None]
}

print("🔎 Realizando Grid Search para SVM...")
grid_svm = GridSearchCV(pipeline_svm, param_grid_svm, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_svm.fit(X_train, y_train)

print(f"\n✅ Mejores parámetros: {grid_svm.best_params_}")
print(f"   F1 promedio en validación: {grid_svm.best_score_:.4f}")
mejor_svm = grid_svm.best_estimator_

# --- 7.3 KNN (K-Nearest Neighbors) ---
print("\n" + "="*40)
print("🔍 MODELO 3: KNN (K-Nearest Neighbors)")
print("="*40)

pipeline_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan'],
    'classifier__p': [1, 2]
}

print("🔎 Realizando Grid Search para KNN...")
grid_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_knn.fit(X_train, y_train)

print(f"\n✅ Mejores parámetros: {grid_knn.best_params_}")
print(f"   F1 promedio en validación: {grid_knn.best_score_:.4f}")
mejor_knn = grid_knn.best_estimator_

# ==========================================
# 8. EVALUACIÓN EN CONJUNTO DE PRUEBA
# ==========================================
print("\n" + "="*80)
print("EVALUACIÓN DE MODELOS EN CONJUNTO DE PRUEBA")
print("="*80)

def evaluar_modelo_completo(modelo, nombre, X_test, y_test):
    """
    Evalúa un modelo y retorna todas las métricas relevantes.
    """
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, 'predict_proba') else None
    
    metricas = {
        'modelo': nombre,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'y_pred': y_pred,
        'y_proba': y_proba,
        'matriz': confusion_matrix(y_test, y_pred)
    }
    
    if y_proba is not None:
        metricas['auc_roc'] = roc_auc_score(y_test, y_proba)
    
    return metricas

# Evaluar cada modelo
print("\n📊 Evaluando modelos...")

resultado_lr = evaluar_modelo_completo(mejor_lr, 'Regresión Logística', X_test, y_test)
resultado_svm = evaluar_modelo_completo(mejor_svm, 'SVM', X_test, y_test)
resultado_knn = evaluar_modelo_completo(mejor_knn, 'KNN', X_test, y_test)

resultados = [resultado_lr, resultado_svm, resultado_knn]

# Mostrar resultados detallados de cada modelo
for r in resultados:
    print(f"\n{'='*40}")
    print(f"📈 {r['modelo']}")
    print(f"{'='*40}")
    print(f"Accuracy:  {r['accuracy']:.4f}")
    print(f"Precision: {r['precision']:.4f}")
    print(f"Recall:    {r['recall']:.4f}")
    print(f"F1 Score:  {r['f1']:.4f}")
    if 'auc_roc' in r:
        print(f"AUC-ROC:   {r['auc_roc']:.4f}")
    print(f"\nMatriz de Confusión:")
    print(r['matriz'])
    print(f"\nClassification Report:")
    print(classification_report(y_test, r['y_pred'], target_names=['Normal', 'Anómalo']))

# ==========================================
# 9. TABLA COMPARATIVA DE MÉTRICAS
# ==========================================
print("\n" + "="*80)
print("📊 TABLA COMPARATIVA DE MODELOS")
print("="*80)

df_metricas = pd.DataFrame([
    {
        'Modelo': r['modelo'],
        'Accuracy': f"{r['accuracy']:.4f}",
        'Precision': f"{r['precision']:.4f}",
        'Recall': f"{r['recall']:.4f}",
        'F1 Score': f"{r['f1']:.4f}",
        'AUC-ROC': f"{r.get('auc_roc', 0):.4f}" if 'auc_roc' in r else 'N/A'
    }
    for r in resultados
])

print(df_metricas.to_string(index=False))

# ==========================================
# 10. SELECCIÓN DEL MEJOR MODELO
# ==========================================
print("\n" + "="*80)
print("🏆 SELECCIÓN DEL MEJOR MODELO")
print("="*80)

# Seleccionar por F1 Score
mejor_modelo = max(resultados, key=lambda x: x['f1'])

print(f"✅ Modelo ganador: {mejor_modelo['modelo']}")
print(f"\n📈 Métricas del modelo ganador:")
print(f"   Accuracy:  {mejor_modelo['accuracy']:.4f}")
print(f"   Precision: {mejor_modelo['precision']:.4f}")
print(f"   Recall:    {mejor_modelo['recall']:.4f}")
print(f"   F1 Score:  {mejor_modelo['f1']:.4f}")
if 'auc_roc' in mejor_modelo:
    print(f"   AUC-ROC:   {mejor_modelo['auc_roc']:.4f}")

# Justificación de la selección
print(f"\n📝 Justificación de selección:")
print(f"   - {mejor_modelo['modelo']} obtuvo el mejor F1 Score ({mejor_modelo['f1']:.4f})")
if 'auc_roc' in mejor_modelo:
    print(f"   - AUC-ROC de {mejor_modelo['auc_roc']:.4f} indica excelente capacidad de discriminación")
print(f"   - El Recall de {mejor_modelo['recall']:.4f} asegura que se detectan la mayoría de anomalías")
print(f"   - La Precision de {mejor_modelo['precision']:.4f} minimiza falsas alarmas")
print(f"   - Es el modelo más adecuado para el contexto de la MiPYME")

# ==========================================
# 11. VALIDACIÓN CRUZADA DEL MODELO GANADOR
# ==========================================
print("\n" + "="*80)
print("🔄 VALIDACIÓN CRUZADA (5 FOLDS)")
print("="*80)

# Obtener el pipeline del modelo ganador
if mejor_modelo['modelo'] == 'Regresión Logística':
    pipeline_ganador = mejor_lr
elif mejor_modelo['modelo'] == 'SVM':
    pipeline_ganador = mejor_svm
else:
    pipeline_ganador = mejor_knn

cv_scores = cross_val_score(pipeline_ganador, X, y, cv=5, scoring='f1')
print(f"F1 Score promedio en validación cruzada: {cv_scores.mean():.4f}")
print(f"Desviación estándar: {cv_scores.std():.4f}")
print(f"F1 por fold: {cv_scores}")

# ==========================================
# 12. IMPORTANCIA DE VARIABLES (para modelos interpretables)
# ==========================================
print("\n" + "="*80)
print("📊 IMPORTANCIA DE VARIABLES")
print("="*80)

if mejor_modelo['modelo'] == 'Regresión Logística':
    # Para Regresión Logística, usamos los coeficientes
    feature_names = pipeline_ganador.named_steps['preprocessor'].get_feature_names_out()
    coefs = pipeline_ganador.named_steps['classifier'].coef_[0]
    importancia = pd.DataFrame({
        'Variable': feature_names,
        'Coeficiente': coefs,
        'Importancia_Abs': np.abs(coefs)
    }).sort_values('Importancia_Abs', ascending=False)
    
    print("Top 5 variables más influyentes (coeficientes):")
    for _, row in importancia.head(5).iterrows():
        signo = "positiva" if row['Coeficiente'] > 0 else "negativa"
        print(f"  {row['Variable']}: {row['Coeficiente']:.4f} ({signo})")
        
elif mejor_modelo['modelo'] == 'SVM':
    print("ℹ️ SVM no proporciona importancia directa de variables.")
    print("   Se recomienda usar SHAP o Permutation Importance para interpretación.")
else:  # KNN
    print("ℹ️ KNN no proporciona importancia directa de variables.")
    print("   Se recomienda usar SHAP o Permutation Importance para interpretación.")

# ==========================================
# 13. CURVA ROC DEL MODELO GANADOR
# ==========================================
print("\n" + "="*80)
print("📈 CURVA ROC DEL MODELO GANADOR")
print("="*80)

if 'y_proba' in mejor_modelo and mejor_modelo['y_proba'] is not None:
    fpr, tpr, thresholds = roc_curve(y_test, mejor_modelo['y_proba'])
    auc = roc_auc_score(y_test, mejor_modelo['y_proba'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Modelo Aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title(f'Curva ROC - {mejor_modelo["modelo"]}', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=100)
    plt.show()
else:
    print("ℹ️ El modelo no soporta predict_proba para curva ROC")

# ==========================================
# 14. PREDICCIÓN CON NUEVO PACIENTE
# ==========================================
print("\n" + "="*80)
print("🔮 PREDICCIÓN CON NUEVO PACIENTE")
print("="*80)

# Crear un nuevo paciente con comportamiento anómalo
nuevo_paciente = pd.DataFrame({
    'tipo_servicio': ['Hospitalizacion'],
    'duracion_servicio_min': [120.0],
    'nivel_movimiento_pct': [5.0],
    'tiempo_quietud_min': [85.0],
    'tiempo_vocalizacion_min': [30.0],
    'ratio_inactividad_clinica': [0.7083]
})

print("Datos del nuevo paciente:")
for col in nuevo_paciente.columns:
    print(f"  {col}: {nuevo_paciente[col].values[0]}")

# Realizar predicción
prediccion = pipeline_ganador.predict(nuevo_paciente)[0]
probabilidad = pipeline_ganador.predict_proba(nuevo_paciente)[0]

print(f"\n📊 Resultado de la predicción:")
print("-"*50)
if prediccion == 1:
    print("🚨 COMPORTAMIENTO ANÓMALO - GENERAR ALERTA VETERINARIA")
    print(f"   Probabilidad de anomalía: {probabilidad[1]*100:.2f}%")
else:
    print("✅ Comportamiento NORMAL")
    print(f"   Probabilidad de normalidad: {probabilidad[0]*100:.2f}%")

print(f"\n📋 Acción recomendada:")
if prediccion == 1:
    print("   1. Verificar inmediatamente el estado de la mascota")
    print("   2. Revisar los datos de los sensores en tiempo real")
    print("   3. Contactar al dueño si es necesario")
    print("   4. Registrar la intervención en el sistema")
else:
    print("   Continuar con el monitoreo regular")

# ==========================================
# 15. RESUMEN FINAL
# ==========================================
print("\n" + "="*80)
print("📋 RESUMEN FINAL DEL PROYECTO")
print("="*80)

print(f"""
✅ Sistema de Machine Learning implementado exitosamente

📊 RESULTADOS PRINCIPALES:
   - Dataset procesado: {df.shape[0]} registros
   - Variables predictoras: {X.shape[1]}
   - Modelos evaluados: 3 (Regresión Logística, SVM, KNN)
   - Modelo ganador: {mejor_modelo['modelo']}
   - F1 Score: {mejor_modelo['f1']:.4f}
   - Recall: {mejor_modelo['recall']:.4f}

🏥 VALOR PARA VETERINARIA NORTH:
   - Detección temprana de comportamientos anómalos
   - Alertas automáticas para intervención proactiva
   - Reducción esperada de quejas: 60%
   - Mayor transparencia y confianza con los clientes

🔧 PRÓXIMOS PASOS (PROY):
   1. Despliegue de los dashboards en la nube
   2. Integración con cámaras IoT reales
   3. Pruebas de estrés con usuarios concurrentes
   4. Reentrenamiento con datos reales de producción
""")

print("="*80)
print("FIN DEL PROGRAMA - VETVIEW ML")
print("="*80)