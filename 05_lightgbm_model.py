import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Usaremos imputer por si acaso
import matplotlib.pyplot as plt
import joblib # Para guardar modelos
import os
import gc # Garbage Collector
import warnings
import lightgbm as lgb

warnings.filterwarnings('ignore', category=FutureWarning) # Para limpiar la salida de sklearn/pandas
warnings.filterwarnings('ignore', category=UserWarning) # Para limpiar la salida de sklearn/pandas

# --- 1. Configuración ---
import config
PARQUET_FILE = config.GOLD_FEATURES_LGBM_PATH # Asegúrate que la ruta sea correcta
OUTPUT_DIR = 'rf_cluster_training_output_cl' # Directorio de salida cambiado
METRICS_FILE = os.path.join(OUTPUT_DIR, 'all_cluster_metrics.csv') # Métricas por cluster
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, 'all_series_predictions.csv') # Predicciones aún por serie
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'prediction_plots_by_series') # Plots por serie
MODELS_DIR = os.path.join(OUTPUT_DIR, 'trained_cluster_models') # Modelos por cluster

# Crear directorios de salida si no existen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_VARIABLE = 'weekly_volume'
SERIES_ID_COLS = ['establecimiento', 'material'] # Siguen siendo identificadores
CLUSTER_COL = 'cluster_label'
DATE_COL = 'week'

# Leer columnas y definir features
parquet_meta = pq.read_metadata(PARQUET_FILE)
ALL_COLS = [col.name for col in parquet_meta.schema]

# Features ahora incluyen establecimiento y material porque son inputs al modelo de cluster
# Excluir target, fecha original, cluster label (ya que filtramos por él)
# y otras columnas no predictivas o que causarían fugas.
FEATURES_COLS = [
    col for col in ALL_COLS
    if col not in [TARGET_VARIABLE, DATE_COL, CLUSTER_COL, 'last_sale_week'] 
]

# Identificar features categóricas y numéricas
# ¡IMPORTANTE! establecimiento y material AHORA son categóricas para el modelo de cluster
CATEGORICAL_FEATURES = ['establecimiento', 'material', 'cluster_label'] # 'cluster_label' aquí es solo para referencia, no se usará como feature directa en el modelo cluster
# Asegurarnos de que solo las que existen en FEATURES_COLS se usen
CATEGORICAL_FEATURES_FOR_MODEL = [f for f in ['establecimiento', 'material'] if f in FEATURES_COLS] 
# Añade otras si las tienes y son categóricas (ej. 'has_promo' si prefieres tratarla así)
# Forzamos las binarias/flags a ser numéricas para simplificar, RF las maneja bien.
NUMERICAL_FEATURES = [col for col in FEATURES_COLS if col not in CATEGORICAL_FEATURES_FOR_MODEL]


# --- 2. Funciones Auxiliares (Iguales que antes) ---

def calculate_mase(y_true_train, y_true_test, y_pred_test):
    """Calcula el Mean Absolute Scaled Error (MASE)."""
    # Asegurar que trabajamos con numpy arrays planos
    y_true_train = np.array(y_true_train).flatten()
    y_true_test = np.array(y_true_test).flatten()
    y_pred_test = np.array(y_pred_test).flatten()

    if len(y_true_train) < 2:
        return np.nan # No se puede calcular el error naive

    # Error absoluto de la predicción naive (valor anterior) en el conjunto de entrenamiento
    # Calculado globalmente sobre el y_train del cluster
    naive_forecast_error_train = np.mean(np.abs(np.diff(y_true_train)))

    if naive_forecast_error_train < 1e-9: # Comparación segura con cero
         # Si el error naive es 0 (serie constante en train), MASE no está definido o es infinito.
         model_mae_test = mean_absolute_error(y_true_test, y_pred_test)
         return np.inf if model_mae_test > 1e-9 else 0.0

    # Error absoluto del modelo en el conjunto de test
    model_mae_test = mean_absolute_error(y_true_test, y_pred_test)

    return model_mae_test / naive_forecast_error_train


def evaluate_model(y_true_train, y_true_test, y_pred_test, label="Cluster"):
    """Calcula un diccionario de métricas agregadas."""
    metrics = {
        f'{label}_mae': mean_absolute_error(y_true_test, y_pred_test),
        f'{label}_rmse': np.sqrt(mean_squared_error(y_true_test, y_pred_test)),
        # Calcular MAPE solo si no hay ceros absolutos en y_true_test para evitar inf
        f'{label}_mape': mean_absolute_percentage_error(y_true_test, y_pred_test) if np.all(np.abs(y_true_test) > 1e-9) else np.nan,
        f'{label}_r2': r2_score(y_true_test, y_pred_test),
        f'{label}_mase': calculate_mase(y_true_train, y_true_test, y_pred_test)
        # Añade aquí TAPE, ROSE, MAS si tienes sus definiciones
    }
    # Reemplazar infinitos en MAPE si ocurren (pueden pasar si y_true_test tiene ceros y no se filtró antes)
    mape_key = f'{label}_mape'
    if mape_key in metrics and np.isinf(metrics[mape_key]):
         metrics[mape_key] = np.nan # O un valor grande representativo
    return metrics


def plot_predictions(dates_test, y_true_test, y_pred_test, estab, material, cluster_id, filepath):
    """Genera y guarda un gráfico de predicciones vs reales para UNA serie."""
    plt.figure(figsize=(15, 6))
    plt.plot(dates_test, y_true_test, label='Real', marker='.', linestyle='-')
    plt.plot(dates_test, y_pred_test, label=f'Predicción (Cluster {cluster_id})', marker='x', linestyle='--')
    plt.title(f'Predicción vs Real - {estab} / {material} (Cluster {cluster_id})')
    plt.xlabel('Semana')
    plt.ylabel('Volumen Semanal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close() # Cierra la figura para liberar memoria

# --- 3. Identificar Clusters Únicos ---
try:
    print(f"Leyendo columna de clusters '{CLUSTER_COL}' del archivo: {PARQUET_FILE}")
    pf = pq.ParquetFile(PARQUET_FILE)
    # Leer solo la columna del cluster para eficiencia
    cluster_labels_df = pf.read(columns=[CLUSTER_COL]).to_pandas()
    unique_clusters = cluster_labels_df[CLUSTER_COL].unique()
    # Remover posibles NaNs si existen
    unique_clusters = [c for c in unique_clusters if pd.notna(c)] 
    print(f"Encontrados {len(unique_clusters)} clusters únicos.")
    del cluster_labels_df # Liberar memoria
    gc.collect()
except Exception as e:
    print(f"Error al leer la columna de clusters del Parquet: {e}")
    # Intentar cargar todo si es necesario (cuidado con la memoria)
    # O manejar el error de otra forma
    exit()


# --- 4. Procesamiento por Cluster (Entrenamiento, CV, HPT, Evaluación) ---

all_cluster_metrics = []
# El archivo de predicciones se inicializa una vez
header_preds = SERIES_ID_COLS + [DATE_COL, 'actual_volume', 'predicted_volume', CLUSTER_COL]
pd.DataFrame(columns=header_preds).to_csv(PREDICTIONS_FILE, index=False)


# Configuración de CV y HPT (igual que antes)
N_SPLITS_CV = 5 
N_ITER_HPT = 10 
SCORING_METRIC_HPT = 'neg_mean_absolute_error' 

# Define el espacio de búsqueda de hiperparámetros para RandomForestRegressor (igual que antes)
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2', 0.7, 1.0] 
}

# Inicializar archivo de métricas de cluster
header_metrics = [CLUSTER_COL] + [f'Cluster_{m}' for m in ['mae', 'rmse', 'mape', 'r2', 'mase']] + ['best_params']
pd.DataFrame(columns=header_metrics).to_csv(METRICS_FILE, index=False)


print(f"\nIniciando procesamiento para {len(unique_clusters)} clusters...")
# Replace tqdm with print statements for progress tracking
for cluster_count, cluster_id in enumerate(unique_clusters):
    print(f"\n--- Procesando Cluster {cluster_count+1}/{len(unique_clusters)}: ID = {cluster_id} ---")
    
    try:
        # 4.1. Cargar datos para el cluster actual
        print(f"Cargando datos para el cluster {cluster_id}...")
        filters = [(CLUSTER_COL, '=', cluster_id)]
        cluster_df = pd.read_parquet(PARQUET_FILE, filters=filters)
        
        # Asegurar orden temporal GLOBAL dentro del cluster para CV
        # Primero por serie, luego por fecha para mantener bloques de series juntos
        cluster_df = cluster_df.sort_values(by=SERIES_ID_COLS + [DATE_COL]).reset_index(drop=True)
        print(f"Cluster {cluster_id}: {len(cluster_df)} filas cargadas.")

        # Mínimo de datos para entrenar/validar (ajusta según necesidad)
        if len(cluster_df) < (N_SPLITS_CV + 2) * 2: # Un umbral más alto para clusters
            print(f"Datos insuficientes para CV en cluster {cluster_id} ({len(cluster_df)} filas). Saltando cluster.")
            continue
            
        # 4.2. Preparar X e y para el cluster
        X_cluster = cluster_df[FEATURES_COLS]
        y_cluster = cluster_df[TARGET_VARIABLE]
        
        # Identificadores y fechas para post-procesamiento (predicciones, gráficos)
        ids_cluster = cluster_df[SERIES_ID_COLS + [CLUSTER_COL]]
        dates_cluster = cluster_df[DATE_COL]

        # 4.3. Preprocesamiento Pipeline
        # Incluye Imputación para numéricas y OneHotEncoding para categóricas
        
        # Pipeline para numéricas: Imputar NaNs (ej. con mediana)
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

        # Pipeline para categóricas: Imputar NaNs (con constante como 'missing') y luego OneHotEncode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ])

        # Crear el preprocesador con ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUMERICAL_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES_FOR_MODEL)
            ], 
            remainder='passthrough' # Mantiene columnas no especificadas si FEATURES_COLS no fue exhaustivo
        )
        
        # 4.4. Separar un conjunto de Test Final (Hold-out) para el CLUSTER
        print(f"Realizando split Train/Test para el cluster {cluster_id}...")
        test_size_ratio = 0.2 # Usar el último 20% como test
        n_rows = len(cluster_df)
        split_point = int(n_rows * (1 - test_size_ratio))

        # Asegurarse que hay suficientes datos para train y test
        if split_point < N_SPLITS_CV + 1 or n_rows - split_point < 1:
             print(f"Datos insuficientes para split Train/Test adecuado en cluster {cluster_id}. Saltando.")
             continue
             
        X_train_val, X_test = X_cluster.iloc[:split_point], X_cluster.iloc[split_point:]
        y_train_val, y_test = y_cluster.iloc[:split_point], y_cluster.iloc[split_point:]
        
        # Guardar IDs y fechas correspondientes al conjunto de test para el guardado final
        ids_test = ids_cluster.iloc[split_point:]
        dates_test = dates_cluster.iloc[split_point:]
        y_true_train_for_mase = y_train_val.values # Para cálculo de MASE agregado

        print(f"Cluster {cluster_id}: Train/Val={len(X_train_val)}, Test={len(X_test)}")

        # 4.5. Configurar TimeSeriesSplit para CV dentro del conjunto Train/Val del CLUSTER
        # test_size debería ser una fracción razonable de los datos de train/val
        cv_test_size = max(1, len(X_train_val) // (N_SPLITS_CV + 1)) 
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV, gap=0, test_size=cv_test_size)

        # 4.6. Configurar y Ejecutar RandomizedSearchCV
        lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1) # LGBM maneja mejor n_jobs=-1

        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('regressor', lgbm) # <-- Usar LGBM
        ])

                # Ajustar el grid de parámetros para que se aplique al paso 'regressor'
        param_dist_pipeline = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.05, 0.1, 0.2],
            'regressor__num_leaves': [20, 31, 40], # Relacionado con max_depth pero no igual
            'regressor__max_depth': [-1, 10, 20], # -1 es sin límite
            'regressor__min_child_samples': [20, 50, 100],
            'regressor__subsample': [0.7, 0.8, 0.9], # Muestreo de filas
            'regressor__colsample_bytree': [0.7, 0.8, 0.9] # Muestreo de features
        }

        # Configura RandomizedSearchCV con el nuevo pipeline y param_dist
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist_pipeline,
            n_iter=N_ITER_HPT, # Puedes empezar con 10-20 para LGBM
            cv=tscv,
            scoring=SCORING_METRIC_HPT,
            n_jobs=1, # El Search sigue secuencial
            refit=True,
            random_state=42,
            verbose=10
        )

        print(f"Iniciando HPT con LightGBM para cluster {cluster_id}...")
        search.fit(X_train_val, y_train_val)
        
        best_model_cluster = search.best_estimator_
        best_params_cluster = search.best_params_
        print(f"Mejores parámetros encontrados para cluster {cluster_id}: {best_params_cluster}")
        
        # 4.7. Predicción en el Conjunto de Test (Hold-out) del CLUSTER
        print(f"Realizando predicciones en el conjunto de test del cluster {cluster_id}...")
        y_pred_test_cluster = best_model_cluster.predict(X_test)
        
        # 4.8. Calcular Métricas AGREGADAS para el CLUSTER
        print(f"Calculando métricas agregadas para el cluster {cluster_id}...")
        cluster_metrics = evaluate_model(y_true_train_for_mase, y_test.values, y_pred_test_cluster, label="Cluster")
        
        # Añadir ID de cluster y parámetros
        metrics_row = {CLUSTER_COL: cluster_id}
        metrics_row.update(cluster_metrics)
        metrics_row['best_params'] = str(best_params_cluster) # Guardar como string
        all_cluster_metrics.append(metrics_row)
        
        # 4.9. Guardar Métricas del Cluster (append)
        pd.DataFrame([metrics_row]).to_csv(METRICS_FILE, mode='a', header=False, index=False)

        # 4.10. Preparar y Guardar Predicciones a Nivel de SERIE (append)
        predictions_df = pd.DataFrame({
            'establecimiento': ids_test['establecimiento'],
            'material': ids_test['material'],
            DATE_COL: dates_test,
            'actual_volume': y_test.values,
            'predicted_volume': y_pred_test_cluster,
            CLUSTER_COL: ids_test[CLUSTER_COL]
        })
        predictions_df.to_csv(PREDICTIONS_FILE, mode='a', header=False, index=False)
        
        # 4.11. Generar y Guardar Gráficos POR SERIE dentro del cluster
        print(f"Generando gráficos para las series del cluster {cluster_id}...")
        unique_series_in_test = ids_test[SERIES_ID_COLS].drop_duplicates().values.tolist()
        
        for series_idx, (estab, material) in enumerate(unique_series_in_test):
             print(f"Generando gráfico {series_idx+1}/{len(unique_series_in_test)} para cluster {cluster_id}")
             # Filtrar las predicciones y reales para esta serie específica del test set
             mask = (predictions_df['establecimiento'] == estab) & (predictions_df['material'] == material)
             series_pred_df = predictions_df[mask]
             
             if not series_pred_df.empty:
                 plot_filename = os.path.join(PLOTS_DIR, f'pred_vs_actual_{estab}_{material}_cluster{cluster_id}.png')
                 plot_predictions(
                     series_pred_df[DATE_COL], 
                     series_pred_df['actual_volume'], 
                     series_pred_df['predicted_volume'], 
                     estab, 
                     material, 
                     cluster_id, 
                     plot_filename
                 )

        # 4.12. Guardar el Modelo Entrenado del CLUSTER
        model_filename = os.path.join(MODELS_DIR, f'rf_model_cluster_{cluster_id}.joblib')
        joblib.dump(best_model_cluster, model_filename)
        print(f"Modelo del cluster {cluster_id} guardado en: {model_filename}")

    except Exception as e:
        print(f"ERROR procesando el cluster {cluster_id}: {e}")
        # Opcional: guardar información del error en el log de métricas
        error_row = {CLUSTER_COL: cluster_id}
        error_row.update({k: 'ERROR' for k in header_metrics if k != CLUSTER_COL and k != 'best_params'})
        error_row['best_params'] = str(e) # Guardar el mensaje de error
        pd.DataFrame([error_row]).to_csv(METRICS_FILE, mode='a', header=False, index=False)

    finally:
        # Liberar memoria explícitamente
        del cluster_df, X_cluster, y_cluster, ids_cluster, dates_cluster
        del X_train_val, X_test, y_train_val, y_test, ids_test, dates_test
        if 'pipeline' in locals(): del pipeline
        if 'search' in locals(): del search
        if 'best_model_cluster' in locals(): del best_model_cluster
        if 'y_pred_test_cluster' in locals(): del y_pred_test_cluster
        if 'predictions_df' in locals(): del predictions_df
        gc.collect()

print("\n--- Proceso Completado ---")
print(f"Resultados guardados en el directorio: {OUTPUT_DIR}")
print(f"Métricas de cluster consolidadas: {METRICS_FILE}")
print(f"Predicciones por serie consolidadas: {PREDICTIONS_FILE}")
print(f"Gráficos individuales por serie en: {PLOTS_DIR}")
print(f"Modelos de cluster entrenados en: {MODELS_DIR}")

# Puedes cargar y analizar los resultados consolidados al final si lo deseas
# metrics_final_df = pd.read_csv(METRICS_FILE)
# print("\nResumen de Métricas (promedio sobre todos los clusters):")
# print(metrics_final_df[[col for col in metrics_final_df.columns if 'Cluster_' in col]].mean(numeric_only=True))