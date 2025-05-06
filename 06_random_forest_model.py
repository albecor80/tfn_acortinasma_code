import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from sklearn.ensemble import RandomForestRegressor # <-- Modelo principal ahora
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import joblib
import os
import gc
import warnings
import time
import datetime
import logging
import psutil
import sys

# Setup detailed logging
LOG_FILE = 'rf_model_progress.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Function to log memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)  # Convert to GB
    logger.info(f"Memory usage: {mem_gb:.2f} GB")

# Function to log progress with timestamp
def log_progress(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Force output to be written immediately

log_progress("Starting Random Forest model training script")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Configuración ---
import config
PARQUET_FILE = config.GOLD_FEATURES_LGBM_PATH # Asegúrate que la ruta sea correcta y contenga los datos
OUTPUT_DIR = 'rf_cluster_training_output_cl' # Directorio de salida específico para RF
METRICS_FILE = os.path.join(OUTPUT_DIR, 'all_cluster_metrics_rf.csv') # Archivo de métricas para RF
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, 'all_series_predictions_rf.csv') # Archivo de predicciones para RF
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'prediction_plots_by_series_rf') # Directorio de plots para RF
MODELS_DIR = os.path.join(OUTPUT_DIR, 'trained_cluster_models_rf') # Directorio de modelos para RF

# Crear directorios de salida si no existen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_VARIABLE = 'weekly_volume'
SERIES_ID_COLS = ['establecimiento', 'material']
CLUSTER_COL = 'cluster_label'
DATE_COL = 'week'

# Leer columnas y definir features (mismo proceso que antes)
try:
    parquet_meta = pq.read_metadata(PARQUET_FILE)
    ALL_COLS = [col.name for col in parquet_meta.schema]
except Exception as e:
    print(f"Error leyendo metadata del archivo Parquet: {e}. Asegúrate que el archivo existe en {PARQUET_FILE}")
    # Considera cargar una muestra si la metadata falla, o detener la ejecución
    # df_sample = pd.read_parquet(PARQUET_FILE, columns=['col1', 'col2']) # Ejemplo
    # ALL_COLS = df_sample.columns.tolist()
    exit()


FEATURES_COLS = [
    col for col in ALL_COLS
    if col not in [TARGET_VARIABLE, DATE_COL, CLUSTER_COL, 'last_sale_week']
]

# Identificar features categóricas y numéricas
CATEGORICAL_FEATURES_FOR_MODEL = [f for f in ['establecimiento', 'material'] if f in FEATURES_COLS]
NUMERICAL_FEATURES = [col for col in FEATURES_COLS if col not in CATEGORICAL_FEATURES_FOR_MODEL]

print(f"Features Numéricas: {NUMERICAL_FEATURES}")
print(f"Features Categóricas para OneHotEncoding: {CATEGORICAL_FEATURES_FOR_MODEL}")


# --- 2. Funciones Auxiliares (Sin cambios) ---

def calculate_mase(y_true_train, y_true_test, y_pred_test):
    """Calcula el Mean Absolute Scaled Error (MASE)."""
    y_true_train = np.array(y_true_train).flatten()
    y_true_test = np.array(y_true_test).flatten()
    y_pred_test = np.array(y_pred_test).flatten()

    if len(y_true_train) < 2:
        return np.nan

    naive_forecast_error_train = np.mean(np.abs(np.diff(y_true_train)))

    if naive_forecast_error_train < 1e-9:
         model_mae_test = mean_absolute_error(y_true_test, y_pred_test)
         return np.inf if model_mae_test > 1e-9 else 0.0

    model_mae_test = mean_absolute_error(y_true_test, y_pred_test)
    return model_mae_test / naive_forecast_error_train


def evaluate_model(y_true_train, y_true_test, y_pred_test, label="Cluster"):
    """Calcula un diccionario de métricas agregadas."""
    metrics = {
        f'{label}_mae': mean_absolute_error(y_true_test, y_pred_test),
        f'{label}_rmse': np.sqrt(mean_squared_error(y_true_test, y_pred_test)),
        f'{label}_mape': mean_absolute_percentage_error(y_true_test, y_pred_test) if np.all(np.abs(y_true_test) > 1e-9) else np.nan,
        f'{label}_r2': r2_score(y_true_test, y_pred_test),
        f'{label}_mase': calculate_mase(y_true_train, y_true_test, y_pred_test)
    }
    mape_key = f'{label}_mape'
    if mape_key in metrics and np.isinf(metrics[mape_key]):
         metrics[mape_key] = np.nan
    return metrics


def plot_predictions(dates_test, y_true_test, y_pred_test, estab, material, cluster_id, filepath):
    """Genera y guarda un gráfico de predicciones vs reales para UNA serie."""
    plt.figure(figsize=(15, 6))
    
    # Convert to pandas Series with DatetimeIndex for proper sorting
    df_plot = pd.DataFrame({
        'date': pd.to_datetime(dates_test),
        'actual': y_true_test,
        'predicted': y_pred_test
    })
    
    # Sort by date to ensure chronological order
    df_plot = df_plot.sort_values('date')
    
    plt.plot(df_plot['date'], df_plot['actual'], label='Real', marker='.', linestyle='-')
    plt.plot(df_plot['date'], df_plot['predicted'], label=f'Predicción RF (Cluster {cluster_id})', marker='x', linestyle='--')
    
    plt.title(f'Predicción RF vs Real - {estab} / {material} (Cluster {cluster_id})')
    plt.xlabel('Semana')
    plt.ylabel('Volumen Semanal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close() # Cierra la figura para liberar memoria

# --- 3. Identificar Clusters Únicos ---
try:
    log_progress(f"Leyendo columna de clusters '{CLUSTER_COL}' del archivo: {PARQUET_FILE}")
    pf = pq.ParquetFile(PARQUET_FILE)
    cluster_labels_df = pf.read(columns=[CLUSTER_COL]).to_pandas()
    unique_clusters = cluster_labels_df[CLUSTER_COL].unique()
    unique_clusters = [c for c in unique_clusters if pd.notna(c)]
    log_progress(f"Encontrados {len(unique_clusters)} clusters únicos.")
    del cluster_labels_df
    gc.collect()
    log_memory_usage()
except Exception as e:
    logger.error(f"Error al leer la columna de clusters del Parquet: {e}")
    exit()


# --- 4. Procesamiento por Cluster (Entrenamiento, CV, HPT, Evaluación) ---

all_cluster_metrics = []
# Inicializar archivo de predicciones (una vez)
header_preds = SERIES_ID_COLS + [DATE_COL, 'actual_volume', 'predicted_volume', CLUSTER_COL]
pd.DataFrame(columns=header_preds).to_csv(PREDICTIONS_FILE, index=False)

# Configuración de CV y HPT
N_SPLITS_CV = 5
N_ITER_HPT = 5 # Puedes ajustar este número (e.g., 20, 50) para una búsqueda más exhaustiva
SCORING_METRIC_HPT = 'neg_mean_absolute_error'

# --- CAMBIO: Define el espacio de búsqueda de hiperparámetros para RandomForestRegressor ---
param_dist = {
    'n_estimators': [50, 100, 200], # Número de árboles
    'max_depth': [None, 10, 20, 30],     # Profundidad máxima
    'min_samples_split': [5, 10],     # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [3, 5],       # Mínimo de muestras por hoja
    'max_features': ['sqrt', 'log2', 0.7] # Número de features a considerar en cada split
                                              # 'sqrt' es un buen default, 1.0 usa todas (puede ser lento)
}
# Ajustar el grid de parámetros para que se aplique al paso 'regressor' del pipeline
param_dist_pipeline = {f'regressor__{key}': val for key, val in param_dist.items()}
# --------------------------------------------------------------------------------------

# Inicializar archivo de métricas de cluster
header_metrics = [CLUSTER_COL] + [f'Cluster_{m}' for m in ['mae', 'rmse', 'mape', 'r2', 'mase']] + ['best_params']
pd.DataFrame(columns=header_metrics).to_csv(METRICS_FILE, index=False)


log_progress(f"\nIniciando procesamiento para {len(unique_clusters)} clusters con RandomForestRegressor...")

# Track overall progress
start_time_total = time.time()
clusters_processed = 0
clusters_skipped = 0

for cluster_count, cluster_id in enumerate(unique_clusters):
    cluster_start_time = time.time()
    log_progress(f"\n--- Procesando Cluster {cluster_count+1}/{len(unique_clusters)}: ID = {cluster_id} ---")

    try:
        # 4.1. Cargar datos para el cluster actual
        log_progress(f"Cargando datos para el cluster {cluster_id}...")
        load_start = time.time()
        filters = [(CLUSTER_COL, '=', cluster_id)]
        cluster_df = pd.read_parquet(PARQUET_FILE, filters=filters)
        cluster_df = cluster_df.sort_values(by=SERIES_ID_COLS + [DATE_COL]).reset_index(drop=True)
        log_progress(f"Cluster {cluster_id}: {len(cluster_df)} filas cargadas en {time.time() - load_start:.2f} segundos.")
        log_memory_usage()

        if len(cluster_df) < (N_SPLITS_CV + 2) * 2:
            log_progress(f"Datos insuficientes para CV en cluster {cluster_id} ({len(cluster_df)} filas). Saltando cluster.")
            clusters_skipped += 1
            continue
        sample_fraction = 0.1 # Empezar con 10%
        # O usar un tamaño fijo: sample_size = 1_000_000
        if len(cluster_df) * sample_fraction > 10000: # Asegurarse de que el muestreo tiene sentido
            log_progress(f"Tomando una muestra del {sample_fraction*100}% de los datos ({int(len(cluster_df) * sample_fraction)} filas) para el cluster {cluster_id}...")
            cluster_df = cluster_df.sample(frac=sample_fraction, random_state=42)
            # O usar tamaño fijo:
            # if len(cluster_df) > sample_size:
            #     log_progress(f"Tomando una muestra de {sample_size} filas para el cluster {cluster_id}...")
            #     cluster_df = cluster_df.sample(n=sample_size, random_state=42)
            log_memory_usage()
        else:
            log_progress(f"No se aplica muestreo al cluster {cluster_id} por ser pequeño.")

        # 4.2. Preparar X e y para el cluster
        log_progress(f"Preparando datos para el cluster {cluster_id}...")
        prep_start = time.time()
        X_cluster = cluster_df[FEATURES_COLS]
        y_cluster = cluster_df[TARGET_VARIABLE]
        ids_cluster = cluster_df[SERIES_ID_COLS + [CLUSTER_COL]]
        dates_cluster = cluster_df[DATE_COL]
        log_progress(f"Datos preparados en {time.time() - prep_start:.2f} segundos.")

        # 4.3. Preprocesamiento Pipeline (Sin cambios, OneHotEncoder es necesario para RF)
        log_progress(f"Configurando pipeline de preprocesamiento...")
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)) # RF no maneja categóricas directamente
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUMERICAL_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES_FOR_MODEL)
            ],
            remainder='passthrough'
        )

        # 4.4. Separar un conjunto de Test Final (Hold-out) para el CLUSTER
        log_progress(f"Realizando split Train/Test para el cluster {cluster_id}...")
        test_size_ratio = 0.2
        n_rows = len(cluster_df)
        split_point = int(n_rows * (1 - test_size_ratio))

        if split_point < N_SPLITS_CV + 1 or n_rows - split_point < 1:
             log_progress(f"Datos insuficientes para split Train/Test adecuado en cluster {cluster_id}. Saltando.")
             clusters_skipped += 1
             continue

        X_train_val, X_test = X_cluster.iloc[:split_point], X_cluster.iloc[split_point:]
        y_train_val, y_test = y_cluster.iloc[:split_point], y_cluster.iloc[split_point:]
        ids_test = ids_cluster.iloc[split_point:]
        dates_test = dates_cluster.iloc[split_point:]
        y_true_train_for_mase = y_train_val.values

        log_progress(f"Cluster {cluster_id}: Train/Val={len(X_train_val)}, Test={len(X_test)}")

        # 4.5. Configurar TimeSeriesSplit para CV dentro del conjunto Train/Val del CLUSTER
        cv_test_size = max(1, len(X_train_val) // (N_SPLITS_CV + 1))
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV, gap=0, test_size=cv_test_size)

        # 4.6. Configurar y Ejecutar RandomizedSearchCV con RandomForestRegressor
        # --- CAMBIO: Instanciar RandomForestRegressor ---
        rf = RandomForestRegressor(random_state=42, n_jobs=-1) # n_jobs=-1 para usar todos los cores en el entrenamiento del RF
        # --------------------------------------------

        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('regressor', rf) # <-- Usar RandomForestRegressor aquí
        ])

        # --- CAMBIO: RandomizedSearchCV usa el param_dist_pipeline de RF ---
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist_pipeline, # <-- Grid de RF
            n_iter=N_ITER_HPT,
            cv=tscv,
            scoring=SCORING_METRIC_HPT,
            n_jobs=1, # <-- Mantener en 1 para evitar paralelización anidada conflictiva
                     # RandomForestRegressor ya usa n_jobs=-1 internamente
            refit=True,
            random_state=42,
            verbose=10 # Puedes ajustar el nivel de verbosidad
        )
        # -----------------------------------------------------------------

        log_progress(f"Iniciando HPT con RandomForestRegressor para cluster {cluster_id}...")
        hpt_start = time.time()
        search.fit(X_train_val, y_train_val)
        log_progress(f"HPT completada para cluster {cluster_id} en {time.time() - hpt_start:.2f} segundos.")
        log_memory_usage()

        best_model_cluster = search.best_estimator_
        best_params_cluster = search.best_params_
        log_progress(f"Mejores parámetros encontrados para cluster {cluster_id}: {best_params_cluster}")

        # 4.7. Predicción en el Conjunto de Test (Hold-out) del CLUSTER
        log_progress(f"Realizando predicciones en el conjunto de test del cluster {cluster_id}...")
        pred_start = time.time()
        y_pred_test_cluster = best_model_cluster.predict(X_test)
        log_progress(f"Predicciones completadas en {time.time() - pred_start:.2f} segundos.")

        # 4.8. Calcular Métricas AGREGADAS para el CLUSTER
        log_progress(f"Calculando métricas agregadas para el cluster {cluster_id}...")
        metrics_start = time.time()
        cluster_metrics = evaluate_model(y_true_train_for_mase, y_test.values, y_pred_test_cluster, label="Cluster")
        metrics_row = {CLUSTER_COL: cluster_id}
        metrics_row.update(cluster_metrics)
        metrics_row['best_params'] = str(best_params_cluster)
        all_cluster_metrics.append(metrics_row)
        log_progress(f"Métricas calculadas en {time.time() - metrics_start:.2f} segundos.")
        
        # Log metrics for easy monitoring
        for key, value in cluster_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                log_progress(f"  {key}: {value:.4f}")

        # 4.9. Guardar Métricas del Cluster (append)
        log_progress(f"Guardando métricas para el cluster {cluster_id}...")
        pd.DataFrame([metrics_row]).to_csv(METRICS_FILE, mode='a', header=False, index=False)

        # 4.10. Preparar y Guardar Predicciones a Nivel de SERIE (append)
        log_progress(f"Guardando predicciones para series en el cluster {cluster_id}...")
        predictions_df = pd.DataFrame({
            'establecimiento': ids_test['establecimiento'],
            'material': ids_test['material'],
            DATE_COL: dates_test,
            'actual_volume': y_test.values,
            'predicted_volume': y_pred_test_cluster,
            CLUSTER_COL: ids_test[CLUSTER_COL]
        })

        # Asegurar que las predicciones estén ordenadas por fecha para cada serie
        predictions_df = predictions_df.sort_values(by=SERIES_ID_COLS + [DATE_COL]).reset_index(drop=True)
        predictions_df.to_csv(PREDICTIONS_FILE, mode='a', header=False, index=False)
        
        # Write a checkpoint file after each cluster
        with open("rf_checkpoint.txt", "w") as f:
            f.write(f"Last completed cluster: {cluster_id} ({cluster_count+1}/{len(unique_clusters)})\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 4.11. Generación de gráficos POR SERIE dentro del cluster (ahora habilitado)
        log_progress(f"Generando gráficos para las series del cluster {cluster_id}...")
        unique_series_in_test = pd.DataFrame(ids_test[SERIES_ID_COLS]).drop_duplicates().values.tolist()

        for series_idx, (estab, material) in enumerate(unique_series_in_test):
            log_progress(f"Graficando serie {series_idx+1}/{len(unique_series_in_test)}...")
            # Filtrar predicciones para esta serie específica
            mask = (predictions_df['establecimiento'] == estab) & (predictions_df['material'] == material)
            series_pred_df = predictions_df[mask]
            
            if not series_pred_df.empty:
                # Asegurar orden cronológico para esta serie
                series_pred_df = series_pred_df.sort_values(by=DATE_COL)
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
        log_progress(f"Guardando modelo para el cluster {cluster_id}...")
        model_filename = os.path.join(MODELS_DIR, f'rf_model_cluster_{cluster_id}.joblib')
        joblib.dump(best_model_cluster, model_filename)

        clusters_processed += 1
        cluster_elapsed = time.time() - cluster_start_time
        log_progress(f"Cluster {cluster_id} completado en {cluster_elapsed:.2f} segundos.")
        
        # Progress summary
        elapsed_total = time.time() - start_time_total
        avg_time_per_cluster = elapsed_total / (cluster_count + 1)
        remaining_clusters = len(unique_clusters) - (cluster_count + 1)
        est_remaining_time = avg_time_per_cluster * remaining_clusters
        log_progress(f"Progreso: {cluster_count+1}/{len(unique_clusters)} clusters")
        log_progress(f"Tiempo promedio por cluster: {avg_time_per_cluster:.2f} segundos")
        log_progress(f"Tiempo restante estimado: {est_remaining_time/60:.2f} minutos")

    except Exception as e:
        logger.error(f"ERROR procesando el cluster {cluster_id}: {e}", exc_info=True)
        # Guardar información del error en el log de métricas
        error_row = {CLUSTER_COL: cluster_id}
        error_row.update({k: 'ERROR' for k in header_metrics if k != CLUSTER_COL and k != 'best_params'})
        error_row['best_params'] = str(e) # Guardar el mensaje de error
        pd.DataFrame([error_row]).to_csv(METRICS_FILE, mode='a', header=False, index=False)

    finally:
        # Liberar memoria explícitamente
        log_progress("Liberando memoria...")
        variables_to_delete = [
            'cluster_df', 'X_cluster', 'y_cluster', 'ids_cluster', 'dates_cluster',
            'X_train_val', 'X_test', 'y_train_val', 'y_test', 'ids_test', 'dates_test',
            'pipeline', 'search', 'best_model_cluster', 'y_pred_test_cluster', 'predictions_df'
        ]
        for var in variables_to_delete:
            if var in locals():
                del locals()[var]
        gc.collect()
        log_memory_usage()

log_progress("\n--- Proceso Completado ---")
log_progress(f"Clusters procesados: {clusters_processed}/{len(unique_clusters)}")
log_progress(f"Clusters saltados: {clusters_skipped}/{len(unique_clusters)}")
log_progress(f"Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos")
log_progress(f"Resultados guardados en el directorio: {OUTPUT_DIR}")
log_progress(f"Métricas de cluster consolidadas: {METRICS_FILE}")
log_progress(f"Predicciones por serie consolidadas: {PREDICTIONS_FILE}")
log_progress(f"Gráficos individuales por serie en: {PLOTS_DIR}")
log_progress(f"Modelos de cluster entrenados en: {MODELS_DIR}")
log_progress(f"Archivo de log: {LOG_FILE}")

# Puedes cargar y analizar los resultados consolidados al final si lo deseas
# metrics_final_df = pd.read_csv(METRICS_FILE)
# log_progress("\nResumen de Métricas (promedio sobre todos los clusters):")
# log_progress(metrics_final_df[[col for col in metrics_final_df.columns if 'Cluster_' in col]].mean(numeric_only=True))