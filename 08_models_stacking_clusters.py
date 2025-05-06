import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge # Meta-modelo lineal simple
# from sklearn.ensemble import RandomForestRegressor # Otra opción para meta-modelo
import matplotlib.pyplot as plt
import warnings
import gc # Garbage Collector
import joblib # Para guardar/cargar el modelo
import os
import pyarrow.parquet as pq
import config
import time
import logging
import psutil
import sys
from memory_profiler import profile # Para monitoreo de memoria

# Configurar logging detallado
LOG_FILE = 'stacking_model_progress.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Función para registrar uso de memoria
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)  # Convertir a GB
    logger.info(f"Uso de memoria: {mem_gb:.2f} GB")

# Función para registrar progreso con timestamp
def log_progress(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Forzar que la salida se escriba inmediatamente

log_progress("Iniciando script de modelos de stacking por clusters")

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuración Global ---
PARQUET_FILE_PATH = config.GOLD_FEATURES_LGBM_PATH # Ruta al archivo Parquet con features y clusters
TARGET_COL = 'weekly_volume'
CLUSTER_COL = 'cluster_label'
DATE_COL = 'week'
SERIES_ID_COLS = ['establecimiento', 'material']

N_TEST_WEEKS = 12 # Semanas para el conjunto de prueba final de CADA cluster
N_SPLITS_CV = 3 # Reducido de 5 a 3 para ahorrar memoria
# Reemplaza con los mejores parámetros encontrados por Hyperopt/Optuna para el MODELO BASE GLOBAL
# (Usaremos estos como punto de partida para cada cluster base)
BASE_LGBM_PARAMS = {
    # Ejemplo - ¡USA TUS MEJORES PARÁMETROS GLOBALES AQUÍ!
    'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1,
    'boosting_type': 'gbdt', 'seed': 42, 'n_jobs': 4, # Limitar n_jobs a 4 para controlar memoria
    'learning_rate': 0.05, 'num_leaves': 50, 'max_depth': 6, # Reduciendo complejidad
    'min_child_samples': 50, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 3, 'reg_alpha': 0.1, 'reg_lambda': 0.1
    # Asegúrate de incluir todos los parámetros optimizados
}
META_MODEL_TYPE = 'ridge' # Opciones: 'ridge', 'lightgbm', 'rf'

# Parámetros para control de memoria
MAX_ROWS_PER_CLUSTER = 100000  # Máximo de filas a procesar por cluster, None para sin límite
CHUNK_SIZE = 10000  # Tamaño de chunk para procesamiento por lotes

# Rutas de salida (ahora por cluster)
OUTPUT_DIR = 'stacking_per_cluster_output'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'trained_models') # Subdir para modelos
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions') # Subdir para predicciones
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'prediction_plots') # Subdir para plots
METRICS_FILE = os.path.join(OUTPUT_DIR, 'all_clusters_metrics.csv') # Archivo consolidado
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'stacking_checkpoint.txt') # Para guardar progreso

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# --- Funciones Modulares (Adaptadas para trabajar por cluster) ---

def load_and_prepare_cluster_data(parquet_path, cluster_id, target_col, date_col, cluster_col, series_id_cols, n_test_weeks, max_rows=None):
    log_progress(f"\n--- Cargando y Preparando Datos para Cluster {cluster_id} ---")
    try:
        filters = [(cluster_col, '=', cluster_id)]
        
        df_initial = pd.read_parquet(parquet_path, filters=filters, columns=[date_col] + series_id_cols + [target_col], use_threads=True)
        total_rows = len(df_initial)
        log_progress(f"Cluster {cluster_id}: Total de {total_rows} filas encontradas.")
        
        if max_rows is not None and total_rows > max_rows:
            log_progress(f"Limitando a {max_rows} filas (de {total_rows}) para control de memoria.")
            df_initial[date_col] = pd.to_datetime(df_initial[date_col])
            df_initial = df_initial.sort_values(by=[date_col])
            cutoff_date = df_initial[date_col].iloc[-max_rows]
            
            del df_initial
            gc.collect()
            
            df_cluster = pd.read_parquet(
                parquet_path, 
                filters=[(cluster_col, '=', cluster_id)],
                use_threads=True
            )
            
            df_cluster[date_col] = pd.to_datetime(df_cluster[date_col])
            df_cluster = df_cluster[df_cluster[date_col] >= cutoff_date]
            log_progress(f"Filtrado por fecha en pandas: manteniendo filas desde {cutoff_date}")
        else:
            del df_initial
            gc.collect()
            df_cluster = pd.read_parquet(parquet_path, filters=filters, use_threads=True)
        
        log_progress(f"Cluster {cluster_id}: {len(df_cluster)} filas cargadas efectivamente.")
        log_memory_usage()
    except Exception as e:
        logger.error(f"Error cargando datos para cluster {cluster_id}: {e}", exc_info=True)
        return (None,) * 7

    if df_cluster.empty:
        logger.warning(f"No hay datos para cluster {cluster_id}.")
        return (None,) * 7

    if not pd.api.types.is_datetime64_any_dtype(df_cluster[date_col]):
        try:
            df_cluster[date_col] = pd.to_datetime(df_cluster[date_col])
        except Exception as parse_error:
            logger.error(f"Error convirtiendo '{date_col}' a datetime: {parse_error}.")
            return (None,) * 7

    df_cluster = df_cluster.sort_values(by=series_id_cols + [date_col]).reset_index(drop=True)

    initial_rows = len(df_cluster)
    cols_with_potential_nans = [col for col in df_cluster.columns if 'lag_' in col or 'roll_' in col or 'days_since_last_sale' in col]
    cols_to_drop_nans = [col for col in cols_with_potential_nans if col in df_cluster.columns]
    if cols_to_drop_nans:
        df_cluster.dropna(subset=cols_to_drop_nans, inplace=True)
    final_rows = len(df_cluster)
    log_progress(f"Manejo de NaNs: Se eliminaron {initial_rows - final_rows} filas.")
    if df_cluster.empty:
        logger.error("Error: DataFrame del cluster vacío después de eliminar NaNs.")
        return (None,) * 7

    categorical_features_list = [
        'establecimiento', 'material', 'year', 'month',
        'week_of_year', 'has_promo', 'is_covid_period',
        'is_holiday_exact_date', 'is_holiday_in_week'
    ]
    categorical_features_list = [col for col in categorical_features_list if col in df_cluster.columns]
    for col in categorical_features_list:
        if not isinstance(df_cluster[col].dtype, pd.CategoricalDtype):
            if df_cluster[col].isnull().any():
                df_cluster[col] = df_cluster[col].fillna('Missing')
            try:
                df_cluster[col] = df_cluster[col].astype('category')
            except TypeError as e:
                 logger.error(f"Error convirtiendo '{col}' a category: {e}.")
                 return (None,) * 7

    columns_to_exclude = [target_col, date_col, cluster_col, 'last_sale_week']
    features = [col for col in df_cluster.columns if col not in columns_to_exclude and col in df_cluster.columns]
    X = df_cluster[features]
    y = df_cluster[target_col]

    if len(df_cluster) < 2 * n_test_weeks:
         logger.warning(f"Datos insuficientes en cluster {cluster_id} para split con {n_test_weeks} semanas de test.")
         return (None,) * 7

    last_date = df_cluster[date_col].max()
    cutoff_date = last_date - pd.Timedelta(weeks=n_test_weeks)
    cutoff_date = max(cutoff_date, df_cluster[date_col].min())

    train_mask = (df_cluster[date_col] <= cutoff_date)
    test_mask = (df_cluster[date_col] > cutoff_date)

    if train_mask.sum() < N_SPLITS_CV + 1:
         logger.warning(f"Datos de entrenamiento insuficientes ({train_mask.sum()}) después del split en cluster {cluster_id} para {N_SPLITS_CV} splits CV.")
         return (None,) * 7
    if test_mask.sum() == 0:
        logger.warning(f"Conjunto de prueba vacío después del split en cluster {cluster_id}.")
        return (None,) * 7

    X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
    y_train, y_test = y[train_mask].copy(), y[test_mask].copy()

    dates_test = df_cluster.loc[test_mask, date_col].copy()
    ids_test = df_cluster.loc[test_mask, series_id_cols].copy()

    del df_cluster
    gc.collect()
    log_memory_usage()

    log_progress(f"Cluster {cluster_id} Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    return X_train, y_train, X_test, y_test, categorical_features_list, dates_test, ids_test


def train_base_model_and_get_oof_cluster(base_model_params, X_train, y_train, X_test, y_test, categorical_features, tscv, cluster_id):
    log_progress(f"--- Entrenando Modelo Base y OOF para Cluster {cluster_id} ---")
    log_memory_usage()
    
    base_model_params = base_model_params.copy()
    base_model_params['n_estimators'] = min(1000, base_model_params.get('n_estimators', 1000))
    
    oof_preds = np.zeros(len(X_train))
    oof_indices = []
    
    log_progress(f"Usando TimeSeriesSplit con {tscv.n_splits} folds")
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X_train)):
        fold_start_time = time.time()
        log_progress(f"  Procesando fold {fold+1}/{tscv.n_splits} para cluster {cluster_id}")
        
        X_train_fold, X_val_fold = X_train.iloc[train_index].copy(), X_train.iloc[val_index].copy()
        y_train_fold, y_val_fold = y_train.iloc[train_index].copy(), y_train.iloc[val_index].copy()

        if X_val_fold.empty:
             log_progress(f"    Advertencia: Fold {fold+1} de validación vacío. Saltando fold.")
             continue

        try:
            model_fold = lgb.LGBMRegressor(**base_model_params)
            model_fold.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                categorical_feature=categorical_features
            )

            preds_val_fold = model_fold.predict(X_val_fold)
            oof_preds[val_index] = preds_val_fold
            oof_indices.extend(val_index)
            
            fold_time = time.time() - fold_start_time
            log_progress(f"    Fold {fold+1} completado en {fold_time:.2f} segundos")

        except Exception as e:
            logger.error(f"¡ERROR en Fold {fold+1} del modelo base (Cluster {cluster_id})!: {e}", exc_info=True)
            if fold == 0:
                return None, None, None
            log_progress(f"    Continuando con los folds restantes...")
        finally:
            del X_train_fold, X_val_fold, y_train_fold, y_val_fold
            if 'model_fold' in locals(): 
                del model_fold
            gc.collect()
            log_memory_usage()

    valid_oof_indices = sorted(list(set(oof_indices)))
    if len(valid_oof_indices) < len(X_train):
         log_progress(f"Advertencia Cluster {cluster_id}: OOF preds generados solo para {len(valid_oof_indices)}/{len(X_train)} índices.")
         oof_series = pd.Series(np.nan, index=X_train.index, name='oof_pred')
         oof_series.iloc[valid_oof_indices] = oof_preds[valid_oof_indices]
         mean_oof = np.nanmean(oof_preds[valid_oof_indices])
         oof_series.fillna(mean_oof, inplace=True)
         log_progress(f"   NaNs en OOF rellenados con la media ({mean_oof:.2f}).")
    else:
         oof_series = pd.Series(oof_preds, index=X_train.index, name='oof_pred')

    log_progress(f"--- Entrenando Modelo Base Final en todo el Train Set (Cluster {cluster_id}) ---")
    
    try:
        base_model_full = lgb.LGBMRegressor(**base_model_params)
        
        if len(X_train) > CHUNK_SIZE and len(X_test) > CHUNK_SIZE:
            log_progress(f"Conjunto de datos grande, usando subconjuntos para entrenamiento")
            
            np.random.seed(42)
            eval_indices = np.random.choice(len(X_test), min(CHUNK_SIZE, len(X_test)), replace=False)
            X_eval = X_test.iloc[eval_indices]
            y_eval = y_test.iloc[eval_indices]
            
            base_model_full.fit(
                X_train, y_train,
                eval_set=[(X_eval, y_eval)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
                categorical_feature=categorical_features
            )
            
            del X_eval, y_eval
            gc.collect()
        else:
            base_model_full.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                categorical_feature=categorical_features
            )

        log_progress(f"--- Generando Predicciones Base en Test Set (Cluster {cluster_id}) ---")
        log_memory_usage()
        
        if len(X_test) > CHUNK_SIZE:
            log_progress(f"Prediciendo en lotes para reducir uso de memoria")
            n_chunks = int(np.ceil(len(X_test) / CHUNK_SIZE))
            test_preds = np.zeros(len(X_test))
            
            for i in range(n_chunks):
                start_idx = i * CHUNK_SIZE
                end_idx = min((i + 1) * CHUNK_SIZE, len(X_test))
                chunk = X_test.iloc[start_idx:end_idx]
                test_preds[start_idx:end_idx] = base_model_full.predict(chunk)
                del chunk
                gc.collect()
        else:
            test_preds = base_model_full.predict(X_test)

        log_progress(f"Guardando modelo base del cluster {cluster_id}")
        model_base_path = os.path.join(MODELS_DIR, f'lgbm_base_cluster_{cluster_id}.pkl')
        joblib.dump(base_model_full, model_base_path)
        log_progress(f"Modelo base del cluster {cluster_id} guardado.")
        
        gc.collect()
        log_memory_usage()

        return oof_series, test_preds, base_model_full

    except Exception as e:
        logger.error(f"¡ERROR entrenando/prediciendo con el modelo base final (Cluster {cluster_id})!: {e}", exc_info=True)
        return None, None, None


def train_meta_model_cluster(X_train_original, oof_predictions, y_train, cluster_id, meta_model_type='ridge'):
    log_progress(f"--- Entrenando Meta-Modelo ({meta_model_type}) para Cluster {cluster_id} ---")
    log_memory_usage()
    
    X_meta_train = X_train_original.copy()
    X_meta_train['oof_lgbm'] = oof_predictions.values

    if meta_model_type == 'ridge':
        log_progress("Preparando datos para Ridge Regression...")
        categorical_meta = X_meta_train.select_dtypes(include='category').columns
        if not categorical_meta.empty:
            X_meta_train = pd.get_dummies(X_meta_train, columns=categorical_meta, 
                                          dummy_na=False, sparse=True)
        meta_model = Ridge(alpha=1.0, random_state=42)
    elif meta_model_type == 'lightgbm':
        log_progress("Preparando datos para LightGBM...")
        meta_params = {
            'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1,
            'boosting_type': 'gbdt', 'seed': 42, 'n_jobs': 4,
            'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': 5,
            'n_estimators': 200, 'feature_fraction': 0.7
        }
        meta_model = lgb.LGBMRegressor(**meta_params)
    else:
        logger.error(f"Error: Tipo de meta-modelo '{meta_model_type}' no soportado.")
        return None

    try:
        if len(X_meta_train) > CHUNK_SIZE * 2 and meta_model_type == 'ridge':
            log_progress(f"Conjunto de datos grande, utilizando submuestreo para entrenamiento del meta-modelo")
            np.random.seed(42)
            sample_indices = np.random.choice(len(X_meta_train), 
                                             min(CHUNK_SIZE*2, len(X_meta_train)), 
                                             replace=False)
            X_meta_sample = X_meta_train.iloc[sample_indices]
            y_meta_sample = y_train.iloc[sample_indices]
            
            log_progress("Entrenando meta-modelo con datos submuestreados...")
            meta_model.fit(X_meta_sample, y_meta_sample)
            
            del X_meta_sample, y_meta_sample
            gc.collect()
        else:
            if meta_model_type == 'lightgbm':
                categorical_meta_lgbm = list(X_meta_train.select_dtypes(include='category').columns)
                if 'oof_lgbm' in categorical_meta_lgbm: 
                    categorical_meta_lgbm.remove('oof_lgbm')
                log_progress("Entrenando meta-modelo LightGBM...")
                meta_model.fit(X_meta_train, y_train, categorical_feature=categorical_meta_lgbm)
            else:
                if any(X_meta_train.dtypes == 'category'):
                    log_progress("Convirtiendo columnas categóricas restantes...")
                    for col in X_meta_train.select_dtypes(include='category').columns:
                        X_meta_train[col] = X_meta_train[col].cat.codes
                log_progress("Entrenando meta-modelo Ridge...")
                meta_model.fit(X_meta_train, y_train)

        log_progress(f"Meta-modelo para cluster {cluster_id} entrenado con éxito.")
        
        model_meta_path = os.path.join(MODELS_DIR, f'meta_model_cluster_{cluster_id}.pkl')
        joblib.dump(meta_model, model_meta_path)
        log_progress(f"Meta-modelo del cluster {cluster_id} guardado.")
        
        del X_meta_train
        gc.collect()
        log_memory_usage()
        
        return meta_model
    except Exception as e:
        logger.error(f"¡ERROR entrenando el meta-modelo (Cluster {cluster_id})!: {e}", exc_info=True)
        return None


def evaluate_model_cluster(y_true, y_pred, cluster_id, label="Stacking"):
    log_progress(f"--- Evaluando Modelo {label} para Cluster {cluster_id} ---")
    
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mae = mean_absolute_error(y_true_np, y_pred_np)
    
    log_progress(f"  RMSE: {rmse:.4f}")
    log_progress(f"  MAE:  {mae:.4f}")
    
    return {'cluster_id': cluster_id, f'{label}_mae': mae, f'{label}_rmse': rmse}


def plot_predictions_series(dates_test, y_true_test, y_pred_test, estab, material, cluster_id, filepath):
    df_plot = pd.DataFrame({
        'date': pd.to_datetime(dates_test),
        'actual': y_true_test,
        'predicted': y_pred_test
    })
    
    df_plot = df_plot.sort_values(by='date')
    
    plt.figure(figsize=(12, 5))
    plt.plot(df_plot['date'], df_plot['actual'], label='Real', marker='.', linestyle='-', alpha=0.7)
    plt.plot(df_plot['date'], df_plot['predicted'], label=f'Predicción Stacking (Cluster {cluster_id})', 
             marker='x', linestyle='--', alpha=0.7)
    plt.title(f'Predicción vs Real - {estab}/{material} (C{cluster_id})')
    plt.xlabel('Semana')
    plt.ylabel('Volumen Semanal')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    try:
        plt.savefig(filepath, dpi=80, bbox_inches='tight')
    except Exception as e:
        logger.error(f"Error guardando gráfico {filepath}: {e}")
    
    plt.close('all')


# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    start_time_total = time.time()
    
    try:
        log_progress(f"Leyendo columna de clusters '{CLUSTER_COL}' del archivo: {PARQUET_FILE_PATH}")
        pf = pq.ParquetFile(PARQUET_FILE_PATH)
        cluster_labels_df = pf.read(columns=[CLUSTER_COL]).to_pandas()
        unique_clusters = cluster_labels_df[CLUSTER_COL].unique()
        unique_clusters = sorted([c for c in unique_clusters if pd.notna(c)])
        log_progress(f"Encontrados {len(unique_clusters)} clusters únicos: {unique_clusters}")
        del cluster_labels_df
        gc.collect()
        log_memory_usage()
    except Exception as e:
        logger.error(f"Error al leer la columna de clusters del Parquet: {e}", exc_info=True)
        exit()

    all_clusters_final_metrics = []
    clusters_processed = 0
    clusters_skipped = 0
    
    header_metrics = ['cluster_id', 'Stacking_mae', 'Stacking_rmse']
    pd.DataFrame(columns=header_metrics).to_csv(METRICS_FILE, index=False)
    
    last_processed_cluster = -1
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint_data = f.read().strip()
                if checkpoint_data:
                    last_processed_cluster = int(checkpoint_data)
                    log_progress(f"Reanudando desde checkpoint: último cluster procesado = {last_processed_cluster}")
        except Exception as e:
            logger.error(f"Error leyendo checkpoint: {e}")
    
    for cluster_idx, cluster_id in enumerate(unique_clusters):
        if cluster_idx <= last_processed_cluster:
            log_progress(f"Saltando cluster {cluster_id} (ya procesado según checkpoint)")
            clusters_processed += 1
            continue
            
        cluster_start_time = time.time()
        log_progress(f"\n================ Procesando Cluster {cluster_idx+1}/{len(unique_clusters)}: ID = {cluster_id} ================")
        log_memory_usage()
        
        try:
            X_train_c, y_train_c, X_test_c, y_test_c, cat_features_c, dates_test_c, ids_test_c = load_and_prepare_cluster_data(
                PARQUET_FILE_PATH, cluster_id, TARGET_COL, DATE_COL, CLUSTER_COL, 
                SERIES_ID_COLS, N_TEST_WEEKS, max_rows=MAX_ROWS_PER_CLUSTER
            )

            if X_train_c is None:
                log_progress(f"No se pudieron cargar/preparar datos para cluster {cluster_id}. Saltando.")
                clusters_skipped += 1
                continue

            n_splits_actual = min(N_SPLITS_CV, max(2, len(X_train_c) // 100))
            if n_splits_actual != N_SPLITS_CV:
                log_progress(f"Ajustando n_splits para CV a {n_splits_actual} (de {N_SPLITS_CV}) por tamaño del dataset")
            tscv_c = TimeSeriesSplit(n_splits=n_splits_actual)

            oof_preds_c, test_preds_base_c, base_model_c = train_base_model_and_get_oof_cluster(
                BASE_LGBM_PARAMS, X_train_c, y_train_c, X_test_c, y_test_c, cat_features_c, tscv_c, cluster_id
            )

            if oof_preds_c is None:
                log_progress(f"Fallo en el modelo base para cluster {cluster_id}. Saltando.")
                clusters_skipped += 1
                continue

            meta_model_c = train_meta_model_cluster(
                X_train_c, oof_preds_c, y_train_c, cluster_id, meta_model_type=META_MODEL_TYPE
            )

            if meta_model_c is None:
                log_progress(f"Fallo en el meta-modelo para cluster {cluster_id}. Saltando.")
                clusters_skipped += 1
                continue
                
            del base_model_c
            gc.collect()

            log_progress(f"--- Preparando datos de Test para Meta-Modelo (Cluster {cluster_id}) ---")
            X_meta_test_c = X_test_c.copy()
            X_meta_test_c['oof_lgbm'] = test_preds_base_c

            if META_MODEL_TYPE == 'ridge':
                categorical_meta_test_c = X_meta_test_c.select_dtypes(include='category').columns
                if not categorical_meta_test_c.empty:
                     try:
                         meta_train_cols = meta_model_c.feature_names_in_
                         X_meta_test_c = pd.get_dummies(X_meta_test_c, columns=categorical_meta_test_c, 
                                                       dummy_na=False, sparse=True)
                         X_meta_test_c = X_meta_test_c.reindex(columns=meta_train_cols, fill_value=0)
                     except AttributeError as e:
                         logger.warning(f"No se pudieron alinear columnas para Ridge: {e}")
                         X_meta_test_c = pd.get_dummies(X_meta_test_c, columns=categorical_meta_test_c, 
                                                       dummy_na=False, sparse=True)
                if any(X_meta_test_c.dtypes == 'category'):
                     for col in X_meta_test_c.select_dtypes(include='category').columns:
                          X_meta_test_c[col] = X_meta_test_c[col].cat.codes

            log_progress(f"--- Generando Predicciones Finales Stacking (Cluster {cluster_id}) ---")
            log_memory_usage()
            
            try:
                X_meta_test_c.fillna(0, inplace=True)
                
                if len(X_meta_test_c) > CHUNK_SIZE:
                    log_progress(f"Prediciendo por lotes (dataset grande)")
                    n_chunks = int(np.ceil(len(X_meta_test_c) / CHUNK_SIZE))
                    stacking_predictions_c = np.zeros(len(X_meta_test_c))
                    
                    for i in range(n_chunks):
                        start_idx = i * CHUNK_SIZE
                        end_idx = min((i + 1) * CHUNK_SIZE, len(X_meta_test_c))
                        chunk = X_meta_test_c.iloc[start_idx:end_idx]
                        stacking_predictions_c[start_idx:end_idx] = meta_model_c.predict(chunk)
                        del chunk
                        gc.collect()
                else:
                    stacking_predictions_c = meta_model_c.predict(X_meta_test_c)
                
                stacking_predictions_c = np.maximum(0, stacking_predictions_c)
                
            except Exception as e:
                logger.error(f"¡ERROR durante la predicción final del stacking (Cluster {cluster_id})!: {e}", 
                           exc_info=True)
                clusters_skipped += 1
                continue

            final_metrics_c = evaluate_model_cluster(y_test_c.values, stacking_predictions_c, cluster_id)
            all_clusters_final_metrics.append(final_metrics_c)
            pd.DataFrame([final_metrics_c]).to_csv(METRICS_FILE, mode='a', header=False, index=False)

            log_progress(f"Guardando predicciones para cluster {cluster_id}")
            final_preds_df_c = pd.DataFrame({
                'establecimiento': ids_test_c['establecimiento'],
                'material': ids_test_c['material'],
                'week': dates_test_c,
                'actual_volume': y_test_c.values,
                'stacking_predicted_volume': stacking_predictions_c,
                'cluster_label': cluster_id
            })
            
            final_preds_df_c = final_preds_df_c.sort_values(by=SERIES_ID_COLS + ['week']).reset_index(drop=True)
            cluster_pred_file = os.path.join(PREDICTIONS_DIR, f'stacking_preds_cluster_{cluster_id}.csv')
            final_preds_df_c.to_csv(cluster_pred_file, index=False)
            log_progress(f"Predicciones guardadas en: {cluster_pred_file}")

            log_progress(f"--- Generando Gráficos para Series del Cluster {cluster_id} ---")
            unique_series_in_test_c = ids_test_c.drop_duplicates().values.tolist()
            
            max_plots = min(10, len(unique_series_in_test_c))
            if len(unique_series_in_test_c) > max_plots:
                log_progress(f"Limitando a {max_plots} gráficos (de {len(unique_series_in_test_c)} series) para ahorrar recursos")
                np.random.seed(42)
                plot_indices = np.random.choice(len(unique_series_in_test_c), max_plots, replace=False)
                series_to_plot = [unique_series_in_test_c[i] for i in plot_indices]
            else:
                series_to_plot = unique_series_in_test_c
            
            for series_idx, (estab, material) in enumerate(series_to_plot):
                mask = (final_preds_df_c['establecimiento'] == estab) & (final_preds_df_c['material'] == material)
                series_pred_df_c = final_preds_df_c[mask]
                if not series_pred_df_c.empty:
                    plot_filename = os.path.join(PLOTS_DIR, f'pred_vs_actual_{estab}_{material}_cluster{cluster_id}.png')
                    plot_predictions_series(
                        series_pred_df_c['week'],
                        series_pred_df_c['actual_volume'],
                        series_pred_df_c['stacking_predicted_volume'],
                        estab, material, cluster_id, plot_filename
                    )
            
            with open(CHECKPOINT_FILE, 'w') as f:
                f.write(str(cluster_idx))
                
            clusters_processed += 1
            cluster_time = time.time() - cluster_start_time
            log_progress(f"--- Cluster {cluster_id} completado en {cluster_time:.2f} segundos ---")
            
            time_so_far = time.time() - start_time_total
            clusters_remaining = len(unique_clusters) - (cluster_idx + 1)
            avg_time_per_cluster = time_so_far / (cluster_idx + 1)
            est_time_remaining = avg_time_per_cluster * clusters_remaining
            log_progress(f"Progreso: {cluster_idx+1}/{len(unique_clusters)} clusters")
            log_progress(f"Tiempo promedio por cluster: {avg_time_per_cluster:.2f} segundos")
            log_progress(f"Tiempo restante estimado: {est_time_remaining/60:.2f} minutos")
            
            gc.collect()
            log_memory_usage()
            
        except Exception as e:
            logger.error(f"ERROR general procesando cluster {cluster_id}: {e}", exc_info=True)
            clusters_skipped += 1
            with open(CHECKPOINT_FILE, 'w') as f:
                f.write(str(cluster_idx))
                
        finally:
            variables_to_delete = [
                'X_train_c', 'y_train_c', 'X_test_c', 'y_test_c', 'cat_features_c', 
                'dates_test_c', 'ids_test_c', 'oof_preds_c', 'test_preds_base_c',
                'meta_model_c', 'X_meta_test_c', 'stacking_predictions_c', 
                'final_preds_df_c', 'unique_series_in_test_c'
            ]
            for var in variables_to_delete:
                if var in locals():
                    del locals()[var]
            gc.collect()

    total_time = time.time() - start_time_total
    
    log_progress("\n================ Procesamiento de Todos los Clusters Completado ================")
    log_progress(f"Tiempo total: {total_time/60:.2f} minutos")
    log_progress(f"Clusters procesados: {clusters_processed}/{len(unique_clusters)}")
    log_progress(f"Clusters saltados: {clusters_skipped}/{len(unique_clusters)}")
    log_progress(f"Resultados guardados en el directorio base: {OUTPUT_DIR}")
    log_progress(f"Métricas consolidadas por cluster: {METRICS_FILE}")
    log_progress(f"Predicciones individuales por cluster en: {PREDICTIONS_DIR}")
    log_progress(f"Gráficos individuales por serie en: {PLOTS_DIR}")
    log_progress(f"Modelos (base y meta) por cluster en: {MODELS_DIR}")

    try:
        metrics_summary_df = pd.read_csv(METRICS_FILE)
        log_progress("\n--- Resumen Métricas Finales (Promedio sobre Clusters Procesados) ---")
        mean_metrics = metrics_summary_df[['Stacking_mae', 'Stacking_rmse']].mean()
        log_progress(f"MAE promedio: {mean_metrics['Stacking_mae']:.4f}")
        log_progress(f"RMSE promedio: {mean_metrics['Stacking_rmse']:.4f}")
    except Exception as e:
        logger.error(f"\nNo se pudo leer o procesar el archivo de métricas consolidadas: {e}")
        
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

