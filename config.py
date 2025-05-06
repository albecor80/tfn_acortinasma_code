"""
Archivo de configuración del proyecto.

Contiene parámetros globales y configuraciones para diferentes partes del proyecto.
"""

import os
import datetime
from pathlib import Path

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Definir rutas base
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Rutas de archivos
SILVER_VENTAS_PATH = os.path.join(DATA_DIR, "silver", "ventas_silver.parquet")
GOLD_DAILY_PATH = os.path.join(DATA_DIR, "gold", "ventas_daily.parquet")
GOLD_WEEKLY_PATH = os.path.join(DATA_DIR, "gold", "ventas_weekly.parquet")
GOLD_WEEKLY_TRAINING_PATH = os.path.join(DATA_DIR, "gold", "ventas_weekly_training.parquet")
GOLD_WEEKLY_FULL_PATH = os.path.join(DATA_DIR, "ventas_weekly_full.parquet")
# Ruta para modelos
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PROPHET_MODELS_DIR = os.path.join(MODELS_DIR, "prophet")
LIGHTGBM_MODELS_DIR = os.path.join(MODELS_DIR, "lightgbm")
CLUSTER_RESULTS_DIR = os.path.join(DATA_DIR, "results", "clusters")
STACKING_MODEL_DIR = os.path.join(MODELS_DIR, "stacking")
GOLD_FEATURES_FULL_PATH = os.path.join(DATA_DIR, "features_full.parquet")
os.makedirs(STACKING_MODEL_DIR, exist_ok=True)

# Parámetros de procesamiento de datos
TIPOS_A_EXCLUIR = ['E-COMMERCE', 'OTROS', 'KAM']
COLUMNS_TO_REMOVE = ['uom', 'material_name', 'tipo', 'region', 'promo_id', 'promo_flag', 'promo_type', 'promo_mechanics']

# Parámetros para modelos
PROPHET_GROWTH = 'linear'  # 'linear' o 'logistic'
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05
PROPHET_SEASONALITY_PRIOR_SCALE = 10.0
PROPHET_YEARLY_SEASONALITY = 'auto'
PROPHET_WEEKLY_SEASONALITY = 'auto'
PROPHET_DAILY_SEASONALITY = False

# Parámetros para clustering
KMEANS_MAX_CLUSTERS = 10
KMEANS_RANDOM_STATE = 42

# Parámetros para LightGBM
LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'n_estimators': 100,
    'verbose': -1
}

# Parámetros para validación cruzada
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Define the project root directory (assuming the script runs from src or the project root)
# Adjust this if the execution context is different
try:
    # Assumes config.py is in src/
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
     # Handle case where __file__ is not defined (e.g., interactive session)
     # This assumes the current working directory is the project root
    PROJECT_ROOT = Path(os.getcwd())


DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

# Bronze layer paths
BRONZE_DETALLISTAS_PATH = DATA_DIR / "bronze_detallistas.parquet"
BRONZE_VENTAS_PATH = DATA_DIR / "bronze_ventas.parquet"

# Silver layer paths
SILVER_VENTAS_PATH = DATA_DIR / "silver_ventas.parquet"

# Gold layer paths
GOLD_DAILY_PATH = DATA_DIR / "gold_ventas_diarias.parquet"
GOLD_WEEKLY_PATH = DATA_DIR / "gold_ventas_semanales.parquet"
GOLD_MONTHLY_PATH = DATA_DIR / "gold_ventas_mensuales.parquet"
GOLD_WEEKLY_FILTERED_PATH = DATA_DIR / "gold_ventas_semanales_filtered.parquet"
GOLD_WEEKLY_TRAINING_PATH = DATA_DIR / "gold_ventas_semanales_training.parquet"
GOLD_WEEKLY_TRAINING_CLUSTERED_PATH = DATA_DIR / "gold_ventas_semanales_training_clustered.parquet"
GOLD_FEATURES_PATH = DATA_DIR / "gold_features.parquet"
GOLD_FEATURES_LGBM_PATH = DATA_DIR / "gold_features_lgbm.parquet"
GOLD_FEATURES_LGBM_FULL_PATH = DATA_DIR / "gold_features_lgbm_full.parquet"
LGBM_MODEL_OUTPUT_PATH = DATA_DIR / "models/lightgbm/lightgbm_final_model_hyperopt.pkl"
RF_MODEL_OUTPUT_PATH = DATA_DIR / "models/rf/rf_final_model_hyperopt.pkl"
RF_ENCODER_OUTPUT_PATH = DATA_DIR / "models/rf/rf_final_encoder"

# Use relative paths from project root
RAW_DATA_DIR = 'data/raw'
BRONZE_DATA_DIR = 'data/bronze'
SILVER_DATA_DIR = 'data/silver'
GOLD_DATA_DIR = 'data/gold'

SILVER_VENTAS_COLUMNS = [
    'establecimiento',
    'material',
    'calday',
    'promo_id',
    'volume_ap',
    'cantidad_umb',
    'tipo'
]

# Rutas de directorios
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Crear directorios si no existen
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)


# Configuración para el procesamiento de datos
DATA_PROCESSING = {
    # Tipos de transacciones a excluir del análisis
    "exclude_transaction_types": ["INTERNAL_TRANSFER", "ADJUSTMENT"],
    
    # Columnas a eliminar por no ser relevantes para el análisis
    "columns_to_remove": ["notes", "reference_id", "metadata"],
    
    # Configuración para el manejo de valores faltantes
    "fill_na": {
        "amount": 0.0,
        "description": "",
        "category": "UNCATEGORIZED"
    },
    
    # Columnas de fecha para estandarización
    "date_columns": ["transaction_date", "posting_date"],
    
    # Extraer componentes de fecha (año, mes, día, día de la semana)
    "extract_date_components": True,
}

# Configuración para la ingeniería de características
FEATURE_ENGINEERING = {
    # Columnas numéricas a normalizar
    "normalize_columns": ["amount", "balance"],
    
    # Columnas para codificación one-hot
    "one_hot_encode": ["category", "merchant"],
    
    # Máximo número de categorías para one-hot encoding
    "max_categories": 50,
    
    # Ventanas para características de series temporales
    "time_windows": [7, 14, 30, 90],
    
    # Funciones de agregación para ventanas temporales
    "agg_functions": ["sum", "mean", "std", "min", "max"]
}

# Configuración para entrenamiento de modelos
MODEL_CONFIG = {
    # Prophet
    "prophet": {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "daily_seasonality": False,
        "weekly_seasonality": True,
        "yearly_seasonality": True
    },
    
    # LightGBM
    "lightgbm": {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1
    },
    
    # Validación cruzada
    "cv": {
        "n_splits": 5,
        "test_size": 0.2,
        "random_state": 42
    }
}

# Configuración de visualización
VISUALIZATION = {
    "default_figsize": (12, 8),
    "dpi": 100,
    "style": "seaborn-v0_8-whitegrid",
    "palette": "viridis",
    "save_format": "png"
}

