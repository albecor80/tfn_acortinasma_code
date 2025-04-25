import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Imports para Hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope # Para asegurar enteros
import matplotlib.pyplot as plt
import warnings
import gc # Garbage Collector
import joblib # Para guardar/cargar el modelo
import config
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuración Global ---
PARQUET_FILE_PATH = config.GOLD_FEATURES_LGBM_PATH
TARGET_COL = 'weekly_volume'
N_TEST_WEEKS = 12 # Semanas para el conjunto de prueba final
N_SPLITS_CV = 5 # Número de divisiones para TimeSeriesSplit en CV
MAX_EVALS_HYPEROPT = 5 # Número de evaluaciones para Hyperopt
MODEL_OUTPUT_PATH = config.LGBM_MODEL_OUTPUT_PATH

# --- Funciones Modulares ---

def load_and_prepare_data(parquet_path, target_col, n_test_weeks):
    """
    Carga datos desde Parquet, realiza preprocesamiento básico, ingeniería de
    features de fecha, maneja NaNs, identifica categóricas y divide en train/test.

    Args:
        parquet_path (str): Ruta al archivo Parquet.
        target_col (str): Nombre de la columna objetivo.
        n_test_weeks (int): Número de semanas finales a usar para el test set.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, categorical_features_list) o (None,)*5 si falla.
    """
    print(f"--- Iniciando Carga y Preparación desde: {parquet_path} ---")
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Forma inicial del DataFrame: {df.shape}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo Parquet en '{parquet_path}'.")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error al cargar el archivo Parquet: {e}")
        return None, None, None, None, None

    # Preprocesamiento Básico
    if 'week' not in df.columns:
        print("Error: Falta la columna 'week'.")
        return None, None, None, None, None
    if not pd.api.types.is_datetime64_any_dtype(df['week']):
        try:
            df['week'] = pd.to_datetime(df['week'])
        except Exception as parse_error:
            print(f"Error al convertir 'week' a datetime: {parse_error}.")
            return None, None, None, None, None

    df = df.sort_values(by=['establecimiento', 'material', 'week']).reset_index(drop=True)
    print("DataFrame ordenado.")

    # Manejo de NaNs (generados por lags/rolling)
    initial_rows = len(df)
    cols_with_potential_nans = [col for col in df.columns if 'lag_' in col or 'roll_' in col or 'days_since_last_sale' in col]
    cols_to_drop_nans = [col for col in cols_with_potential_nans if col in df.columns]
    if cols_to_drop_nans:
        df.dropna(subset=cols_to_drop_nans, inplace=True)
    else:
        print("Advertencia: No se encontraron columnas de lag/roll para dropear NaNs.")
    final_rows = len(df)
    print(f"Manejo de NaNs: Se eliminaron {initial_rows - final_rows} filas.")
    if df.empty:
        print("Error: DataFrame vacío después de eliminar NaNs.")
        return None, None, None, None, None

    # Identificar y Preparar Categóricas
    categorical_features_list = [
        'establecimiento', 'material', 'cluster_label', 'year', 'month',
        'week_of_year', 'has_promo', 'is_covid_period',
        'is_holiday_exact_date', 'is_holiday_in_week'
        # Añadir 'day_of_week' si existe y es relevante
    ]
    categorical_features_list = [col for col in categorical_features_list if col in df.columns]
    print(f"Identificando {len(categorical_features_list)} columnas como categóricas.")
    for col in categorical_features_list:
        if not pd.api.types.is_categorical_dtype(df[col]):
            if df[col].isnull().any():
                df[col] = df[col].fillna('Missing') # Estrategia simple para NaNs en categóricas
            df[col] = df[col].astype('category')
    print(f"Columnas categóricas: {categorical_features_list}")
    # Separar Features (X) y Target (y)
    columns_to_exclude = [target_col, 'week', 'last_sale_week'] # Excluir target y fechas auxiliares
    features = [col for col in df.columns if col not in columns_to_exclude and col in df.columns]
    X = df[features]
    y = df[target_col]
    print(f"Target: '{target_col}'. Features: {X.shape[1]} columnas.")

    # Dividir en Entrenamiento y Prueba (Cronológico)
    cutoff_date = df['week'].max() - pd.Timedelta(weeks=n_test_weeks)
    train_mask = (df['week'] <= cutoff_date)
    test_mask = (df['week'] > cutoff_date)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"División Entrenamiento/Prueba (Test >= {cutoff_date.date()}):")
    print(f"  Tamaño Entrenamiento: {X_train.shape[0]} filas")
    print(f"  Tamaño Prueba:       {X_test.shape[0]} filas")

    if X_test.empty or X_train.empty:
        print("Error: Conjunto de entrenamiento o prueba vacío después de la división.")
        return None, None, None, None, None

    print("--- Carga y Preparación Completada ---")
    return X_train, y_train, X_test, y_test, categorical_features_list


def objective_hyperopt(params, X_train, y_train, categorical_features, tscv):
    """
    Función objetivo para Hyperopt, realiza CV temporal y devuelve el MAE promedio
    (métrica principal a minimizar) y el RMSE promedio (informativo).
    """
    global eval_num # Usa el contador global

    # Asegurar tipos enteros
    for p in ['num_leaves', 'max_depth', 'min_child_samples', 'bagging_freq']:
        if p in params:
            params[p] = int(params[p])

    mae_scores = []
    rmse_scores = [] # Lista para guardar RMSE de cada fold
    print(f"\n  Evaluación {eval_num}: Probando params...") # Más conciso

    for fold, (train_index, val_index) in enumerate(tscv.split(X_train)):
        print(f"    Fold {fold+1}/{tscv.n_splits}: Entrenando...")
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Usar MAE como métrica principal para early stopping y objetivo
        model_fold = lgb.LGBMRegressor(**params, metric='mae') # Especificar métrica aquí también
        model_fold.fit(X_train_fold, y_train_fold,
                       eval_set=[(X_val_fold, y_val_fold)],
                       eval_metric='mae', # Early stopping basado en MAE
                       callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
                       categorical_feature=categorical_features)

        preds_val_fold = model_fold.predict(X_val_fold)
        mae_fold = mean_absolute_error(y_val_fold, preds_val_fold)
        rmse_fold = np.sqrt(mean_squared_error(y_val_fold, preds_val_fold)) # Calcular RMSE
        mae_scores.append(mae_fold)
        rmse_scores.append(rmse_fold) # Guardar RMSE

        del X_train_fold, X_val_fold, y_train_fold, y_val_fold, model_fold
        gc.collect()

    average_mae = np.mean(mae_scores)
    average_rmse = np.mean(rmse_scores) # Calcular RMSE promedio
    print(f"  Evaluación {eval_num} -> MAE Promedio CV: {average_mae:.4f}, RMSE Promedio CV: {average_rmse:.4f}")
    eval_num += 1 # Incrementar contador aquí

    # Hyperopt minimiza 'loss'. Añadimos 'rmse' para información en los trials.
    return {'loss': average_mae, 'rmse': average_rmse, 'status': STATUS_OK, 'params': params}


def tune_hyperparameters(X_train, y_train, categorical_features, tscv, max_evals):
    """
    Realiza la optimización de hiperparámetros usando Hyperopt y TimeSeriesSplit.

    Args:
        X_train, y_train: Datos de entrenamiento.
        categorical_features (list): Lista de nombres de columnas categóricas.
        tscv: Objeto TimeSeriesSplit configurado.
        max_evals (int): Número de evaluaciones para Hyperopt.

    Returns:
        tuple: (best_params_dict, best_mae, best_rmse)
               Diccionario con los mejores hiperparámetros, el mejor MAE promedio
               y el RMSE promedio correspondiente a ese mejor MAE.
    """
    print(f"\n--- Iniciando Optimización de Hiperparámetros (Hyperopt, {max_evals} evals) ---")
    global eval_num # Reiniciar contador global para esta ejecución
    eval_num = 1

    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
        'num_leaves': scope.int(hp.quniform('num_leaves', 20, 300, 1)),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 12, 1)),
        'min_child_samples': scope.int(hp.quniform('min_child_samples', 5, 100, 1)),
        'feature_fraction': hp.uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': scope.int(hp.quniform('bagging_freq', 1, 7, 1)),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(10.0)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10.0)),
        # Parámetros fijos
        'objective': 'regression_l1', # Objetivo principal sigue siendo MAE
        'verbosity': -1,
        'boosting_type': 'gbdt', 'n_estimators': 1000, 'seed': 42, 'n_jobs': -1
        # 'metric' se especifica en objective_hyperopt y fit
    }

    trials = Trials()
    # Usar lambda para pasar argumentos adicionales a la función objetivo
    objective_with_args = lambda params: objective_hyperopt(params, X_train, y_train, categorical_features, tscv)

    best_hyperparams_raw = fmin(fn=objective_with_args,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=max_evals,
                                trials=trials,
                                rstate=np.random.default_rng(42))

    # Obtener el diccionario completo del mejor trial (el que minimizó 'loss' = MAE)
    best_trial = trials.best_trial
    best_mae = best_trial['result']['loss']
    best_rmse = best_trial['result']['rmse'] # Obtener el RMSE de ese mejor trial
    best_params_dict = best_trial['result']['params']

    # Asegurar tipos correctos finales
    for p in ['num_leaves', 'max_depth', 'min_child_samples', 'bagging_freq']:
         if p in best_params_dict:
               best_params_dict[p] = int(best_params_dict[p])

    print("\n--- Optimización Completada ---")
    print(f"Mejor MAE promedio en CV: {best_mae:.4f} (RMSE correspondiente: {best_rmse:.4f})") # Mostrar ambos
    print("Mejores Hiperparámetros encontrados (para MAE):")
    # Imprimir de forma legible
    for key, value in best_params_dict.items():
        # No imprimir los parámetros fijos que añadimos nosotros o métricas
        if key not in ['objective', 'metric', 'verbosity', 'boosting_type', 'seed', 'n_jobs', 'n_estimators']:
             print(f"  {key}: {value}")

    return best_params_dict, best_mae, best_rmse


def train_final_model(X_train, y_train, X_test, y_test, best_params, categorical_features):
    """
    Entrena el modelo LightGBM final con los mejores parámetros encontrados.
    Monitoriza MAE y RMSE para early stopping.

    Args:
        X_train, y_train: Datos completos de entrenamiento.
        X_test, y_test: Datos de prueba (usados para early stopping).
        best_params (dict): Diccionario con los mejores hiperparámetros.
        categorical_features (list): Lista de columnas categóricas.

    Returns:
        lightgbm.LGBMRegressor: El modelo final entrenado.
    """
    print("\n--- Entrenando Modelo Final ---")
    final_params = {
        'objective': 'regression_l1', # Mantener objetivo MAE si es el principal
        'metric': ['mae', 'rmse'], # Añadir ambas métricas para monitoreo
        'verbosity': -1,
        'boosting_type': 'gbdt', 'seed': 42, 'n_jobs': -1,
        'n_estimators': 2000 # Número alto, early stopping ajustará
    }
    final_params.update(best_params) # Combinar con los mejores encontrados

    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    eval_metric=['mae', 'rmse'], # Evaluar ambas métricas
                    callbacks=[lgb.early_stopping(stopping_rounds=5, verbose=True)], # Early stopping basado en la primera métrica (mae)
                    categorical_feature=categorical_features)

    best_iteration = final_model.best_iteration_
    print(f"Entrenamiento final completado. Mejor iteración (basada en MAE): {best_iteration}")
    return final_model


def evaluate_model(model, X_test, y_test):
    """Calcula y muestra métricas de evaluación en el conjunto de prueba."""
    print("\n--- Evaluando Modelo Final en Conjunto de Prueba ---")
    predictions = model.predict(X_test)
    # Opcional: predictions[predictions < 0] = 0

    final_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    final_mae = mean_absolute_error(y_test, predictions)

    print(f"RMSE: {final_rmse:.4f}") # Ya se calculaba
    print(f"MAE:  {final_mae:.4f}") # Ya se calculaba
    print(f"Volumen promedio real (prueba):   {y_test.mean():.2f}")
    print(f"Volumen promedio predicho (prueba): {predictions.mean():.2f}")


def plot_feature_importance(model, top_n=20):
    """Genera y muestra el gráfico de importancia de features."""
    print(f"\n--- Importancia de Features (Top {top_n}) ---")
    try:
        # Verificar si el modelo está entrenado
        if not hasattr(model, 'feature_importances_'):
             print("El modelo no parece estar entrenado. No se puede mostrar la importancia.")
             return

        feature_importances = pd.DataFrame({'feature': model.feature_name_,
                                           'importance': model.feature_importances_})
        feature_importances = feature_importances.sort_values('importance', ascending=False).head(top_n)
        print(feature_importances)

        plt.figure(figsize=(10, max(6, top_n * 0.3))) # Ajustar altura
        lgb.plot_importance(model, max_num_features=top_n, height=0.5)
        plt.title(f"Importancia de Features (Top {top_n})")
        plt.tight_layout()
        plt.show()
    except Exception as plot_err:
        print(f"No se pudo generar el gráfico de importancia: {plot_err}")


# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    


    # Load model from file
    model = joblib.load(config.LIGHTGBM_MODELS_DIR + "/lightgbm_final_model_hyperopt.pkl")
    # Load 5 series from gold_weekly_training.parquet
    df = pd.read_parquet(config.GOLD_WEEKLY_TRAINING_PATH)
    # Get 5 random series
    series = df.sample(n=5)
    
    

    # Predict with model
    predictions = model.predict(series)
    print(predictions)
    # Plot predictions
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predictions')
    plt.plot(series['weekly_volume'], label='Actual')
    plt.legend()
    plt.show()


    '''
    # 1. Cargar y Preparar Datos
    X_train, y_train, X_test, y_test, categorical_features = load_and_prepare_data(
        PARQUET_FILE_PATH, TARGET_COL, N_TEST_WEEKS
    )

    if X_train is not None: # Continuar solo si la carga fue exitosa
        # 2. Configurar Validación Cruzada
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

        # 3. Optimizar Hiperparámetros
        best_params, best_cv_mae, best_cv_rmse = tune_hyperparameters( # Capturar RMSE también
            X_train, y_train, categorical_features, tscv, MAX_EVALS_HYPEROPT
        )

        # 4. Entrenar Modelo Final
        final_model = train_final_model(
            X_train, y_train, X_test, y_test, best_params, categorical_features
        )

        # 5. Evaluar Modelo Final
        evaluate_model(final_model, X_test, y_test)

        # 6. Mostrar Importancia de Features
        plot_feature_importance(final_model)

        # 7. Guardar Modelo (Opcional)
        try:
            joblib.dump(final_model, MODEL_OUTPUT_PATH)
            print(f"\nModelo final guardado en: {MODEL_OUTPUT_PATH}")
        except Exception as save_err:
            print(f"\nError al guardar el modelo: {save_err}")

    else:
        print("\nEl proceso no pudo continuar debido a errores en la carga/preparación de datos.")

    print("\nProceso completo.")


    '''
