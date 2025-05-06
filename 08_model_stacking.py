import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge # Meta-modelo lineal simple
# from sklearn.ensemble import RandomForestRegressor # Otra opción para meta-modelo
import matplotlib.pyplot as plt
import warnings
import gc # Garbage Collector
import joblib # Para guardar/cargar el modelo
import os
import config
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuración Global ---
PARQUET_FILE_PATH = config.GOLD_WEEKLY_TRAINING_CLUSTERED_PATH # Ruta al archivo Parquet con features
TARGET_COL = 'weekly_volume'
N_TEST_WEEKS = 12 # Semanas para el conjunto de prueba final
N_SPLITS_CV = 5 # Número de divisiones para TimeSeriesSplit en OOF y CV del meta-modelo si fuera necesario
# Reemplaza con los mejores parámetros encontrados por Hyperopt/Optuna
BEST_LGBM_PARAMS = {
    # Ejemplo - ¡USA TUS MEJORES PARÁMETROS AQUÍ!
    'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1,
    'boosting_type': 'gbdt', 'seed': 42, 'n_jobs': -1,
    'learning_rate': 0.05, 'num_leaves': 100, 'max_depth': 8,
    'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 3, 'reg_alpha': 0.1, 'reg_lambda': 0.1
    # Asegúrate de incluir todos los parámetros optimizados
}
META_MODEL_TYPE = 'ridge' # Opciones: 'ridge', 'lightgbm', 'rf'

# Rutas de salida
OUTPUT_DIR = config.STACKING_MODEL_DIR
MODEL_BASE_FULL_PATH = os.path.join(OUTPUT_DIR, 'lgbm_base_full_train.pkl')
MODEL_META_PATH = os.path.join(OUTPUT_DIR, 'meta_model_final.pkl')
OOF_PREDS_PATH = os.path.join(OUTPUT_DIR, 'oof_predictions.csv')
FINAL_PREDS_PATH = os.path.join(OUTPUT_DIR, 'stacking_final_predictions.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Funciones Modulares (Reutilizar/Adaptar) ---

def load_and_prepare_data(parquet_path, target_col, n_test_weeks):
    """
    Carga datos desde Parquet, realiza preprocesamiento básico, maneja NaNs,
    identifica categóricas y divide en train/test.
    Devuelve también las features originales antes de cualquier transformación
    adicional que pudiera hacerse en un pipeline (importante para meta-modelo).
    """
    print(f"--- Iniciando Carga y Preparación desde: {parquet_path} ---")
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Forma inicial del DataFrame: {df.shape}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo Parquet en '{parquet_path}'.")
        return (None,) * 6 # Devuelve 6 Nones
    except Exception as e:
        print(f"Error al cargar el archivo Parquet: {e}")
        return (None,) * 6

    # Preprocesamiento Básico
    if 'week' not in df.columns:
        print("Error: Falta la columna 'week'.")
        return (None,) * 6
    if not pd.api.types.is_datetime64_any_dtype(df['week']):
        try:
            df['week'] = pd.to_datetime(df['week'])
        except Exception as parse_error:
            print(f"Error al convertir 'week' a datetime: {parse_error}.")
            return (None,) * 6

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
        return (None,) * 6

    # Identificar Categóricas (solo para info y posible uso futuro)
    categorical_features_list = [
        'establecimiento', 'material', 'cluster_label', 'year', 'month',
        'week_of_year', 'has_promo', 'is_covid_period',
        'is_holiday_exact_date', 'is_holiday_in_week'
    ]
    categorical_features_list = [col for col in categorical_features_list if col in df.columns]
    print(f"Identificando {len(categorical_features_list)} columnas como categóricas (para info).")
    # Convertir a tipo 'category' para LightGBM base
    for col in categorical_features_list:
        if not pd.api.types.is_categorical_dtype(df[col]):
            if df[col].isnull().any():
                df[col] = df[col].fillna('Missing')
            try:
                df[col] = df[col].astype('category')
            except TypeError as e:
                 print(f"Error convirtiendo '{col}' a category: {e}.")
                 return (None,) * 6

    # Separar Features (X) y Target (y)
    columns_to_exclude = [target_col, 'week', 'last_sale_week']
    features = [col for col in df.columns if col not in columns_to_exclude and col in df.columns]
    X = df[features]
    y = df[target_col]
    print(f"Target: '{target_col}'. Features: {X.shape[1]} columnas.")

    # Dividir en Entrenamiento y Prueba (Cronológico)
    cutoff_date = df['week'].max() - pd.Timedelta(weeks=n_test_weeks)
    train_mask = (df['week'] <= cutoff_date)
    test_mask = (df['week'] > cutoff_date)
    X_train, X_test = X[train_mask].copy(), X[test_mask].copy() # Usar .copy()
    y_train, y_test = y[train_mask].copy(), y[test_mask].copy()

    # Guardar también fechas e IDs para test set
    dates_test = df.loc[test_mask, 'week'].copy()
    ids_test = df.loc[test_mask, ['establecimiento', 'material']].copy()


    print(f"División Entrenamiento/Prueba (Test >= {cutoff_date.date()}):")
    print(f"  Tamaño Entrenamiento: {X_train.shape[0]} filas")
    print(f"  Tamaño Prueba:       {X_test.shape[0]} filas")

    if X_test.empty or X_train.empty:
        print("Error: Conjunto de entrenamiento o prueba vacío después de la división.")
        return (None,) * 6

    print("--- Carga y Preparación Completada ---")
    # Devolver también dates_test e ids_test
    return X_train, y_train, X_test, y_test, categorical_features_list, dates_test, ids_test


def train_base_model_and_get_oof(base_model_params, X_train, y_train, X_test, categorical_features, tscv):
    """
    Entrena el modelo base usando CV temporal para generar predicciones OOF
    y también entrena un modelo final en todo el train set para predecir en test.

    Args:
        base_model_params (dict): Parámetros para el modelo base LGBM.
        X_train, y_train: Datos de entrenamiento.
        X_test: Datos de prueba.
        categorical_features (list): Lista de columnas categóricas.
        tscv: Objeto TimeSeriesSplit configurado.

    Returns:
        tuple: (oof_preds, test_preds, base_model_full)
               Predicciones OOF para X_train, predicciones para X_test,
               y el modelo base entrenado en todo X_train. Retorna Nones si falla.
    """
    print("\n--- Entrenando Modelo Base y Generando Predicciones OOF ---")
    oof_preds = np.zeros(len(X_train)) # Inicializar array para OOF
    oof_indices = [] # Guardar índices para alinear correctamente
    test_preds_list = [] # Guardar predicciones de test de cada fold (opcional, promediar puede ser útil)
    models_fold = [] # Guardar modelos de cada fold (opcional)

    base_model_params['n_estimators'] = 2000 # Usar n_estimators alto, early stopping controlará

    for fold, (train_index, val_index) in enumerate(tscv.split(X_train)):
        print(f"  Fold {fold+1}/{tscv.get_n_splits()}")
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model_fold = lgb.LGBMRegressor(**base_model_params)
        try:
            model_fold.fit(X_train_fold, y_train_fold,
                           eval_set=[(X_val_fold, y_val_fold)],
                           eval_metric='mae',
                           callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                           categorical_feature=categorical_features)

            # Predecir en validación y guardar OOF
            preds_val_fold = model_fold.predict(X_val_fold)
            oof_preds[val_index] = preds_val_fold
            oof_indices.extend(val_index) # Guardar los índices correspondientes

            # Opcional: Predecir en test con el modelo del fold
            # test_preds_fold = model_fold.predict(X_test)
            # test_preds_list.append(test_preds_fold)
            # models_fold.append(model_fold) # Guardar modelo si se quiere promediar

            print(f"    Fold {fold+1} completado.")

        except Exception as e:
            print(f"¡ERROR en Fold {fold+1} del modelo base!: {e}")
            # Considerar si continuar o detenerse
            return None, None, None
        finally:
            del X_train_fold, X_val_fold, y_train_fold, y_val_fold, model_fold
            gc.collect()

    # Verificar si se generaron OOF para todos los índices esperados
    if len(oof_indices) != len(X_train):
         print(f"Advertencia: OOF preds generados para {len(oof_indices)} índices, pero se esperaban {len(X_train)}.")
         # Podría ocurrir si TimeSeriesSplit tiene huecos o el primer split es muy pequeño
         # Rellenar los faltantes (ej. con la media o un valor constante) podría ser necesario
         # Por ahora, continuaremos, pero esto podría afectar al meta-modelo

    # Crear DataFrame OOF alineado con X_train original
    oof_df = pd.DataFrame({'oof_pred': oof_preds}, index=X_train.index)


    print("\n--- Entrenando Modelo Base Final en TODO el Train Set ---")
    base_model_full = lgb.LGBMRegressor(**base_model_params)
    try:
        # Usar X_test para early stopping es común aquí, aunque técnicamente
        # introduce una pequeña fuga de información sobre la distribución de test.
        # Alternativa: no usar early stopping o usar un split interno de X_train.
        base_model_full.fit(X_train, y_train,
                            eval_set=[(X_test, y_test)], # Monitor en test set real
                            eval_metric='mae',
                            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
                            categorical_feature=categorical_features)

        print("--- Generando Predicciones del Modelo Base en Test Set ---")
        test_preds = base_model_full.predict(X_test)

        # Opcional: Promediar predicciones de test de los modelos de fold
        # if test_preds_list:
        #    test_preds_avg = np.mean(test_preds_list, axis=0)
        #    print("Usando promedio de predicciones de test de los folds.")
        #    test_preds = test_preds_avg # Sobrescribir

        # Guardar el modelo base entrenado en full data
        joblib.dump(base_model_full, MODEL_BASE_FULL_PATH)
        print(f"Modelo base (entrenado en full train) guardado en: {MODEL_BASE_FULL_PATH}")

        return oof_df['oof_pred'], test_preds, base_model_full # Devolver Series OOF y array test_preds

    except Exception as e:
        print(f"¡ERROR entrenando/prediciendo con el modelo base final!: {e}")
        return None, None, None


def train_meta_model(X_train_original, oof_predictions, y_train, meta_model_type='ridge'):
    """
    Entrena el meta-modelo usando las predicciones OOF y opcionalmente las features originales.

    Args:
        X_train_original (pd.DataFrame): Features originales del conjunto de entrenamiento.
        oof_predictions (pd.Series): Predicciones OOF alineadas con X_train_original.
        y_train (pd.Series): Target original de entrenamiento.
        meta_model_type (str): Tipo de meta-modelo ('ridge', 'lightgbm', 'rf').

    Returns:
        Entrenado meta-modelo (ej. Ridge, LGBMRegressor) o None si falla.
    """
    print(f"\n--- Entrenando Meta-Modelo ({meta_model_type}) ---")

    # Crear el dataset para el meta-modelo
    # Opción 1: Solo predicciones OOF como feature
    # X_meta_train = pd.DataFrame({'oof_lgbm': oof_predictions.values}, index=X_train_original.index)

    # Opción 2: Predicciones OOF + Features Originales (más común)
    # Asegurarse de que oof_predictions esté alineado correctamente con X_train_original
    X_meta_train = X_train_original.copy()
    X_meta_train['oof_lgbm'] = oof_predictions.values

    # Convertir categóricas a numéricas para modelos lineales como Ridge
    # O mantenerlas como 'category' si el meta-modelo es LGBM/RF
    if meta_model_type == 'ridge':
        print("Codificando categóricas para Ridge (OneHot)...")
        categorical_meta = X_meta_train.select_dtypes(include='category').columns
        if not categorical_meta.empty:
            # Usar get_dummies es más simple aquí que ColumnTransformer
            X_meta_train = pd.get_dummies(X_meta_train, columns=categorical_meta, dummy_na=False)
        else:
             print("No se encontraron columnas categóricas para codificar.")

    print(f"Forma del dataset del meta-modelo (X_meta_train): {X_meta_train.shape}")

    # Definir y entrenar el meta-modelo
    if meta_model_type == 'ridge':
        # Ridge es simple y a menudo funciona bien como meta-modelo
        meta_model = Ridge(alpha=1.0, random_state=42) # Alpha es un hiperparámetro a ajustar
    elif meta_model_type == 'lightgbm':
        # Usar parámetros más simples para el meta-LGBM
        meta_params = BEST_LGBM_PARAMS.copy() # Empezar con los del base
        meta_params['n_estimators'] = 200 # Menos árboles
        meta_params['num_leaves'] = 31    # Menos complejidad
        meta_params['learning_rate'] = 0.1
        # Quitar parámetros de bagging/feature fraction si se simplifica mucho
        meta_model = lgb.LGBMRegressor(**meta_params)
    # elif meta_model_type == 'rf':
    #     meta_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    else:
        print(f"Error: Tipo de meta-modelo '{meta_model_type}' no soportado.")
        return None

    try:
        if meta_model_type == 'lightgbm':
             # Indicar categóricas si existen y el modelo las soporta
             categorical_meta_lgbm = list(X_meta_train.select_dtypes(include='category').columns)
             meta_model.fit(X_meta_train, y_train, categorical_feature=categorical_meta_lgbm)
        else:
             meta_model.fit(X_meta_train, y_train)

        print("Meta-modelo entrenado exitosamente.")
        # Guardar el meta-modelo
        joblib.dump(meta_model, MODEL_META_PATH)
        print(f"Meta-modelo guardado en: {MODEL_META_PATH}")
        return meta_model
    except Exception as e:
        print(f"¡ERROR entrenando el meta-modelo!: {e}")
        return None


def evaluate_model(y_true, y_pred, label="Final"):
    """Calcula y muestra métricas de evaluación."""
    print(f"\n--- Evaluando Modelo {label} ---")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"Volumen promedio real:   {np.mean(y_true):.2f}")
    print(f"Volumen promedio predicho: {np.mean(y_pred):.2f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    # 1. Cargar y Preparar Datos
    X_train, y_train, X_test, y_test, categorical_features, dates_test, ids_test = load_and_prepare_data(
        PARQUET_FILE_PATH, TARGET_COL, N_TEST_WEEKS
    )

    if X_train is not None:
        # 2. Configurar Validación Cruzada para OOF
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

        # 3. Entrenar Modelo Base, Obtener OOF y Predicciones Test
        oof_preds, test_preds_base, base_model_full = train_base_model_and_get_oof(
            BEST_LGBM_PARAMS, X_train, y_train, X_test, categorical_features, tscv
        )

        if oof_preds is not None and test_preds_base is not None:
            # Guardar OOF para análisis
            oof_df_save = pd.DataFrame({'oof_pred': oof_preds, 'actual': y_train}, index=X_train.index)
            oof_df_save.to_csv(OOF_PREDS_PATH)
            print(f"Predicciones OOF guardadas en: {OOF_PREDS_PATH}")

            # Evaluar modelo base en Test (como referencia)
            _ = evaluate_model(y_test, test_preds_base, label="Base (LGBM)")

            # 4. Entrenar Meta-Modelo
            meta_model = train_meta_model(
                X_train, oof_preds, y_train, meta_model_type=META_MODEL_TYPE
            )

            if meta_model is not None:
                # 5. Preparar Datos para Inferencia del Meta-Modelo
                print("\n--- Preparando datos de Test para Meta-Modelo ---")
                # Opción 1: Solo predicciones base
                # X_meta_test = pd.DataFrame({'oof_lgbm': test_preds_base}, index=X_test.index)

                # Opción 2: Predicciones base + Features Originales Test
                X_meta_test = X_test.copy()
                X_meta_test['oof_lgbm'] = test_preds_base

                # Aplicar la misma codificación que en el entrenamiento del meta-modelo
                if META_MODEL_TYPE == 'ridge':
                    print("Codificando categóricas en Test para Ridge (OneHot)...")
                    categorical_meta_test = X_meta_test.select_dtypes(include='category').columns
                    if not categorical_meta_test.empty:
                        # Usar get_dummies y reindexar para asegurar consistencia con train
                        X_meta_train_cols = joblib.load(MODEL_META_PATH).feature_names_in_ # Cargar cols del modelo entrenado
                        X_meta_test = pd.get_dummies(X_meta_test, columns=categorical_meta_test, dummy_na=False)
                        X_meta_test = X_meta_test.reindex(columns=X_meta_train_cols, fill_value=0) # Alinear columnas
                    else:
                        print("No se encontraron columnas categóricas para codificar en Test.")


                print(f"Forma del dataset de Test del meta-modelo (X_meta_test): {X_meta_test.shape}")

                # 6. Generar Predicciones Finales (Stacking)
                print("\n--- Generando Predicciones Finales (Stacking) ---")
                try:
                    stacking_predictions = meta_model.predict(X_meta_test)
                    # Opcional: Poner a 0 si son negativas
                    stacking_predictions = np.maximum(0, stacking_predictions)
                    print("Predicciones finales generadas.")

                    # 7. Evaluar Modelo Stacking Final
                    final_metrics = evaluate_model(y_test, stacking_predictions, label="Stacking")

                    # 8. Guardar Predicciones Finales
                    final_preds_df = pd.DataFrame({
                        'establecimiento': ids_test['establecimiento'],
                        'material': ids_test['material'],
                        'week': dates_test,
                        'actual_volume': y_test.values,
                        'stacking_predicted_volume': stacking_predictions
                    })
                    final_preds_df.to_csv(FINAL_PREDS_PATH, index=False)
                    print(f"Predicciones finales guardadas en: {FINAL_PREDS_PATH}")

                except Exception as e:
                    print(f"¡ERROR durante la predicción/evaluación final del stacking!: {e}")
            else:
                print("\nNo se pudo entrenar el meta-modelo. No se generan predicciones finales.")
        else:
            print("\nFallo en el entrenamiento del modelo base o generación OOF. No se puede continuar con el stacking.")
    else:
        print("\nEl proceso no pudo continuar debido a errores en la carga/preparación de datos.")

    print("\nProceso de Stacking completo.")

