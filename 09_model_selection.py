import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib # Para cargar modelos
import pyarrow.parquet as pq
import duckdb
import holidays
import datetime
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuración ---
# Rutas a los modelos entrenados (asume que tienes un LGBM y un RF guardados)
# Podrían ser modelos globales o específicos de cluster si los entrenaste así.
# Aquí asumimos modelos globales entrenados con 'cluster_label' como feature.
LGBM_MODEL_PATH = 'lightgbm_final_model_hyperopt.pkl' # Modelo LGBM optimizado
RF_MODEL_PATH = 'rf_final_model.pkl' # Necesitarías entrenar y guardar un RF global o específico
# Ruta al archivo con features históricas (necesario para generar features nuevas)
HISTORICAL_FEATURES_PATH = 'final_features_table.parquet'
# Ruta a un mapeo de Serie -> Cluster (necesario para la selección)
# Este archivo lo tendrías que generar a partir de tus resultados de clustering
SERIES_TO_CLUSTER_MAP_PATH = 'series_cluster_mapping.csv' # Ejemplo: CSV con cols 'establecimiento', 'material', 'cluster_label'

HOLIDAY_COUNTRY = 'ES'
HOLIDAY_PROV = 'CT' # Asumiendo Cataluña si aplica

# --- Funciones del Pipeline ---

def load_inference_assets(lgbm_path, rf_path, cluster_map_path):
    """Carga los modelos y el mapeo de clusters."""
    print("--- Cargando Assets de Inferencia ---")
    try:
        model_lgbm = joblib.load(lgbm_path)
        print(f"Modelo LightGBM cargado desde {lgbm_path}")
    except Exception as e:
        print(f"Error cargando modelo LightGBM: {e}")
        model_lgbm = None

    try:
        model_rf = joblib.load(rf_path)
        print(f"Modelo Random Forest cargado desde {rf_path}")
    except Exception as e:
        print(f"Error cargando modelo Random Forest: {e}")
        model_rf = None

    try:
        cluster_map_df = pd.read_csv(cluster_map_path)
        # Crear un índice multi-nivel para búsqueda rápida
        cluster_map_df.set_index(['establecimiento', 'material'], inplace=True)
        print(f"Mapeo Serie -> Cluster cargado desde {cluster_map_path}")
    except Exception as e:
        print(f"Error cargando mapeo de clusters: {e}")
        cluster_map_df = None

    # Obtener features esperadas (idealmente del modelo LGBM si tiene feature_name_)
    model_features = None
    if model_lgbm and hasattr(model_lgbm, 'feature_name_'):
        model_features = model_lgbm.feature_name_
        print(f"Modelo LGBM espera {len(model_features)} features.")
    elif model_rf and hasattr(model_rf, 'feature_names_in_'): # RF usa feature_names_in_
         model_features = model_rf.feature_names_in_
         print(f"Modelo RF espera {len(model_features)} features.")
    else:
         print("Advertencia: No se pudo obtener la lista de features de los modelos.")


    return model_lgbm, model_rf, cluster_map_df, model_features

def get_cluster_for_series(establecimiento, material, cluster_map):
    """Obtiene el cluster_label para una combinación establecimiento/material."""
    if cluster_map is None:
        print("Advertencia: Mapeo de clusters no disponible.")
        return None # O un cluster por defecto
    try:
        # Buscar en el índice multi-nivel
        cluster_label = cluster_map.loc[(establecimiento, material), 'cluster_label']
        return cluster_label
    except KeyError:
        print(f"Advertencia: No se encontró cluster para {establecimiento}/{material}. Asignando cluster por defecto (ej. -1 o el más común).")
        return -1 # O None, o el cluster más común (ej. 2)
    except Exception as e:
        print(f"Error buscando cluster: {e}")
        return None

# --- Reutilizar funciones `get_series_history`, `prepare_holidays_table`, `generate_inference_features` ---
# Asegúrate de tener estas funciones definidas como en el pipeline de inferencia anterior.
# La función `generate_inference_features` debe calcular TODAS las features que
# esperan los modelos LGBM y RF.

# Placeholder para la función de generación de features (adaptar de la versión anterior)
def generate_inference_features(history_df, target_date, cluster_label_for_feature,
                                future_promo, future_covid, holidays_table,
                                model_features_expected):
     print(f"--- (Simulado) Generando features para fecha objetivo: {target_date} ---")
     # ... (Implementación completa de DuckDB/SQL aquí, igual que antes) ...
     # ... Esta función debe devolver un DataFrame de Pandas con 1 fila ...

     # Ejemplo de salida simulada (DEBES REEMPLAZAR CON LA LÓGICA REAL)
     if history_df is None or history_df.empty: return None
     last_row_hist = history_df.iloc[-1:].copy() # Tomar última fila como base
     last_row_hist['week'] = pd.to_datetime(target_date)
     last_row_hist['has_promo'] = np.int8(future_promo)
     last_row_hist['is_covid_period'] = np.int8(future_covid)
     last_row_hist['cluster_label'] = np.int32(cluster_label_for_feature) # Añadir el cluster como feature
     # Rellenar lags/rolling con valores placeholder (la lógica real los calcularía)
     for col in model_features_expected:
         if col not in last_row_hist.columns:
             last_row_hist[col] = 0.0 # O np.nan
     # Asegurar categóricas
     categorical_features_list = [
        'establecimiento', 'material', 'cluster_label', 'year', 'month',
        'week_of_year', 'has_promo', 'is_covid_period',
        'is_holiday_exact_date', 'is_holiday_in_week'
     ]
     categorical_features_list = [col for col in categorical_features_list if col in last_row_hist.columns]
     for col in categorical_features_list:
         if not pd.api.types.is_categorical_dtype(last_row_hist[col]):
             last_row_hist[col] = last_row_hist[col].astype(str).astype('category')

     # Seleccionar y reordenar
     if model_features_expected:
         try:
             # Añadir columnas faltantes con NaN
             for col in model_features_expected:
                 if col not in last_row_hist.columns:
                     last_row_hist[col] = np.nan
             last_row_hist = last_row_hist[model_features_expected]
         except KeyError as e:
             print(f"Error: Falta la columna '{e}' esperada.")
             return None
     print("   (Simulado) Features generadas.")
     return last_row_hist


def run_inference_model_selection(establecimiento, material, history_df,
                                  target_date, future_promo, future_covid,
                                  model_lgbm, model_rf, cluster_map,
                                  model_features_expected, holidays_table):
    """
    Ejecuta el pipeline de inferencia seleccionando el modelo según el cluster.

    Returns:
        tuple: (predicted_volume, selected_model_type, cluster_label) o (np.nan, None, None)
    """
    print(f"\n--- Ejecutando Inferencia (Selección por Cluster) para Est: {establecimiento}, Mat: {material}, Fecha: {target_date} ---")

    # 1. Determinar el Cluster
    cluster_label = get_cluster_for_series(establecimiento, material, cluster_map)
    if cluster_label is None:
        print("No se pudo determinar el cluster. No se puede predecir.")
        return np.nan, None, None

    print(f"Serie pertenece al Cluster: {cluster_label}")

    # 2. Generar Features (pasando el cluster como info)
    inference_features_df = generate_inference_features(
        history_df, target_date, cluster_label, future_promo, future_covid,
        holidays_table, model_features_expected
    )

    if inference_features_df is None:
        print("Fallo en la generación de features.")
        return np.nan, None, cluster_label

    # 3. Seleccionar Modelo basado en el Cluster
    selected_model = None
    selected_model_type = None

    if cluster_label == 2:
        selected_model = model_rf
        selected_model_type = "RandomForest"
        print("Seleccionando modelo: Random Forest (Cluster 2)")
    elif cluster_label in [0, 4, 6]:
        selected_model = model_lgbm
        selected_model_type = "LightGBM"
        print("Seleccionando modelo: LightGBM (Clusters 0, 4, 6)")
    elif cluster_label in [1, 3, 5]:
        # Para clusters con mal rendimiento general, podríamos usar LGBM
        # pero añadir una advertencia o flag. O devolver un valor por defecto.
        selected_model = model_lgbm
        selected_model_type = "LightGBM (Low Confidence)"
        print(f"Seleccionando modelo: LightGBM (Cluster {cluster_label} - Baja Confianza)")
    else: # Cluster desconocido o no mapeado
        # Usar un modelo por defecto (ej. LGBM) o devolver NaN
        selected_model = model_lgbm
        selected_model_type = "LightGBM (Default)"
        print(f"Seleccionando modelo por defecto: LightGBM (Cluster {cluster_label})")


    # 4. Realizar Predicción
    if selected_model is None:
        print(f"Error: No hay un modelo disponible para el tipo seleccionado '{selected_model_type}'.")
        return np.nan, selected_model_type, cluster_label

    try:
        # Manejar NaNs en las features antes de predecir
        inference_features_df.fillna(0, inplace=True)

        # Asegurar que las categorías estén presentes si el modelo las espera
        # (Esto es más complejo, idealmente las categorías se manejan antes)
        if hasattr(selected_model, 'booster_') and hasattr(selected_model.booster_, 'feature_name'): # LGBM
             current_features = inference_features_df.columns
             expected_features = selected_model.booster_.feature_name()
             # Reordenar y añadir faltantes si es necesario (simplificado)
             if set(current_features) != set(expected_features):
                  print("Advertencia: Features no coinciden exactamente con el modelo LGBM.")
                  # Intentar reordenar, puede fallar si faltan cruciales
                  try:
                      inference_features_df = inference_features_df[expected_features]
                  except KeyError:
                       print("Error fatal: Faltan features clave para LGBM.")
                       return np.nan, selected_model_type, cluster_label
        elif hasattr(selected_model, 'feature_names_in_'): # RF
              current_features = inference_features_df.columns
              expected_features = selected_model.feature_names_in_
              if set(current_features) != set(expected_features):
                  print("Advertencia: Features no coinciden exactamente con el modelo RF.")
                  try:
                      inference_features_df = inference_features_df[expected_features]
                  except KeyError:
                       print("Error fatal: Faltan features clave para RF.")
                       return np.nan, selected_model_type, cluster_label


        prediction = selected_model.predict(inference_features_df)
        predicted_volume = prediction[0]
        predicted_volume = max(0, predicted_volume) # Asegurar no negativo
        print(f"Predicción ({selected_model_type}) generada: {predicted_volume:.2f}")
        return predicted_volume, selected_model_type, cluster_label
    except ValueError as ve:
         print(f"Error durante la predicción ({selected_model_type}): {ve}")
         print("Verifica tipos de datos, NaNs, y categorías.")
         print("Features pasadas al modelo:")
         print(inference_features_df.info())
         return np.nan, selected_model_type, cluster_label
    except Exception as e:
        print(f"Error inesperado durante la predicción ({selected_model_type}): {e}")
        return np.nan, selected_model_type, cluster_label


# --- Bloque Principal de Ejecución (Ejemplo) ---
if __name__ == "__main__":
    # 1. Cargar modelos y mapeo
    model_lgbm, model_rf, cluster_map, model_features = load_inference_assets(
        LGBM_MODEL_PATH, RF_MODEL_PATH, SERIES_TO_CLUSTER_MAP_PATH
    )

    # Salir si los assets críticos no se cargaron
    if model_lgbm is None or model_rf is None or cluster_map is None:
        print("\nError crítico: No se pudieron cargar todos los assets necesarios. Saliendo.")
        exit()

    # 2. Preparar tabla de festivos
    current_year = datetime.date.today().year
    holidays_table = prepare_holidays_table(current_year - 3, current_year + 1, HOLIDAY_COUNTRY, HOLIDAY_PROV)
    if holidays_table is None:
        exit()

    # --- Ejemplo de Inferencia ---
    target_establecimiento = "8100009469" # Ejemplo de cluster 2 (según análisis previo)
    target_material = "FD13"       # Ejemplo
    target_date_predict = datetime.date.today() + datetime.timedelta(weeks=1)
    next_week_promo = 0
    next_week_covid = 0

    # Obtener historial
    history = get_series_history(HISTORICAL_FEATURES_PATH, target_establecimiento, target_material)

    if history is not None:
        # Ejecutar inferencia con selección de modelo
        prediction_result, model_used, cluster_found = run_inference_model_selection(
            establecimiento=target_establecimiento,
            material=target_material,
            history_df=history,
            target_date=target_date_predict,
            future_promo=next_week_promo,
            future_covid=next_week_covid,
            model_lgbm=model_lgbm,
            model_rf=model_rf,
            cluster_map=cluster_map,
            model_features_expected=model_features, # Pasar lista de features esperadas
            holidays_table=holidays_table
        )

        if not np.isnan(prediction_result):
            print(f"\n>> Predicción final para {target_establecimiento}/{target_material} en {target_date_predict} (Cluster {cluster_found}, Modelo: {model_used}): {prediction_result:.2f}")
        else:
            print(f"\n>> No se pudo generar la predicción para {target_establecimiento}/{target_material} en {target_date_predict}.")
    else:
        print(f"\nNo se pudo obtener historial para {target_establecimiento}/{target_material}, no se puede predecir.")

    print("\nPipeline de inferencia (selección por cluster) de ejemplo completado.")