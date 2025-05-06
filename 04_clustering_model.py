import pyarrow.parquet as pq
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
import config as config
import pandas as pd
import pyarrow as pa
import os

import pyarrow as pa
# Ignorar FutureWarnings de sklearn que pueden aparecer con n_init='auto'
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. Cargar Datos desde Parquet ---
parquet_file = config.GOLD_FEATURES_FULL_PATH # Asegúrate que el nombre sea correcto


def calculate_silhouette_score(features_table):

    print("Esquema de la tabla cargada:")
    print(features_table.schema)

# --- 2. Preparar Datos para Scikit-learn ---

    # Nombres de las columnas de features (deben coincidir con las del archivo Parquet)
    # Asume que las primeras columnas son 'establecimiento', 'material' y el resto son features
    all_columns = features_table.schema.names
    identifier_columns = ['establecimiento', 'material'] # Ajusta si tienes otros IDs
    feature_columns = [col for col in all_columns if col not in identifier_columns]

    print(f"\nColumnas identificadoras: {identifier_columns}")
    print(f"Columnas de features para clustering: {feature_columns}")

    # Extraer solo las columnas de features a un array NumPy
    try:
        feature_arrays = [features_table.column(col_name).to_numpy(zero_copy_only=False)
                        for col_name in feature_columns]
        X_features = np.stack(feature_arrays, axis=1)
        print(f"Datos de features extraídos en array NumPy con forma: {X_features.shape}")

        # Verificar si hay NaNs o Infs residuales (importante para K-Means)
        if np.isnan(X_features).any() or np.isinf(X_features).any():
            print("\n¡Advertencia! Se encontraron valores NaN o Inf en las features."
                " K-Means fallará. Revisa el paso de limpieza/relleno de nulos.")
            # Opcional: intentar rellenar aquí, aunque es mejor hacerlo antes de guardar
            # X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0) # Ejemplo simple
            # print("Se intentó rellenar NaNs/Infs con 0.")
        else:
            print("No se encontraron NaNs/Infs en las features.")

    except Exception as e:
        print(f"Error al extraer features a NumPy: {e}")
        exit()


    # --- 3. Determinar el Número Óptimo de Clusters (k) ---

    # Rango de k a probar
    k_range = range(2, 11) # Probar de 2 a 10 clusters (ajusta según necesidad)
    inertia_values = []
    silhouette_scores = []

    print("\nCalculando métricas para determinar k óptimo...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
        kmeans.fit(X_features)
        inertia_values.append(kmeans.inertia_) # Inertia: Suma de distancias al cuadrado al centroide (WCSS)
        # Silhouette score necesita al menos 2 clusters y puede ser lento en datos grandes
        if k >= 2:
            score = silhouette_score(X_features, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"  k={k}, Inertia={kmeans.inertia_:.2f}, Silhouette Score={score:.4f}")
        else:
            silhouette_scores.append(np.nan) # No aplicable para k=1
            print(f"  k={k}, Inertia={kmeans.inertia_:.2f}")


    # --- Visualización para encontrar k ---

    plt.figure(figsize=(12, 5))

    # Gráfico del Codo (Elbow Method)
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia_values, marker='o')
    plt.title('Método del Codo (Elbow Method)')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.xticks(list(k_range))
    plt.grid(True)

    # Gráfico del Coeficiente de Silueta
    plt.subplot(1, 2, 2)
    # Asegurarse de que k_range y silhouette_scores tengan la misma longitud si k_range empieza en 2
    valid_k_range_for_silhouette = [k for k in k_range if k >= 2]
    plt.plot(valid_k_range_for_silhouette, silhouette_scores, marker='o')
    plt.title('Coeficiente de Silueta')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score Promedio')
    plt.xticks(valid_k_range_for_silhouette)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n--- Interpretación para elegir k ---")
    print(" - Método del Codo: Busca el 'codo' en la gráfica de Inertia, el punto donde la disminución se vuelve menos pronunciada.")
    print(" - Coeficiente de Silueta: Busca el valor de 'k' que maximiza el Silhouette Score (más cercano a 1 es mejor).")
    print("   Valores cercanos a 0 indican clusters solapados. Valores negativos indican que las muestras pueden estar en el cluster incorrecto.")
    print("Elige un valor de 'k' basado en estas gráficas y/o conocimiento del negocio.")



def train_clustering_model(features_table, k, transpose_view=False):
    all_columns = features_table.schema.names
    identifier_columns = ['establecimiento', 'material'] # Ajusta si tienes otros IDs
    feature_columns = [col for col in all_columns if col not in identifier_columns]

    feature_arrays = [features_table.column(col_name).to_numpy(zero_copy_only=False)
                      for col_name in feature_columns]
    X_features = np.stack(feature_arrays, axis=1)
    print(f"Datos de features extraídos en array NumPy con forma: {X_features.shape}")

    # Verificar si hay NaNs o Infs residuales (importante para K-Means)
    if np.isnan(X_features).any() or np.isinf(X_features).any():
        print("\n¡Advertencia! Se encontraron valores NaN o Inf en las features."
              " K-Means fallará. Revisa el paso de limpieza/relleno de nulos.")
        # Opcional: intentar rellenar aquí, aunque es mejor hacerlo antes de guardar
        # X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0) # Ejemplo simple
        # print("Se intentó rellenar NaNs/Infs con 0.")
    else:
        print("No se encontraron NaNs/Infs en las features.")



    # !!! CAMBIA ESTE VALOR BASADO EN EL ANÁLISIS ANTERIOR !!!
    print(f"\nEntrenando modelo K-Means final con k={k}...")

    kmeans_final = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans_final.fit(X_features)

    # Obtener las etiquetas de cluster para cada punto de datos (cada serie)
    cluster_labels = kmeans_final.labels_

    print("Entrenamiento completado.")
    print(f"Número de puntos de datos por cluster: {np.bincount(cluster_labels)}")

    # --- 5. Añadir Etiquetas al Dataset Original y Guardar (Opcional) ---

    # Crear una columna Arrow con las etiquetas
    labels_column = pa.array(cluster_labels, type=pa.int32())

    # Añadir la columna de etiquetas a la tabla original (la que tiene los identificadores)
    # Nota: Esto asume que el orden de las filas en X_features coincide con features_table
    try:
        results_table = features_table.add_column(features_table.num_columns,
                                            pa.field('cluster_label', pa.int32()),
                                            labels_column)

        print("\n--- Tabla con Etiquetas de Cluster Añadidas (Primeras filas) ---")
        print(results_table.slice(0, 10))

        # Opcional: Guardar la tabla con etiquetas en un nuevo archivo Parquet
        # results_output_file = 'clustered_features.parquet'
        # pq.write_table(results_table, results_output_file)
        # print(f"\nTabla con etiquetas guardada en: {results_output_file}")

    except Exception as e:
        print(f"\nError al añadir/guardar etiquetas: {e}")
        # Si falla la adición directa, crear un mapeo y unirlo puede ser alternativa
        # Crear DataFrame de IDs y labels
        id_df = features_table.select(identifier_columns).to_pandas()
        id_df['cluster_label'] = cluster_labels
        print("\nMapeo de Identificadores a Cluster Labels (DataFrame Pandas):")
        print(id_df.head())
        # Aquí podrías unir este id_df con la tabla original si es necesario


    # --- 6. Analizar Centroides (Opcional pero útil) ---
    # Los centroides están en el espacio escalado. Para interpretarlos,
    # necesitarías el objeto 'scaler' del paso anterior para aplicar 'inverse_transform'.
    # Si no guardaste el scaler, solo puedes analizar los centroides escalados.
    print("\n--- Centroides de los Clusters (en espacio escalado) ---")
    centroids_scaled = kmeans_final.cluster_centers_
    for i, centroid in enumerate(centroids_scaled):
        print(f"\nCluster {i}:")
        for feature_name, value in zip(feature_columns, centroid):
            print(f"  {feature_name}: {value:.3f}")
    import pandas as pd
    # print as a table with feature_names as row names and values as column values
    print(pd.DataFrame(centroids_scaled))

    # Si tuvieras el scaler:
    try:
        from joblib import load
        import os
        scaler = load(os.path.join(config.DATA_DIR, "scaler", "features_scaler.joblib")) # Asume que guardaste el scaler
        centroids_original_scale = scaler.inverse_transform(centroids_scaled)
        print("\n--- Centroides de los Clusters (en escala original) ---")
        print(pd.DataFrame(centroids_original_scale))

        # Si se solicita la vista transpuesta
        if transpose_view:
            print("\n--- Generando tabla transpuesta de centroides ---")
            # Crear DataFrame transpuesto con nombres descriptivos de las features
            df_centroids = pd.DataFrame(
                centroids_original_scale.T,  # Transponemos la matriz
                index=feature_columns,  # Nombres de las features como índice (filas)
                columns=[f"Cluster {i}" for i in range(k)]  # Nombres de clusters como columnas
            )
            
            # Nombres más descriptivos para las features
            feature_descriptions = {
                'total_liters': 'Volumen Total (litros)',
                'mean_liters': 'Volumen Medio (litros/semana)',
                'median_liters': 'Volumen Mediana (litros/semana)',
                'max_liters': 'Volumen Máximo (litros/semana)',
                'std_liters': 'Desviación Estándar Volumen',
                'nonzero_weeks_count': 'Semanas con Venta',
                'zero_ratio': 'Ratio Semanas sin Venta',
                'mean_nonzero_liters': 'Volumen Medio en Semanas con Venta',
                'median_nonzero_liters': 'Mediana Volumen en Semanas con Venta',
                'std_nonzero_liters': 'Desv. Est. en Semanas con Venta',
                'cv_squared': 'Coeficiente Variación al Cuadrado',
                'adi': 'Intervalo Medio entre Demandas (ADI)',
                'promo_lift': 'Lift por Promoción'
            }
            
            # Reemplazar índices con descripciones más legibles
            df_centroids.index = [feature_descriptions.get(feat, feat) for feat in feature_columns]
            
            # Mostrar tabla transpuesta
            print("\n--- Tabla Transpuesta de Centroides (Escala Original) ---")
            pd.set_option('display.max_rows', None)  # Mostrar todas las filas
            pd.set_option('display.width', 120)  # Ancho suficiente para ver bien
            pd.set_option('display.precision', 2)  # Reducir decimales para legibilidad
            print(df_centroids)
            
            # Guardar como CSV para fácil acceso
            csv_path = os.path.join(config.DATA_DIR, "centroids_transposed.csv")
            df_centroids.to_csv(csv_path)
            print(f"\nTabla guardada en: {csv_path}")
            
            # Crear una visualización más atractiva con matplotlib
            plt.figure(figsize=(14, 8))
            
            # Crear tabla en matplotlib
            the_table = plt.table(
                cellText=df_centroids.round(2).values,
                rowLabels=df_centroids.index,
                colLabels=df_centroids.columns,
                loc='center',
                cellLoc='center'
            )
            
            # Ajustar tamaño y estilo
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(9)
            the_table.scale(1.2, 1.5)
            
            # Eliminar ejes
            plt.axis('off')
            plt.title('Centroides de Clusters (Valores Transpuestos)', fontsize=16)
            
            # Guardar imagen
            plt_path = os.path.join(config.DATA_DIR, "centroids_transposed.png")
            plt.savefig(plt_path, bbox_inches='tight', dpi=150)
            print(f"Visualización guardada en: {plt_path}")
            plt.close()

    except FileNotFoundError:
        print("\nNo se encontró 'scaler.joblib'. No se pueden mostrar centroides en escala original.")


def transpose_cluster_centroids():
    """
    Genera una tabla transpuesta de los centroides de los clusters.
    Las filas serán los atributos/features y las columnas serán los clusters.
    """
    import pandas as pd
    from joblib import load
    import os
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    
    print("Generando tabla transpuesta de centroides...")
    
    # Cargar datos
    features_table = pq.read_table(config.GOLD_FEATURES_FULL_PATH)
    
    # Extraer nombres de features
    all_columns = features_table.schema.names
    identifier_columns = ['establecimiento', 'material']
    feature_columns = [col for col in all_columns if col not in identifier_columns]
    
    # Extraer datos para clustering
    feature_arrays = [features_table.column(col_name).to_numpy(zero_copy_only=False)
                      for col_name in feature_columns]
    X_features = np.stack(feature_arrays, axis=1)
    
    # Entrenar modelo KMeans con k=6 (o el número que se determinó como óptimo)
    k = 6
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(X_features)
    centroids_scaled = kmeans.cluster_centers_
    
    # Cargar el scaler para convertir a escala original
    try:
        scaler = load(os.path.join(config.DATA_DIR, "scaler", "features_scaler.joblib"))
        centroids_original = scaler.inverse_transform(centroids_scaled)
        
        # Crear DataFrame transpuesto con nombres descriptivos de las features
        # Los nombres de las columnas serán "Cluster 0", "Cluster 1", etc.
        df_centroids = pd.DataFrame(
            centroids_original.T,  # Transponemos la matriz
            index=feature_columns,  # Nombres de las features como índice (filas)
            columns=[f"Cluster {i}" for i in range(k)]  # Nombres de clusters como columnas
        )
        
        # Nombres más descriptivos para las features
        feature_descriptions = {
            'total_liters': 'Volumen Total (litros)',
            'mean_liters': 'Volumen Medio (litros/semana)',
            'median_liters': 'Volumen Mediana (litros/semana)',
            'max_liters': 'Volumen Máximo (litros/semana)',
            'std_liters': 'Desviación Estándar Volumen',
            'nonzero_weeks_count': 'Semanas con Venta',
            'zero_ratio': 'Ratio Semanas sin Venta',
            'mean_nonzero_liters': 'Volumen Medio en Semanas con Venta',
            'median_nonzero_liters': 'Mediana Volumen en Semanas con Venta',
            'std_nonzero_liters': 'Desv. Est. en Semanas con Venta',
            'cv_squared': 'Coeficiente Variación al Cuadrado',
            'adi': 'Intervalo Medio entre Demandas (ADI)',
            'promo_lift': 'Lift por Promoción'
        }
        
        # Reemplazar índices con descripciones más legibles
        df_centroids.index = [feature_descriptions.get(feat, feat) for feat in feature_columns]
        
        # Mostrar tabla transpuesta
        print("\n--- Tabla Transpuesta de Centroides (Escala Original) ---")
        pd.set_option('display.max_rows', None)  # Mostrar todas las filas
        pd.set_option('display.width', 120)  # Ancho suficiente para ver bien
        pd.set_option('display.precision', 2)  # Reducir decimales para legibilidad
        print(df_centroids)
        
        # Guardar como CSV para fácil acceso
        csv_path = os.path.join(config.DATA_DIR, "centroids_transposed.csv")
        df_centroids.to_csv(csv_path)
        print(f"\nTabla guardada en: {csv_path}")
        
        # Crear una visualización más atractiva con matplotlib
        plt.figure(figsize=(14, 8))
        
        # Crear tabla en matplotlib
        the_table = plt.table(
            cellText=df_centroids.round(2).values,
            rowLabels=df_centroids.index,
            colLabels=df_centroids.columns,
            loc='center',
            cellLoc='center'
        )
        
        # Ajustar tamaño y estilo
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1.2, 1.5)
        
        # Eliminar ejes
        plt.axis('off')
        plt.title('Centroides de Clusters (Valores Transpuestos)', fontsize=16)
        
        # Guardar imagen
        plt_path = os.path.join(config.DATA_DIR, "centroids_transposed.png")
        plt.savefig(plt_path, bbox_inches='tight', dpi=150)
        print(f"Visualización guardada en: {plt_path}")
        plt.close()
        
    except FileNotFoundError:
        print("\nNo se encontró el archivo scaler. No se puede generar la tabla transpuesta.")
    except Exception as e:
        print(f"\nError al generar tabla transpuesta: {e}")

# Función para asignar clusters a los datos de entrenamiento
def assign_clusters_to_training_data(k: int):
    """
    Función principal que asigna los clusters a los datos de entrenamiento.
    """
    print("Asignando clusters a los datos de entrenamiento...")
    
    # 1. Cargar datos de entrenamiento
    training_data = pq.read_table(config.GOLD_WEEKLY_FULL_PATH)
    print(f"Datos de entrenamiento cargados desde {config.GOLD_WEEKLY_FULL_PATH}")
    print(f"Número de registros: {training_data.num_rows}")
    
    # 2. Cargar datos de features (que contienen los identificadores y las features para clustering)
    features_table = pq.read_table(config.GOLD_FEATURES_FULL_PATH)
    print(f"Datos de features cargados desde {config.GOLD_FEATURES_FULL_PATH}")
    print(f"Número de registros: {features_table.num_rows}")
    
    # 3. Extraer identificadores y features para clustering
    all_columns = features_table.schema.names
    identifier_columns = ['establecimiento', 'material']
    feature_columns = [col for col in all_columns if col not in identifier_columns]
    
    # 4. Extraer datos para clustering
    feature_arrays = [features_table.column(col_name).to_numpy(zero_copy_only=False)
                    for col_name in feature_columns]
    X_features = np.stack(feature_arrays, axis=1)
    
    # 5. Ejecutar clustering (mismo modelo, mismos parámetros que en 04_clustering_model.py)
    
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(X_features)
    
    # 6. Obtener etiquetas
    cluster_labels = kmeans.labels_
    
    # 7. Crear un DataFrame con identificadores y etiquetas de cluster
    id_df = features_table.select(identifier_columns).to_pandas()
    id_df['cluster_label'] = cluster_labels
    
    # 8. Convertir los datos de entrenamiento a DataFrame para facilitar el merge
    training_df = training_data.to_pandas()
    
    # 9. Combinar los datos de entrenamiento con las etiquetas de cluster
    # Usamos merge en lugar de join para mantener todos los registros
    merged_df = pd.merge(
        training_df, 
        id_df, 
        on=['establecimiento', 'material'],
        how='left'
    )
    
    # 10. Verificar que todos los registros han sido asignados a un cluster
    null_clusters = merged_df['cluster_label'].isna().sum()
    if null_clusters > 0:
        print(f"ADVERTENCIA: {null_clusters} registros no tienen asignado un cluster.")
        # Rellenar valores nulos con un valor que indique "no clasificado" (-1)
        merged_df['cluster_label'] = merged_df['cluster_label'].fillna(-1).astype(int)
    
    # 11. Convertir de nuevo a tabla Arrow
    result_table = pa.Table.from_pandas(merged_df)
    
    # 12. Guardar el resultado
    output_path = os.path.join(config.DATA_DIR, "gold_ventas_semanales_training_clustered.parquet")
    pq.write_table(result_table, output_path)
    print(f"Datos con clusters guardados en: {output_path}")
    
    # 13. Mostrar distribución de clusters
    cluster_counts = merged_df['cluster_label'].value_counts().sort_index()
    print("\nDistribución de registros por cluster:")
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} registros ({count/len(merged_df)*100:.2f}%)")
    
    return output_path


if __name__ == "__main__":

    print("--- INICIO DEL SCRIPT DE CLUSTERING ---")

    # --- PASO 1: Determinar el número óptimo de clusters (k) ---
    # Cargar los datos de features SOLO para este paso
    print("\nCargando datos de features para análisis de k...")
    try:
        features_path = config.GOLD_FEATURES_FULL_PATH
        features_table_for_k = pq.read_table(config.GOLD_FEATURES_FULL_PATH)

        print(f"Datos cargados desde {features_path}")
        # imprimir materiales unicos

        # Llamar a la función que calcula y muestra las gráficas
        calculate_silhouette_score(features_table_for_k)

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de features en {features_path}")
        print("No se puede proceder con el análisis de k.")
    except Exception as e:
        print(f"Error durante el análisis de k: {e}")

    # --- FIN DEL SCRIPT (en esta modificación) ---
    # El código se detendrá aquí después de mostrar las gráficas.
    # Para ejecutar los siguientes pasos (entrenamiento final, asignación),
    # necesitarías:
    # 1. Analizar las gráficas mostradas y ELEGIR un valor para 'k'.
    # 2. Modificar este bloque __main__ o crear otro script para llamar a:
    #    - train_clustering_model(features_table, k_elegido, transpose_view=True)
    #    - assign_clusters_to_training_data(k_elegido)
    #    (Asegúrate de cargar 'features_table' de nuevo si es necesario para esas funciones)

    print("\n--- SCRIPT FINALIZADO ---")
    print("Revisa las gráficas mostradas (Codo y Silueta) para elegir el valor óptimo de 'k'.")
    print("Luego, modifica el script o ejecuta las funciones correspondientes")
    print("con el 'k' elegido para entrenar el modelo final y/o asignar clusters.")

'''
if __name__ == "__main__":
    features_table = pq.read_table(config.GOLD_FEATURES_FULL_PATH)
    # Modificar número de clusters si es necesario
    k = 7
    
    # Puedes descomentar una de estas funciones según lo que quieras hacer
    # Opción 1: Ejecutar clustering y generar vista transpuesta
    train_clustering_model(features_table, k, transpose_view=True)
    
    # Opción 2: Asignar clusters a los datos de entrenamiento
    assign_clusters_to_training_data(k)

'''