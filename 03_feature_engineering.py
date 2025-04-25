import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import numpy as np
from sklearn.preprocessing import StandardScaler
import config
import os

import pandas as pd # Only used for date range and holiday generation
import holidays # pip install holidays
import datetime

def feature_engineering_clustering(table_path: str):
# --- 1. Preparar Datos de Entrada como PyArrow Table ---

    arrow_table = pq.read_table(table_path)


    print("--- Tabla Arrow de Entrada ---")
    print(arrow_table)
    print("\nEsquema:")
    print(arrow_table.schema)

    # --- 2. Conectar a DuckDB y Registrar la Tabla Arrow ---
    con = duckdb.connect(database=':memory:', read_only=False)

    # Registrar la tabla Arrow para que DuckDB pueda consultarla
    con.register('sales_data', arrow_table)
    print("--- Tabla Arrow Registrada en DuckDB ---")
    print(con.sql("SELECT * FROM sales_data LIMIT 5").fetch_arrow_table())
    # --- 3. Construir y Ejecutar la Consulta SQL para Extraer Features ---

    # Usaremos CTEs (Common Table Expressions) para organizar el cálculo, especialmente para ADI
    sql_query = """
    WITH WeeklyData AS (
        -- Convertir fecha a número de semana o algo secuencial y ordenable
        SELECT
            establecimiento,
            material,
            epoch(week) AS week_epoch, -- Usar epoch para cálculos de diferencia simples
            weekly_volume,
            has_promo,
            is_covid_period
        FROM sales_data
    ),
    SalesWeeks AS (
        -- Identificar las semanas con ventas para calcular ADI
        SELECT
            establecimiento,
            material,
            week_epoch
        FROM WeeklyData
        WHERE weekly_volume > 0
    ),
    LaggedSalesWeeks AS (
        -- Calcular la semana anterior con ventas para cada semana de venta
        SELECT
            establecimiento,
            material,
            week_epoch,
            LAG(week_epoch, 1) OVER (PARTITION BY establecimiento, material ORDER BY week_epoch) AS prev_sale_epoch
        FROM SalesWeeks
    ),
    Intervals AS (
        -- Calcular los intervalos en segundos entre ventas
        SELECT
            establecimiento,
            material,
            (week_epoch - prev_sale_epoch) AS interval_seconds
        FROM LaggedSalesWeeks
        WHERE prev_sale_epoch IS NOT NULL
    ),
    AdiAvg AS (
        -- Calcular el ADI promedio en segundos y convertir a semanas (aprox)
        SELECT
            establecimiento,
            material,
            AVG(interval_seconds) / (60*60*24*7) AS adi_weeks -- Segundos a semanas
        FROM Intervals
        GROUP BY establecimiento, material
    ),
    BaseFeatures AS (
        -- Calcular la mayoría de las features usando agregaciones estándar y condicionales
        SELECT
            establecimiento,
            material,
            -- Volumen / Magnitud
            SUM(weekly_volume) AS total_liters,
            AVG(weekly_volume) AS mean_liters,
            MEDIAN(weekly_volume) AS median_liters,
            MAX(weekly_volume) AS max_liters,
            STDDEV_SAMP(weekly_volume) AS std_liters, -- Desviación estándar muestral
            -- Intermitencia
            COUNT(*) AS num_weeks,
            COUNT(CASE WHEN weekly_volume > 0 THEN 1 ELSE NULL END) AS nonzero_weeks_count,
            AVG(CASE WHEN weekly_volume = 0 THEN 1.0 ELSE 0.0 END) AS zero_ratio,
            AVG(weekly_volume) FILTER (WHERE weekly_volume > 0) AS mean_nonzero_liters,
            MEDIAN(weekly_volume) FILTER (WHERE weekly_volume > 0) AS median_nonzero_liters,
            STDDEV_SAMP(weekly_volume) FILTER (WHERE weekly_volume > 0) AS std_nonzero_liters,
            -- Respuesta a Eventos
            AVG(weekly_volume) FILTER (WHERE has_promo = 1) AS mean_liters_promo,
            AVG(weekly_volume) FILTER (WHERE has_promo = 0) AS mean_liters_no_promo
            -- Agrega aquí cálculos para 'is_covid_period' si es necesario
        FROM WeeklyData
        GROUP BY establecimiento, material
    )
    -- Query Final: Unir features base con ADI y calcular CV2, Promo Lift
    SELECT
        bf.establecimiento,
        bf.material,
        bf.total_liters,
        bf.mean_liters,
        bf.median_liters,
        bf.max_liters,
        COALESCE(bf.std_liters, 0) as std_liters, -- Coalesce para stddev de grupos con 1 elemento
        bf.num_weeks,
        bf.nonzero_weeks_count,
        bf.zero_ratio,
        bf.mean_nonzero_liters,
        bf.median_nonzero_liters,
        COALESCE(bf.std_nonzero_liters, 0) as std_nonzero_liters,
        -- Calcular CV2 (manejar división por cero)
        CASE
            WHEN bf.mean_nonzero_liters IS NOT NULL AND bf.mean_nonzero_liters != 0
            THEN pow(COALESCE(bf.std_nonzero_liters, 0) / bf.mean_nonzero_liters, 2)
            ELSE NULL -- O 0 si prefieres
        END AS cv_squared,
        -- Unir ADI
        COALESCE(adi.adi_weeks, bf.num_weeks) AS adi, -- Si no hay ADI (pocas ventas), usar num_weeks como valor alto? O NULL?
        -- Calcular Promo Lift (manejar división por cero)
        CASE
            WHEN bf.mean_liters_no_promo IS NOT NULL AND bf.mean_liters_no_promo != 0
            THEN bf.mean_liters_promo / bf.mean_liters_no_promo
            ELSE NULL -- O 1 si no hay efecto medible o datos base
        END AS promo_lift
    FROM BaseFeatures bf
    LEFT JOIN AdiAvg adi
        ON bf.establecimiento = adi.establecimiento AND bf.material = adi.material
    ORDER BY bf.establecimiento, bf.material;
    """

    # Ejecutar la consulta y obtener el resultado como una Tabla Arrow
    features_arrow_table = con.execute(sql_query).arrow()

    # Cerrar conexión DuckDB
    con.close()

    print("\n--- Tabla Arrow con Features Agregadas (Antes de Escalar y Limpiar) ---")
    print(features_arrow_table)
    print("\nEsquema:")
    print(features_arrow_table.schema)

    features_arrow_table = fill_null_column(features_arrow_table, 'cv_squared', 0.0)
    features_arrow_table = fill_null_column(features_arrow_table, 'promo_lift', 1.0)
    features_arrow_table = fill_null_column(features_arrow_table, 'mean_nonzero_liters', 0.0)
    features_arrow_table = fill_null_column(features_arrow_table, 'median_nonzero_liters', 0.0)
    import pyarrow.compute as pc
    max_weeks = pc.max(features_arrow_table['num_weeks']).as_py()
    features_arrow_table = fill_null_column(features_arrow_table, 'adi', float(max_weeks))

    print("\n--- Tabla Arrow con Features Limpias ---")
    print(features_arrow_table.slice(0, 5)) # Mostrar primeras filas

    # --- 5. Escalar las Features usando NumPy y Scikit-learn ---

    # Nombres de las columnas de features a escalar
    feature_columns_for_clustering = [
        'total_liters', 'mean_liters', 'median_liters', 'max_liters', 'std_liters',
        'nonzero_weeks_count', 'zero_ratio', 'mean_nonzero_liters', 'median_nonzero_liters',
        'std_nonzero_liters', 'cv_squared', 'adi', 'promo_lift'
        # 'num_weeks' podría no ser necesaria si usas 'zero_ratio' o 'nonzero_weeks_count'
    ]
    # Asegurarse que todas las columnas seleccionadas existen
    feature_columns_for_clustering = [col for col in feature_columns_for_clustering if col in features_arrow_table.schema.names]


    # Extraer solo las columnas de features a un array NumPy
    feature_arrays = [features_arrow_table.column(col_name).to_numpy(zero_copy_only=False)
                    for col_name in feature_columns_for_clustering]
    features_numpy = np.stack(feature_arrays, axis=1)

    # Escalar
    scaler = StandardScaler()
    scaled_features_numpy = scaler.fit_transform(features_numpy)
    # Guardar el scaler
    import joblib
    import os
    # Crear directorio para el scaler si no existe
    os.makedirs(os.path.join(config.DATA_DIR, "scaler"), exist_ok=True)
    scaler_path = os.path.join(config.DATA_DIR, "scaler", "features_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    # --- 6. Crear la Tabla Arrow Final para Clustering ---

    # Crear nuevas columnas Arrow a partir de los arrays NumPy escalados
    scaled_arrow_cols = [pa.array(scaled_features_numpy[:, i]) for i in range(scaled_features_numpy.shape[1])]

    # Seleccionar las columnas identificadoras de la tabla original
    id_cols = [features_arrow_table.column('establecimiento'), features_arrow_table.column('material')]
    id_names = ['establecimiento', 'material']

    # Combinar identificadores y features escaladas en una nueva tabla
    final_arrow_table = pa.Table.from_arrays(
        id_cols + scaled_arrow_cols,
        names=id_names + feature_columns_for_clustering
    )

    print("\n--- Tabla Arrow Final Lista para Clustering (Features Escaladas) ---")
    print(final_arrow_table)
    print("\nEsquema Final:")
    print(final_arrow_table.schema)


    # Guardar como parquet file
    pq.write_table(final_arrow_table, config.GOLD_FEATURES_PATH)
    print(f"Saved features table to {config.GOLD_FEATURES_PATH}")

    # Para asegurarnos que se guarda correctamente, comprobamos si el archivo existe
    if os.path.exists(config.GOLD_FEATURES_PATH):
        print(f"Confirmed: Features file exists at {config.GOLD_FEATURES_PATH}")
    else:
        print(f"Warning: Features file was not created at {config.GOLD_FEATURES_PATH}")


    return features_arrow_table

    # --- 4. Limpieza de NULLs/NaNs en PyArrow (si es necesario) ---
    # DuckDB con COALESCE ya manejó algunos NULLs, pero podríamos tener otros
    # Por ejemplo, promo_lift puede ser NULL. Rellenamos con 1 (sin efecto)
    # cv_squared puede ser NULL. Rellenamos con 0 (sin varianza relativa)
    # mean/median_nonzero_liters pueden ser NULL si nunca hubo ventas. Rellenamos con 0.

def fill_null_column(table, col_name, fill_value):
    """Helper para rellenar NULLs en una columna de una tabla Arrow."""
    col_index = table.schema.get_field_index(col_name)
    if col_index == -1:
        return table # Columna no encontrada

    col = table.column(col_name)
    mask = pc.is_null(col)
    filled_col = pc.if_else(mask, pa.scalar(fill_value, type=col.type), col)
    return table.set_column(col_index, pa.field(col_name, filled_col.type), filled_col)





def feature_engineering_lightgbm(parquet_path: str, holiday_country: str = 'ES') -> pa.Table:
    """
    Reads a Parquet file containing time series data and generates features
    using PyArrow and DuckDB.

    Args:
        parquet_path: Path to the input Parquet file.
                      Expected columns: establecimiento, material, week, weekly_volume,
                                        has_promo, is_covid_period, cluster_label (optional).
        holiday_country: Country code for the holidays library (e.g., 'ES').

    Returns:
        A PyArrow Table containing the original data plus the generated features.
        Returns None if an error occurs during loading or processing.
    """
    try:
        # --- 1. Load Data with PyArrow ---
        print(f"Loading data from: {parquet_path}")
        arrow_table = pq.read_table(parquet_path)
        print(f"Initial table loaded with shape: ({arrow_table.num_rows}, {arrow_table.num_columns})")
        print("Initial schema:")
        print(arrow_table.schema)

        # Validate essential columns
        required_cols = {'establecimiento', 'material', 'week', 'weekly_volume'}
        if not required_cols.issubset(arrow_table.schema.names):
            missing = required_cols - set(arrow_table.schema.names)
            print(f"Error: Missing required columns: {missing}")
            return None

        # Ensure 'week' is a date type
        week_col_idx = arrow_table.schema.get_field_index('week')
        if week_col_idx != -1 and not pa.types.is_temporal(arrow_table.schema.field('week').type):
            print("Attempting to cast 'week' column to date32...")
            try:
                # Try casting, assuming it's string or similar. Adjust if needed.
                date_col = pc.cast(arrow_table.column('week'), pa.date32())
                arrow_table = arrow_table.set_column(week_col_idx, pa.field('week', pa.date32()), date_col)
                print("'week' column successfully cast to date32.")
            except Exception as e:
                print(f"Error casting 'week' column to date: {e}. Please ensure it's in a parseable format.")
                return None
        elif week_col_idx == -1:
             print("Error: 'week' column not found.")
             return None


    except Exception as e:
        print(f"Error loading Parquet file '{parquet_path}': {e}")
        return None

    try:
        # --- 2. Prepare Holiday Data ---
        print("Generating holiday features...")
        # Determine year range from data
        min_date_pa = pc.min(arrow_table.column('week')).as_py()
        max_date_pa = pc.max(arrow_table.column('week')).as_py()

        # Handle potential NaT dates if column is empty or has issues
        if min_date_pa is None or max_date_pa is None:
             print("Error: Could not determine date range from 'week' column.")
             return None

        # Ensure they are date objects if they are datetime
        if isinstance(min_date_pa, datetime.datetime):
            min_date_pa = min_date_pa.date()
        if isinstance(max_date_pa, datetime.datetime):
            max_date_pa = max_date_pa.date()

        years = list(range(min_date_pa.year, max_date_pa.year + 1))

        # Get holidays using the library
        country_holidays = holidays.country_holidays(holiday_country, years=years)
        all_holiday_dates = {**country_holidays} # Combine dicts

        # Create a PyArrow table with holiday dates for joining in DuckDB
        holiday_dates_list = sorted(list(all_holiday_dates.keys()))
        holidays_table = pa.Table.from_pydict({
            'holiday_date': pa.array(holiday_dates_list, type=pa.date32()),
            'is_holiday_flag': pa.array([1] * len(holiday_dates_list), type=pa.int8())
        })
        print(f"Generated {holidays_table.num_rows} holiday entries for years {min(years)}-{max(years)}.")

        # --- 3. Connect to DuckDB & Register Tables ---
        print("Connecting to DuckDB and registering tables...")
        con = duckdb.connect(database=':memory:', read_only=False)
        con.register('sales_data', arrow_table)
        con.register('holidays_data', holidays_table)

        # --- 4. Build and Execute Feature Engineering SQL Query ---
        print("Building and executing feature engineering SQL query...")
        # Define rolling window sizes and lag sizes
        rolling_windows = [4, 8, 12, 52]
        lags = [1, 2, 4, 8, 12, 26, 52]

        # Start building the SQL query using CTEs
        sql_parts = []
        sql_parts.append("""
WITH InputData AS (
    -- Select and potentially cast types if needed
    SELECT
        establecimiento,
        material,
        week,
        CAST(weekly_volume AS DOUBLE) AS weekly_volume, -- Use double for calculations
        try_cast(has_promo as TINYINT) as has_promo, -- Ensure correct types
        try_cast(is_covid_period as TINYINT) as is_covid_period,
        cluster_label -- Assumes this column exists from clustering step
    FROM sales_data
),
DateFeatures AS (
    -- Calculate basic and cyclical date features
    SELECT
        *,
        CAST(strftime(week, '%Y') AS INTEGER) AS year,
        CAST(strftime(week, '%m') AS INTEGER) AS month,
        CAST(strftime(week, '%W') AS INTEGER) AS week_of_year, -- ISO week number
        sin(2 * pi() * CAST(strftime(week, '%m') AS INTEGER) / 12.0) AS month_sin,
        cos(2 * pi() * CAST(strftime(week, '%m') AS INTEGER) / 12.0) AS month_cos,
        sin(2 * pi() * CAST(strftime(week, '%W') AS INTEGER) / 52.0) AS week_of_year_sin,
        cos(2 * pi() * CAST(strftime(week, '%W') AS INTEGER) / 52.0) AS week_of_year_cos,
        -- Add day_of_year sin/cos if needed
    FROM InputData
),
HolidayFeatures AS (
    -- Join with holiday data to get flags
    SELECT
        d.*,
        -- Flag if the week *starts* on a holiday date (adjust logic if needed)
        COALESCE(h.is_holiday_flag, 0) AS is_holiday_exact_date,
        -- Flag if *any* day within the week ending on 'week' is a holiday
        -- This requires looking back 6 days. Max window function helps.
        MAX(COALESCE(h.is_holiday_flag, 0)) OVER (
            PARTITION BY establecimiento, material
            ORDER BY week
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS is_holiday_in_week
    FROM DateFeatures d
    LEFT JOIN holidays_data h ON d.week = h.holiday_date -- Join on exact date first
),
LagFeatures AS (
    -- Calculate lag features for weekly_volume
    SELECT
        *,
""")
        # Add lag features dynamically
        lag_cols = []
        for lag in lags:
            lag_cols.append(f"        LAG(weekly_volume, {lag}) OVER (PARTITION BY establecimiento, material ORDER BY week) AS volume_lag_{lag}")
        sql_parts.append(",\n".join(lag_cols))
        sql_parts.append("""
    FROM HolidayFeatures
),
RollingWindowFeatures AS (
    -- Calculate rolling window features
    SELECT
        *,
""")
        # Add rolling window features dynamically
        rolling_cols = []
        base_col = 'weekly_volume'
        # Use shift(1) equivalent by adjusting the window frame
        # ROWS BETWEEN N PRECEDING AND 1 PRECEDING excludes current row
        for w in rolling_windows:
            rolling_cols.extend([
                f"        AVG({base_col}) OVER (PARTITION BY establecimiento, material ORDER BY week ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING) AS volume_roll_mean_{w}w",
                f"        MEDIAN({base_col}) OVER (PARTITION BY establecimiento, material ORDER BY week ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING) AS volume_roll_median_{w}w",
                f"        STDDEV_SAMP({base_col}) OVER (PARTITION BY establecimiento, material ORDER BY week ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING) AS volume_roll_std_{w}w",
                f"        MIN({base_col}) OVER (PARTITION BY establecimiento, material ORDER BY week ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING) AS volume_roll_min_{w}w",
                f"        MAX({base_col}) OVER (PARTITION BY establecimiento, material ORDER BY week ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING) AS volume_roll_max_{w}w",
                # Stats on non-zero values within the window
                f"        COUNT(CASE WHEN {base_col} > 0 THEN 1 ELSE NULL END) OVER (PARTITION BY establecimiento, material ORDER BY week ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING) AS volume_roll_count_nonzero_{w}w",
                f"        AVG(CASE WHEN {base_col} > 0 THEN {base_col} ELSE NULL END) OVER (PARTITION BY establecimiento, material ORDER BY week ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING) AS volume_roll_mean_nonzero_{w}w",
                f"        MEDIAN(CASE WHEN {base_col} > 0 THEN {base_col} ELSE NULL END) OVER (PARTITION BY establecimiento, material ORDER BY week ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING) AS volume_roll_median_nonzero_{w}w",
                f"        STDDEV_SAMP(CASE WHEN {base_col} > 0 THEN {base_col} ELSE NULL END) OVER (PARTITION BY establecimiento, material ORDER BY week ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING) AS volume_roll_std_nonzero_{w}w",
                # Ratio of non-zero weeks in window
                f"        CAST(COUNT(CASE WHEN {base_col} > 0 THEN 1 ELSE NULL END) OVER (PARTITION BY establecimiento, material ORDER BY week ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING) AS DOUBLE) / "
                f"          NULLIF(COUNT(*) OVER (PARTITION BY establecimiento, material ORDER BY week ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING), 0) AS volume_roll_ratio_nonzero_{w}w"
            ])
        sql_parts.append(",\n".join(rolling_cols))
        sql_parts.append("""
    FROM LagFeatures
),
IntermittencyFeatures AS (
    -- Calculate time since last sale
    SELECT
        *,
        -- Find the date of the previous sale
        LAG(CASE WHEN weekly_volume > 0 THEN week ELSE NULL END IGNORE NULLS) OVER (PARTITION BY establecimiento, material ORDER BY week) AS last_sale_week,
        -- Calculate difference in days
        (week - LAG(CASE WHEN weekly_volume > 0 THEN week ELSE NULL END IGNORE NULLS) OVER (PARTITION BY establecimiento, material ORDER BY week)) AS days_since_last_sale
    FROM RollingWindowFeatures
)
-- Final Selection: Select all original columns and newly created features
SELECT * FROM IntermittencyFeatures ORDER BY establecimiento, material, week;
""")

        final_sql = "\n".join(sql_parts)
        # print("\n--- Generated SQL Query ---")
        # print(final_sql)
        # print("--- End SQL Query ---")

        # Execute the query
        result_arrow_table = con.execute(final_sql).arrow()
        print(f"Feature engineering completed. Result table shape: ({result_arrow_table.num_rows}, {result_arrow_table.num_columns})")

        # --- 5. Post-processing and Cleanup ---
        con.close() # Close DuckDB connection

        # Handle potential NaNs/Infs introduced (though COALESCE in SQL is preferred)
        # Example: Fill NaNs in 'days_since_last_sale' for the first sale
        if 'days_since_last_sale' in result_arrow_table.schema.names:
             days_col_idx = result_arrow_table.schema.get_field_index('days_since_last_sale')
             days_col = result_arrow_table.column('days_since_last_sale')
             # Fill with a large number or 0, depending on desired meaning
             # Using fill_null which is simpler than pc.if_else for simple replacement
             filled_days_col = days_col.fill_null(0) # Fill initial NaNs with 0 days
             result_arrow_table = result_arrow_table.set_column(days_col_idx, pa.field('days_since_last_sale', filled_days_col.type), filled_days_col)

        # Fill NaNs in rolling features (often occur at the start)
        # Example: fill rolling std with 0, rolling means/medians with 0 or forward fill
        for col_name in result_arrow_table.schema.names:
             if 'roll_' in col_name:
                 col_idx = result_arrow_table.schema.get_field_index(col_name)
                 col = result_arrow_table.column(col_name)
                 # Simple fill with 0 for demonstration
                 # More sophisticated filling might be needed (e.g., ffill within groups - harder without pandas)
                 if pa.types.is_floating(col.type) or pa.types.is_integer(col.type):
                     filled_col = col.fill_null(0)
                     result_arrow_table = result_arrow_table.set_column(col_idx, pa.field(col_name, filled_col.type), filled_col)


        print("Final Schema after feature engineering:")
        print(result_arrow_table.schema)
        print(f"Final table shape: ({result_arrow_table.num_rows}, {result_arrow_table.num_columns})")
        pq.write_table(result_arrow_table, config.GOLD_FEATURES_LGBM_PATH)
        print(f"Saved features table to {config.GOLD_FEATURES_LGBM_PATH}")

        # Para asegurarnos que se guarda correctamente, comprobamos si el archivo existe
        if os.path.exists(config.GOLD_FEATURES_LGBM_PATH):
            print(f"Confirmed: Features file exists at {config.GOLD_FEATURES_LGBM_PATH}")
        else:
            print(f"Warning: Features file was not created at {config.GOLD_FEATURES_LGBM_PATH}")




        return result_arrow_table

    except Exception as e:
        print(f"An error occurred during feature engineering: {e}")
        if 'con' in locals() and con:
            con.close()
        return None

# --- Example Usage ---
if __name__ == "__main__":



    processed_table = feature_engineering_lightgbm(config.GOLD_WEEKLY_TRAINING_CLUSTERED_PATH)

    if processed_table:
        print("\n--- Processed Table Sample (First 10 rows) ---")
        # Convert to Pandas just for easy printing of the head
        print(processed_table.slice(0, 10).to_pandas())
        print("\n--- Processed Table Schema ---")
        print(processed_table.schema)
    else:
        print("\nFeature engineering failed.")
