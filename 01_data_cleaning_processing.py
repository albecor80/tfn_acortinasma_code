import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
# import pandas as pd # No longer needed
# from utils import load_parquet_file # No longer needed directly in filter function
import config

def filter_sales_by_not_type(table: pa.Table, types: list[str]) -> pa.Table:
    """
    Filter sales by type using DuckDB SQL on a PyArrow Table.
    Creates a temporary in-memory DuckDB connection.
    Returns a new PyArrow Table object.
    """
    con = duckdb.connect() # Create temporary connection
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        # Create a SQL-safe string representation of the list for the IN clause
        types_sql = ", ".join([f"'{t}'" for t in types])
        
        # Construct the filtering query to run on the registered table
        query = f"""
            SELECT *
            FROM input_table
            WHERE tipo NOT IN ({types_sql})
        """
        # Execute the query and fetch the result as an Arrow table
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close() # Ensure connection is closed
    return result_table


def filter_sales_date(table: pa.Table, date_from: str, date_to: str) -> pa.Table:
    """
    Filter sales by date using DuckDB SQL on a PyArrow Table.
    Creates a temporary in-memory DuckDB connection.
    Returns a new PyArrow Table object.
    """
    con = duckdb.connect() # Create temporary connection
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        query = f"""
            SELECT *
            FROM input_table
            WHERE week BETWEEN '{date_from}' AND '{date_to}'
        """
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close() # Ensure connection is closed
    return result_table


def promoid_to_boolean(table: pa.Table) -> pa.Table:
    """
    Create a binary flag indicating whether a row has a promotion or not.
    
    Args:
        table: Input PyArrow table
        column: Name of the promotion column to check
    
    Returns:
        PyArrow table with an additional binary column 'has_promo'
    """
    con = duckdb.connect() # Create temporary connection
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        query = f"""
            SELECT *,
                CASE
                    WHEN promo_id IS NOT NULL THEN 1
                    ELSE 0
                END AS has_promo
            FROM input_table
        """
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close() # Ensure connection is closed
    return result_table

def remove_columns(table: pa.Table, columns: list[str]) -> pa.Table:
    """
    Remove specified columns from a PyArrow Table.
    
    Args:
        table: Input PyArrow table
        columns: List of column names to remove

    Returns:
        PyArrow table with the specified columns removed
    """
    con = duckdb.connect() # Create temporary connection
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        # Get all column names from the table
        all_columns = table.column_names
        
        # Filter out the columns we want to remove
        columns_to_keep = [col for col in all_columns if col not in columns]
        
        # If there are no columns left, return an empty table
        if not columns_to_keep:
            raise ValueError("Cannot remove all columns from the table")
        
        # Create the SELECT clause with the columns to keep
        select_clause = ', '.join([f'"{col}"' for col in columns_to_keep])
        
        query = f"""
            SELECT {select_clause}
            FROM input_table
        """
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close() # Ensure connection is closed
    return result_table

def covid_flag(table: pa.Table) -> pa.Table:
    """
    Create a binary flag indicating whether a row is in the COVID period or not.
    
    Args:
        table: Input PyArrow table
    
    Returns:    
        PyArrow table with an additional binary column 'is_covid'
    """
    con = duckdb.connect() # Create temporary connection
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        query = f"""
            SELECT *,
                CASE
                    WHEN calday BETWEEN '2020-03-01' AND '2022-04-30' THEN 1
                    ELSE 0
                END AS is_covid_period
            FROM input_table
        """
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close() # Ensure connection is closed
    return result_table


def filter_by_string_in_column(table: pa.Table, column: str, string_to_filter: str) -> pa.Table:
    """
    Filter rows based on whether a column contains a specific string.
    
    Args:
        table: Input PyArrow table
        column: Name of the column to filter    
        string_to_filter: String to filter for
    
    Returns:
        PyArrow table with rows where the specified column contains the string
    """
    con = duckdb.connect() # Create temporary connection
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        query = f"""
            SELECT *
            FROM input_table
            WHERE {column} LIKE '%{string_to_filter}%'
        """
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close() # Ensure connection is closed
    return result_table


def process_data(initial_table: pa.Table, processing_functions: list, 
               show_intermediate: bool = False,
               save_result: bool = False,
               output_path: str = None,
               output_compression: str = 'snappy') -> pa.Table:
    """
    Apply a list of processing functions to a PyArrow table in sequence.
    
    Args:
        initial_table: The starting PyArrow table
        processing_functions: List of functions to apply, where each function:
                             - Takes a PyArrow table as its first argument
                             - May take additional args/kwargs
                             - Returns a PyArrow table
        show_intermediate: Whether to print information about intermediate tables
        save_result: Whether to save the final result as a Parquet file
        output_path: Path where to save the Parquet file (required if save_result=True)
        output_compression: Compression algorithm to use (default: 'snappy')
    
    Returns:
        The final PyArrow table after all processing steps
    """
    current_table = initial_table
    
    # Create a temporary connection for displaying results if needed
    con = None
    if show_intermediate:
        con = duckdb.connect()
    
    try:
        # Apply each function in the list
        for i, func_info in enumerate(processing_functions):
            # Each func_info should be either:
            # 1. A function reference
            # 2. A tuple of (function, args, kwargs)
            
            if callable(func_info):
                # Just a function with no extra args
                function = func_info
                args = []
                kwargs = {}
            elif isinstance(func_info, tuple) and len(func_info) >= 1 and callable(func_info[0]):
                # Tuple of (function, args, kwargs)
                function = func_info[0]
                args = func_info[1] if len(func_info) > 1 else []
                kwargs = func_info[2] if len(func_info) > 2 else {}
            else:
                raise ValueError(f"Invalid function specification at position {i}")
            
            # Apply the function
            function_name = function.__name__
            if show_intermediate:
                print(f"\nStep {i+1}: Applying {function_name}")
                print(f"Rows before: {len(current_table):,}")
            
            # Apply the function with current_table as first arg
            current_table = function(current_table, *args, **kwargs)
            
            if show_intermediate:
                print(f"Rows after: {len(current_table):,}")
                # Show first few rows
                con.register('current_table', current_table)
                print(f"\nSample after {function_name} (first 5 rows):")
                con.sql("SELECT * FROM current_table LIMIT 5").show()
        
        # Save the result if requested
        if save_result:
            if output_path is None:
                raise ValueError("output_path must be specified when save_result=True")
            
            print(f"\nSaving result to {output_path}")
            pq.write_table(current_table, output_path, compression=output_compression)
            print(f"Saved {len(current_table):,} rows to {output_path}")
        
        return current_table
    
    finally:
        # Close the connection if it was created
        if con:
            con.close()

def group_by_week(table: pa.Table) -> pa.Table:
    """
    Group sales by week derived from calday.
    
    Args:
        table: Input PyArrow table with calday column (date format)
    
    Returns:
        PyArrow table with sales grouped by week, with aggregated metrics
    """
    con = duckdb.connect() # Create temporary connection
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        query = """
            SELECT 
                establecimiento,
                material,
                DATE_TRUNC('week', calday) AS week,
                -- Keep has_promo as 1 if ANY row in the group had a promotion
                MAX(has_promo) AS has_promo,
                -- Aggregate metrics
                SUM(volume_ap) AS weekly_volume,
                -- Keep other dimension columns
                MAX(is_covid_period) AS is_covid_period
            FROM input_table
            GROUP BY 
                establecimiento,
                material,
                DATE_TRUNC('week', calday)
            ORDER BY 
                establecimiento,
                material,
                week
        """
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close() # Ensure connection is closed
    return result_table

def filter_by_min_weeks(table: pa.Table, min_weeks: int) -> pa.Table:
    """
    Filter out store-product combinations that have fewer than min_weeks of data.
    
    Args:
        table: Input PyArrow table with week column
        min_weeks: Minimum number of weeks required to keep a store-product combination
    
    Returns:
        PyArrow table with only store-product combinations having sufficient data
    """
    con = duckdb.connect() # Create temporary connection
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        query = f"""
            WITH series_counts AS (
                SELECT 
                    establecimiento, 
                    material,
                    COUNT(DISTINCT week) as week_count
                FROM input_table
                GROUP BY establecimiento, material
            )
            SELECT t.*
            FROM input_table t
            JOIN series_counts s
                ON t.establecimiento = s.establecimiento 
                AND t.material = s.material
            WHERE s.week_count >= {min_weeks}
        """
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close() # Ensure connection is closed
    return result_table


def fill_time_series_gaps(table: pa.Table) -> pa.Table:
    """
    Fill gaps in time series data for each store-product combination.
    For each combination, generates rows for any missing weeks between min and max date.
    
    Args:
        table: Input PyArrow table with 'week' column and store-product identifiers
    
    Returns:
        PyArrow table with continuous weekly data, filling missing weeks with NULL values
    """
    con = duckdb.connect() # Create temporary connection
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        # This query:
        # 1. Finds min and max weeks for each store-product combination
        # 2. Generates a continuous sequence of weeks for each combination
        # 3. Left joins with original data to get metrics where available
        # 4. Fills NULL with 0 for numeric columns and appropriate values for flags
        query = """
            WITH 
            -- Get min and max week for each store-product combination
            date_ranges AS (
                SELECT 
                    establecimiento,
                    material,
                    MIN(week) AS min_week,
                    MAX(week) AS max_week
                FROM input_table
                GROUP BY establecimiento, material
            ),
            
            -- Generate all weeks between min and max for each combination
            all_weeks AS (
                SELECT 
                    d.establecimiento,
                    d.material,
                    -- Cast GENERATE_SERIES result to DATE explicitly
                    calendar_value::DATE AS week
                FROM date_ranges d,
                LATERAL UNNEST(
                    GENERATE_SERIES(
                        d.min_week, 
                        d.max_week, 
                        INTERVAL '1 week'
                    )
                ) AS t(calendar_value)
            )
            
            -- Join with original data to get metrics where available
            SELECT 
                a.establecimiento,
                a.material,
                a.week,
                COALESCE(o.has_promo, 0) AS has_promo,
                COALESCE(o.weekly_volume, 0) AS weekly_volume,
                COALESCE(o.is_covid_period, 
                    CASE 
                        WHEN a.week BETWEEN '2020-03-01' AND '2022-04-30' THEN 1
                        ELSE 0
                    END
                ) AS is_covid_period
            FROM all_weeks a
            LEFT JOIN input_table o
                ON a.establecimiento = o.establecimiento
                AND a.material = o.material
                AND a.week = o.week
            ORDER BY 
                a.establecimiento,
                a.material,
                a.week
        """
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close() # Ensure connection is closed
    return result_table

def sort_series_by_volume(table: pa.Table) -> pa.Table:
    """
    Sort the time series data by total volume for each store-product combination.
    
    Args:
        table: Input PyArrow table with weekly_volume column and store-product identifiers
    
    Returns:
        PyArrow table sorted by total volume of each series (establecimiento-material pair)
    """
    con = duckdb.connect() # Create temporary connection
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        # Calculate total volume for each store-product combination
        # Then join back to the original data and sort
        query = """
            WITH series_totals AS (
                SELECT 
                    establecimiento,
                    material,
                    SUM(weekly_volume) AS total_volume
                FROM input_table
                GROUP BY establecimiento, material
            )
            SELECT t.*
            FROM input_table t
            JOIN series_totals s
                ON t.establecimiento = s.establecimiento 
                AND t.material = s.material
            ORDER BY 
                s.total_volume DESC,  -- Primary sort by total volume
                t.establecimiento,    -- Secondary sort to keep series together
                t.material,
                t.week                -- Maintain time order within each series
        """
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close() # Ensure connection is closed
    return result_table

def create_nested_series_format(table: pa.Table, output_path: str = None) -> pa.Table:
    """
    Create a nested format of the time series data, with one row per series 
    (establecimiento-material combination) and the time series data stored as 
    a list of dicts in a 'series' column.
    
    Each dict in the series contains:
    - ds: week date
    - y: weekly volume
    - has_promo: promotion flag
    - is_covid_period: covid period flag
    
    Args:
        table: Input PyArrow table with weekly_volume data
        output_path: Path to save the Parquet file (optional)
    
    Returns:
        PyArrow table with nested series format
    """
    con = duckdb.connect() # Create temporary connection
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        # Use DuckDB's list_aggr and struct_pack to create nested structure
        query = """
            SELECT 
                establecimiento,
                material,
                -- Create the nested series array with date-value pairs including flags
                LIST(STRUCT_PACK(
                    ds := week::VARCHAR, 
                    y := weekly_volume,
                    has_promo := has_promo,
                    is_covid_period := is_covid_period
                )) AS series,
                -- Add a count of points for reference
                COUNT(*) AS num_points,
                -- Add total and average volume for quick reference
                SUM(weekly_volume) AS total_volume,
                AVG(weekly_volume) AS avg_weekly_volume
            FROM input_table
            GROUP BY establecimiento, material
            ORDER BY 
                SUM(weekly_volume) DESC,  -- Sort by total volume
                establecimiento,
                material
        """
        result_table = con.sql(query).fetch_arrow_table()
        
        # Save to parquet if output_path provided
        if output_path:
            print(f"\nSaving nested series format to {output_path}")
            pq.write_table(result_table, output_path)
            print(f"Saved {len(result_table):,} series to {output_path}")
            
            # Log a sample to show structure
            sample = con.sql("""
                SELECT 
                    establecimiento, 
                    material, 
                    num_points, 
                    series[1:3] AS sample_points
                FROM result_table 
                LIMIT 1
            """).fetchall()
            
            print("\nSample of nested structure:")
            print(f"Series for {sample[0][0]}-{sample[0][1]} has {sample[0][2]} points")
            print(f"First few points: {sample[0][3]}")
    finally:
        con.close() # Ensure connection is closed
    
    return result_table

def list_materials_from_parquet(con: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    """
    List the materials from a parquet file.
    """
    return con.execute(f"SELECT DISTINCT material FROM read_parquet('{table_name}')").fetchdf()['material'].tolist()


def filter_by_materials(table: pa.Table) -> pa.Table:
    """
    Filter the table to only include rows where the material is in the list.
    Only includes materials starting with: 'ED', 'FD', 'DL', 'BD', 'VD', 'VI'
    """
    # Create connection first
    con = duckdb.connect()
    
    try:
        # Get all distinct materials in gold_ventas_semanales
        materials = list_materials_from_parquet(con, config.SILVER_VENTAS_PATH)
        
        # Filter materials by prefix
        materials = [material for material in materials if material.startswith(('ED', 'FD', 'DL', 'BD', 'VD', 'VI'))]
        
        # Register input table and run query
        con.register('input_table', table)
        query = f"""
            SELECT *
            FROM input_table
            WHERE material IN ({', '.join([f"'{m}'" for m in materials])})
        """
        result_table = con.sql(query).fetch_arrow_table()
        
        return result_table
    finally:
        con.close()  # Ensure connection is closed

if __name__ == '__main__':
    con = duckdb.connect()
    
    # 1. Load the initial data into a PyArrow Table
    print(f"Loading initial data from: {config.GOLD_WEEKLY_PATH}")
    initial_table = con.sql(f"SELECT * FROM read_parquet('{config.GOLD_WEEKLY_PATH}')").fetch_arrow_table()
    print(f"Initial rows: {len(initial_table):,}")
    

    filter_pipeline = [
        filter_by_materials,
        # Pass arguments as a list (not a tuple within a list)
        (filter_sales_date, ['2022-04-01', '2024-12-31'], {})
    ]

    # Enable show_intermediate to debug
    result = process_data(initial_table, filter_pipeline, show_intermediate=True)
    # Create nested series format

    nested_output_path = str(config.DATA_DIR / config.GOLD_WEEKLY_FILTERED_PATH)
    nested_table = create_nested_series_format(result, nested_output_path)
    # Show series count
    print(f"Created {len(result):,} nested series")
    # print first 5 rows of nested_table
    
    con.close()