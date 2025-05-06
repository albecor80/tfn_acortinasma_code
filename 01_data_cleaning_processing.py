import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
# import pandas as pd # No longer needed
# from utils import load_parquet_file # No longer needed directly in filter function
import config
import gc  # For garbage collection

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
               output_compression: str = 'snappy',
               memory_limit: str = '4GB') -> pa.Table:
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
        memory_limit: Memory limit for DuckDB operations (default: '4GB')
    
    Returns:
        The final PyArrow table after all processing steps
    """
    current_table = initial_table
    
    # Create a temporary connection for displaying results if needed
    con = None
    if show_intermediate:
        con = duckdb.connect()
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    
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
            
            # Force garbage collection after each step to free memory
            gc.collect()
            
            if show_intermediate:
                print(f"Rows after: {len(current_table):,}")
                # Show first few rows
                if con:
                    con.register('current_table', current_table)
                    print(f"\nSample after {function_name} (first 5 rows):")
                    con.sql("SELECT * FROM current_table LIMIT 5").show()
                    
                    # Reset DuckDB connection to free memory
                    con.close()
                    con = duckdb.connect()
                    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        
        # Save the result if requested
        if save_result:
            if output_path is None:
                raise ValueError("output_path must be specified when save_result=True")
            
            print(f"\nSaving result to {output_path}")
            # Use chunked writing for large tables
            if len(current_table) > 1000000:  # If table is large
                print("Using chunked writing for large table...")
                # For large tables, we'll split the table into chunks
                import os
                from pathlib import Path
                
                # Create a temporary directory for chunks
                temp_dir = Path(output_path).parent / f"temp_{Path(output_path).stem}"
                os.makedirs(temp_dir, exist_ok=True)
                
                # Write chunks to individual files
                chunk_size = 250000
                num_chunks = (len(current_table) + chunk_size - 1) // chunk_size  # Ceiling division
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(current_table))
                    
                    # Extract chunk
                    chunk = current_table.slice(start_idx, end_idx - start_idx)
                    
                    # Write to temp file
                    chunk_path = temp_dir / f"chunk_{i}.parquet"
                    pq.write_table(chunk, chunk_path, compression=output_compression)
                    print(f"  - Saved chunk {i+1}/{num_chunks} to {chunk_path}")
                    
                    # Release memory
                    del chunk
                    gc.collect()
                
                # Merge chunks into final file
                print(f"Merging {num_chunks} chunks into final file...")
                
                # Read and concatenate all chunks
                chunk_files = sorted(temp_dir.glob("chunk_*.parquet"))
                tables = []
                
                for chunk_file in chunk_files:
                    tables.append(pq.read_table(chunk_file))
                
                # Write concatenated table to final path
                merged_table = pa.concat_tables(tables)
                pq.write_table(merged_table, output_path, compression=output_compression)
                
                # Clean up temporary files
                for chunk_file in chunk_files:
                    os.remove(chunk_file)
                os.rmdir(temp_dir)
                
                print(f"Successfully merged chunks and cleaned up temporary files")
            else:
                # Use standard PyArrow writing for smaller tables
                pq.write_table(current_table, output_path, compression=output_compression)
            
            print(f"Saved {len(current_table):,} rows to {output_path}")
        
        return current_table
    
    finally:
        # Close the connection if it was created
        if con:
            con.close()
        
        # Final garbage collection
        gc.collect()

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
    
    Processes data in smaller chunks to avoid memory issues.
    
    Args:
        table: Input PyArrow table with 'week' column and store-product identifiers
    
    Returns:
        PyArrow table with continuous weekly data, filling missing weeks with NULL values
    """
    import pyarrow as pa
    import pandas as pd
    from tqdm import tqdm
    
    # Create a DuckDB connection with memory limits
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")  # Limit DuckDB memory usage
    
    try:
        # Register the input table
        con.register('input_table', table)
        
        # Get unique store-product combinations
        combinations = con.execute("""
            SELECT DISTINCT establecimiento, material 
            FROM input_table
            ORDER BY establecimiento, material
        """).fetchall()
        
        print(f"Processing {len(combinations)} unique store-product combinations in batches")
        
        # Process in smaller batches to avoid memory issues
        batch_size = 500  # Adjust based on memory constraints
        all_results = []
        
        # Process in batches
        for i in range(0, len(combinations), batch_size):
            batch = combinations[i:i+batch_size]
            batch_conditions = []
            
            # Build WHERE conditions for the current batch
            for estab, mat in batch:
                batch_conditions.append(f"(establecimiento = '{estab}' AND material = '{mat}')")
            
            # Process this batch
            where_clause = " OR ".join(batch_conditions)
            
            # This query:
            # 1. Finds min and max weeks for each store-product combination in the batch
            # 2. Generates a continuous sequence of weeks for each combination
            # 3. Left joins with original data to get metrics where available
            # 4. Fills NULL with 0 for numeric columns and appropriate values for flags
            batch_query = f"""
                WITH 
                -- Get min and max week for each store-product in this batch
                date_ranges AS (
                    SELECT 
                        establecimiento,
                        material,
                        MIN(week) AS min_week,
                        MAX(week) AS max_week
                    FROM input_table
                    WHERE {where_clause}
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
            
            # Execute query and collect results
            print(f"Processing batch {i//batch_size + 1}/{(len(combinations)-1)//batch_size + 1} " +
                  f"(items {i+1}-{min(i+batch_size, len(combinations))})")
            
            batch_result = con.execute(batch_query).fetch_arrow_table()
            all_results.append(batch_result)
            
            # Force memory cleanup
            # Close and reopen connection to clear memory between batches
            con.close()
            con = duckdb.connect()
            con.execute("PRAGMA memory_limit='4GB'")
            con.register('input_table', table)
            
            # Also force Python garbage collection
            gc.collect()
            
        # Combine all batches into one table
        if len(all_results) == 1:
            result_table = all_results[0]
        else:
            result_table = pa.concat_tables(all_results)
            
        return result_table
    finally:
        con.close()  # Ensure connection is closed

def sort_series_by_volume(table: pa.Table) -> pa.Table:
    """
    Sort the time series data by total volume for each store-product combination.
    
    Args:
        table: Input PyArrow table with weekly_volume column and store-product identifiers
    
    Returns:
        PyArrow table sorted by total volume of each series (establecimiento-material pair)
    """
    con = duckdb.connect() # Create temporary connection
    con.execute("PRAGMA memory_limit='4GB'")  # Limit DuckDB memory usage
    
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
        
        # For very large tables, process in batches
        if table.num_rows > 1000000:
            # Get the unique combinations and their total volumes
            totals_df = con.execute("""
                SELECT 
                    establecimiento,
                    material,
                    SUM(weekly_volume) AS total_volume
                FROM input_table
                GROUP BY establecimiento, material
                ORDER BY SUM(weekly_volume) DESC
            """).fetchdf()
            
            # Process in batches of combinations
            batch_size = 500
            all_results = []
            
            for i in range(0, len(totals_df), batch_size):
                batch_df = totals_df.iloc[i:i+batch_size]
                estabs = [f"'{e}'" for e in batch_df['establecimiento']]
                mats = [f"'{m}'" for m in batch_df['material']]
                
                batch_conditions = []
                for idx in range(len(batch_df)):
                    e = batch_df.iloc[idx]['establecimiento']
                    m = batch_df.iloc[idx]['material']
                    batch_conditions.append(f"(establecimiento = '{e}' AND material = '{m}')")
                
                where_clause = " OR ".join(batch_conditions)
                
                batch_query = f"""
                    WITH series_totals AS (
                        SELECT 
                            establecimiento,
                            material,
                            SUM(weekly_volume) AS total_volume
                        FROM input_table
                        WHERE {where_clause}
                        GROUP BY establecimiento, material
                    )
                    SELECT t.*
                    FROM input_table t
                    JOIN series_totals s
                        ON t.establecimiento = s.establecimiento 
                        AND t.material = s.material
                    ORDER BY 
                        s.total_volume DESC,
                        t.establecimiento,
                        t.material,
                        t.week
                """
                
                print(f"Processing sort batch {i//batch_size + 1}/{(len(totals_df)-1)//batch_size + 1}")
                batch_result = con.execute(batch_query).fetch_arrow_table()
                all_results.append(batch_result)
                
                # Force memory cleanup
                # Close and reopen connection to clear memory between batches
                con.close()
                con = duckdb.connect()
                con.execute("PRAGMA memory_limit='4GB'")
                con.register('input_table', table)
                
                # Also force Python garbage collection
                gc.collect()
            
            # Combine results
            result_table = pa.concat_tables(all_results)
        else:
            # For smaller tables, process all at once
            result_table = con.execute(query).fetch_arrow_table()
    finally:
        con.close() # Ensure connection is closed
    
    # Force garbage collection
    gc.collect()
    
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
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")  # Limit DuckDB memory usage
    
    try:
        # Register the PyArrow table with DuckDB
        con.register('input_table', table)
        
        # For large tables, process in batches by store-product combinations
        if table.num_rows > 1000000:
            print("Large table detected, processing nested series in batches...")
            
            # Get unique store-product combinations with their total volumes
            combinations = con.execute("""
                SELECT 
                    establecimiento, 
                    material,
                    SUM(weekly_volume) AS total_volume
                FROM input_table
                GROUP BY establecimiento, material
                ORDER BY 
                    total_volume DESC,
                    establecimiento, 
                    material
            """).fetchall()
            
            # Process in batches of combinations
            batch_size = 500
            all_results = []
            
            for i in range(0, len(combinations), batch_size):
                batch = combinations[i:i+batch_size]
                batch_conditions = []
                
                # Build WHERE conditions for the current batch
                for estab, mat, _ in batch:
                    batch_conditions.append(f"(establecimiento = '{estab}' AND material = '{mat}')")
                
                # Process this batch
                where_clause = " OR ".join(batch_conditions)
                
                batch_query = f"""
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
                    WHERE {where_clause}
                    GROUP BY establecimiento, material
                    ORDER BY 
                        SUM(weekly_volume) DESC,
                        establecimiento,
                        material
                """
                
                print(f"Processing nested series batch {i//batch_size + 1}/{(len(combinations)-1)//batch_size + 1}")
                batch_result = con.execute(batch_query).fetch_arrow_table()
                all_results.append(batch_result)
                
                # Force memory cleanup
                con.close()
                con = duckdb.connect()
                con.execute("PRAGMA memory_limit='4GB'")
                con.register('input_table', table)
                
                # Force Python garbage collection
                gc.collect()
            
            # Combine all batches
            result_table = pa.concat_tables(all_results)
        else:
            # For smaller tables, process all at once using the original query
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
            result_table = con.execute(query).fetch_arrow_table()
        
        # Save to parquet if output_path provided
        if output_path:
            print(f"\nSaving nested series format to {output_path}")
            # For large tables, write in chunks
            if len(result_table) > 100000:
                print("Large result table, writing in chunks...")
                import os
                from pathlib import Path
                
                # Create a temporary directory for chunks
                temp_dir = Path(output_path).parent / f"temp_{Path(output_path).stem}"
                os.makedirs(temp_dir, exist_ok=True)
                
                # Write chunks to individual files
                chunk_size = 50000
                num_chunks = (len(result_table) + chunk_size - 1) // chunk_size  # Ceiling division
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(result_table))
                    
                    # Extract chunk
                    chunk = result_table.slice(start_idx, end_idx - start_idx)
                    
                    # Write to temp file
                    chunk_path = temp_dir / f"chunk_{i}.parquet"
                    pq.write_table(chunk, chunk_path, compression='snappy')
                    print(f"  - Saved chunk {i+1}/{num_chunks} to {chunk_path}")
                    
                    # Release memory
                    del chunk
                    gc.collect()
                
                # Merge chunks into final file
                print(f"Merging {num_chunks} chunks into final file...")
                
                # Read and concatenate all chunks
                chunk_files = sorted(temp_dir.glob("chunk_*.parquet"))
                tables = []
                
                for chunk_file in chunk_files:
                    tables.append(pq.read_table(chunk_file))
                
                # Write concatenated table to final path
                merged_table = pa.concat_tables(tables)
                pq.write_table(merged_table, output_path, compression='snappy')
                
                # Clean up temporary files
                for chunk_file in chunk_files:
                    os.remove(chunk_file)
                os.rmdir(temp_dir)
                
                print(f"Successfully merged chunks and cleaned up temporary files")
            else:
                # Standard write for smaller tables
                pq.write_table(result_table, output_path)
                
            print(f"Saved {len(result_table):,} series to {output_path}")
            
            # Log a sample to show structure
            sample = con.execute("""
                SELECT 
                    establecimiento, 
                    material, 
                    num_points, 
                    series[1:3] AS sample_points
                FROM result_table 
                LIMIT 1
            """).fetchall()
            
            if sample:
                print("\nSample of nested structure:")
                print(f"Series for {sample[0][0]}-{sample[0][1]} has {sample[0][2]} points")
                print(f"First few points: {sample[0][3]}")
    finally:
        con.close() # Ensure connection is closed
    
    # Final garbage collection
    gc.collect()
    
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
        materials = [material for material in materials if material.startswith(('ED13', 'FD13', 'DL13', 'VI13', 'ED30', 'FD30', 'DL30', 'VI30', 'ED15', 'FD15', 'DL15', 'VI15' ))]

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