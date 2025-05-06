import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import config
import gc

def filter_sales_by_not_type(table: pa.Table, types: list[str]) -> pa.Table:
    """
    Filter sales by type using DuckDB SQL on a PyArrow Table.
    Returns a new PyArrow Table object.
    """
    con = duckdb.connect()
    try:
        con.register('input_table', table)
        types_sql = ", ".join([f"'{t}'" for t in types])
        query = f"SELECT * FROM input_table WHERE tipo NOT IN ({types_sql})"
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close()
    return result_table


def filter_sales_date(table: pa.Table, date_from: str, date_to: str) -> pa.Table:
    """
    Filter sales by date using DuckDB SQL on a PyArrow Table.
    Returns a new PyArrow Table object.
    """
    con = duckdb.connect()
    try:
        con.register('input_table', table)
        query = f"SELECT * FROM input_table WHERE week BETWEEN '{date_from}' AND '{date_to}'"
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close()
    return result_table


def promoid_to_boolean(table: pa.Table) -> pa.Table:
    """
    Create a binary flag indicating whether a row has a promotion or not.
    Returns a PyArrow table with an additional binary column 'has_promo'.
    """
    con = duckdb.connect()
    try:
        con.register('input_table', table)
        query = """
            SELECT *,
                CASE
                    WHEN promo_id IS NOT NULL THEN 1
                    ELSE 0
                END AS has_promo
            FROM input_table
        """
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close()
    return result_table

def remove_columns(table: pa.Table, columns: list[str]) -> pa.Table:
    """
    Remove specified columns from a PyArrow Table.
    Returns a PyArrow table with the specified columns removed.
    """
    con = duckdb.connect()
    try:
        con.register('input_table', table)
        all_columns = table.column_names
        columns_to_keep = [col for col in all_columns if col not in columns]
        select_clause = ', '.join([f'"{col}"' for col in columns_to_keep])
        query = f"SELECT {select_clause} FROM input_table"
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close()
    return result_table


def covid_flag(table: pa.Table) -> pa.Table:
    """
    Create a binary flag indicating whether a row is in the COVID period or not.
    Returns a PyArrow table with an additional binary column 'is_covid'.
    """
    con = duckdb.connect()
    try:
        con.register('input_table', table)
        query = """
            SELECT *,
                CASE
                    WHEN calday BETWEEN '2020-03-01' AND '2022-04-30' THEN 1
                    ELSE 0
                END AS is_covid_period
            FROM input_table
        """
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close()
    return result_table


def filter_by_string_in_column(table: pa.Table, column: str, string_to_filter: str) -> pa.Table:
    """
    Filter rows based on whether a column contains a specific string.
    Returns a PyArrow table with rows where the specified column contains the string.
    """
    con = duckdb.connect()
    try:
        con.register('input_table', table)
        query = f"SELECT * FROM input_table WHERE {column} LIKE '%{string_to_filter}%'"
        result_table = con.sql(query).fetch_arrow_table()
    finally:
        con.close()
    return result_table


def process_data(initial_table: pa.Table, processing_functions: list, 
               show_intermediate: bool = False,
               save_result: bool = False,
               output_path: str = None,
               output_compression: str = 'snappy',
               memory_limit: str = '4GB') -> pa.Table:
    """
    Apply a list of processing functions to a PyArrow table in sequence.
    Returns the final PyArrow table after all processing steps.
    """
    current_table = initial_table
    con = None
    if show_intermediate:
        con = duckdb.connect()
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    try:
        for i, func_info in enumerate(processing_functions):
            func, args, kwargs = func_info if isinstance(func_info, tuple) else (func_info, [], {})
            current_table = func(current_table, *args, **kwargs)
            if show_intermediate:
                print(f"Step {i+1}: {func.__name__} applied. Rows: {len(current_table)}")
        if save_result:
            pq.write_table(current_table, output_path, compression=output_compression)
        return current_table
    finally:
        if con:
            con.close()
        gc.collect()

if __name__ == '__main__':
    con = duckdb.connect()
    print(f"Loading initial data from: {config.GOLD_WEEKLY_PATH}")
    initial_table = con.sql(f"SELECT * FROM read_parquet('{config.GOLD_WEEKLY_PATH}')").fetch_arrow_table()
    print(f"Initial rows: {len(initial_table):,}")
    filter_pipeline = [
        filter_by_string_in_column,
        (filter_sales_date, ['2022-04-01', '2024-12-31'], {})
    ]
    result = process_data(initial_table, filter_pipeline, show_intermediate=True)
    nested_output_path = str(config.DATA_DIR / config.GOLD_WEEKLY_FILTERED_PATH)
    nested_table = process_data(result, filter_pipeline, save_result=True, output_path=nested_output_path)
    print(f"Created {len(result):,} nested series")