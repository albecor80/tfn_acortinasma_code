import duckdb
import os
import pyarrow as pa

def load_parquet_file(con: duckdb.DuckDBPyConnection, table_name: str) -> pa.Table:
    """
    Load a parquet file into a pyarrow table.
    """
    return con.execute(f"SELECT * FROM read_parquet('{table_name}')").fetch_arrow_table()



def print_description_parquetfile(con: duckdb.DuckDBPyConnection, table_name: str):
    """
    Prints the description of a parquet file in DuckDB.
    
    Args:
        con: DuckDB connection
        table_name: Name of the parquet file to describe
    """
    # Get the table name for logging
    base_name = os.path.basename(table_name)

    try:
        # Get column list to check what columns are available
        schema_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{table_name}')").fetchdf()
        columns = schema_df['column_name'].tolist()
        
        # Print basic info
        result = con.execute(f"SELECT COUNT(*) as count FROM read_parquet('{table_name}')").fetchdf()['count'].values[0]
        print(f"\n===== DESCRIPTION OF {base_name} =====")
        print(f"Number of records: {result:,}")
        
        # Print schema
        print("\nSchema:")
        print(schema_df)

        # Print unique establishments (if exists)
        if 'establecimiento' in columns:
            try:
                unique_establecimientos = con.execute(f"SELECT COUNT(DISTINCT establecimiento) as count FROM read_parquet('{table_name}')").fetchdf()['count'].values[0]
                print(f"\nUnique establishments: {unique_establecimientos:,}")
            except Exception as e:
                print(f"\nError getting unique establishments: {str(e)}")

        # Print date range - handle both 'calday' and 'week' columns
        if 'calday' in columns:
            try:
                first_date = con.execute(f"SELECT MIN(calday) AS min_date FROM read_parquet('{table_name}')").fetchdf()['min_date'].values[0]
                last_date = con.execute(f"SELECT MAX(calday) AS max_date FROM read_parquet('{table_name}')").fetchdf()['max_date'].values[0]
                print(f"\nDate range: {first_date} to {last_date}")
            except Exception as e:
                print(f"\nError getting date range: {str(e)}")
        elif 'week' in columns:
            try:
                first_week = con.execute(f"SELECT MIN(week) AS min_week FROM read_parquet('{table_name}')").fetchdf()['min_week'].values[0]
                last_week = con.execute(f"SELECT MAX(week) AS max_week FROM read_parquet('{table_name}')").fetchdf()['max_week'].values[0]
                print(f"\nWeek range: {first_week} to {last_week}")
            except Exception as e:
                print(f"\nError getting week range: {str(e)}")

        # Print unique promotions (if exists)
        if 'promo_id' in columns:
            try:
                unique_promo_ids = con.execute(f"SELECT COUNT(DISTINCT promo_id) as count FROM read_parquet('{table_name}')").fetchdf()['count'].values[0]
                print(f"\nUnique promotions: {unique_promo_ids:,}")
            except Exception as e:
                print(f"\nError getting unique promotions: {str(e)}")
        
        # Print promotion statistics for gold table
        if 'has_promo' in columns:
            try:
                weeks_with_promo = con.execute(f"""
                    SELECT COUNT(*) as count 
                    FROM read_parquet('{table_name}')
                    WHERE has_promo = 1
                """).fetchdf()['count'].values[0]
                
                total_rows = con.execute(f"SELECT COUNT(*) as count FROM read_parquet('{table_name}')").fetchdf()['count'].values[0]
                promo_percentage = (weeks_with_promo / total_rows) * 100 if total_rows > 0 else 0
                
                print(f"\nPromotion statistics:")
                print(f"  Records with promotions: {weeks_with_promo:,} ({promo_percentage:.2f}%)")
                print(f"  Records without promotions: {total_rows - weeks_with_promo:,} ({100 - promo_percentage:.2f}%)")
            except Exception as e:
                print(f"\nError getting promotion statistics: {str(e)}")

        # Print unique materials (if exists)
        if 'material' in columns:
            try:
                unique_materials = con.execute(f"SELECT COUNT(DISTINCT material) as count FROM read_parquet('{table_name}')").fetchdf()['count'].values[0]
                print(f"\nUnique materials: {unique_materials:,}")
            except Exception as e:
                print(f"\nError getting unique materials: {str(e)}")
        
        # Print additional stats for gold weekly table
        if 'weekly_volume' in columns:
            try:
                total_volume = con.execute(f"SELECT SUM(weekly_volume) as total FROM read_parquet('{table_name}')").fetchdf()['total'].values[0]
                avg_volume = con.execute(f"SELECT AVG(weekly_volume) as avg FROM read_parquet('{table_name}')").fetchdf()['avg'].values[0]
                print(f"\nWeekly volume stats:")
                print(f"  Total volume: {total_volume:,.2f}")
                print(f"  Average weekly volume: {avg_volume:,.2f}")
            except Exception as e:
                print(f"\nError getting weekly volume stats: {str(e)}")
    except Exception as e:
        print(f"\nError describing parquet file {table_name}: {str(e)}")

    print("\n" + "="*40 + "\n")


def show_first_rows(parquet_file: str, n: int = 50):
    """
    Show the first n rows of a parquet file.
    """
    con = duckdb.connect()
    return con.execute(f"SELECT * FROM read_parquet('{parquet_file}')").fetch_arrow_table()


if __name__ == "__main__":
    con = duckdb.connect()
    import config
    print(show_first_rows(config.GOLD_WEEKLY_FILTERED_PATH))
    con.close()

