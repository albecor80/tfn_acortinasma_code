import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Import config
import config

# Get the directory of the current script
# script_dir = Path(__file__).parent
# Construct paths relative to the script directory
# bronze_detallistas_path = script_dir / "../data/bronze_detallistas.parquet"
# bronze_ventas_path = script_dir / "../data/bronze_ventas.parquet"


def create_sales_silver_parquet(con: duckdb.DuckDBPyConnection, file_name: str):

    # Use paths from config
    con.sql(f"CREATE OR REPLACE TABLE detallistas AS SELECT * FROM read_parquet('{config.BRONZE_DETALLISTAS_PATH}')")
    con.sql(f"CREATE OR REPLACE TABLE ventas AS SELECT * FROM read_parquet('{config.BRONZE_VENTAS_PATH}')")

    copy_sql = f"""
    COPY (
        SELECT 
            d.establecimiento,
            v.material,
            v.calday,
            v.promo_id,
            SUM(v.volume_ap) AS volume_ap,
            SUM(v.cantidad_umb) AS cantidad_umb,
            d.type AS tipo
        FROM ventas v
        JOIN detallistas d 
        ON CAST(v.detallista AS VARCHAR) = d.detallista
        GROUP BY d.establecimiento, v.material, v.calday, v.promo_id, d.type
    ) TO '{file_name}' (FORMAT parquet)
    """
    
    con.execute(copy_sql)

if __name__ == "__main__":
    # Construct output path relative to the script directory
    # output_path = script_dir / "../data/silver_ventas.parquet"
    con = duckdb.connect()
    # Use output path from config
    create_sales_silver_parquet(con, str(config.SILVER_VENTAS_PATH))

    # Show first 5 rows of the created silver table
    print("\nFirst 5 rows of silver_ventas.parquet:")
    con.sql(f"SELECT * FROM read_parquet('{config.SILVER_VENTAS_PATH}') LIMIT 5").show()

    con.close() # Good practice to close the connection
    
