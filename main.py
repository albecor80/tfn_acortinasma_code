import config
import duckdb
import importlib
import sys
from pathlib import Path
import gc

# Add the src directory to the path if needed
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# Import the module using importlib
data_cleaning = importlib.import_module("01_data_cleaning_processing")

# Extract the functions we need
filter_sales_by_not_type = data_cleaning.filter_sales_by_not_type
promoid_to_boolean = data_cleaning.promoid_to_boolean
remove_columns = data_cleaning.remove_columns
covid_flag = data_cleaning.covid_flag
filter_by_string_in_column = data_cleaning.filter_by_string_in_column
filter_by_min_weeks = data_cleaning.filter_by_min_weeks
group_by_week = data_cleaning.group_by_week
fill_time_series_gaps = data_cleaning.fill_time_series_gaps
sort_series_by_volume = data_cleaning.sort_series_by_volume
process_data = data_cleaning.process_data
filter_by_materials = data_cleaning.filter_by_materials
filter_sales_date = data_cleaning.filter_sales_date



def create_gold_weekly_table():
    con = duckdb.connect()
    
    # 1. Load the initial data into a PyArrow Table
    print(f"Loading initial data from: {config.SILVER_VENTAS_PATH}")
    initial_table = con.sql(f"SELECT * FROM read_parquet('{config.SILVER_VENTAS_PATH}')").fetch_arrow_table()
    print(f"Initial rows: {len(initial_table):,}")
    

    # Register the table for querying
    con.register('initial_table', initial_table)
    # Show first 5 rows
    print("\nInitial Silver Table (first 5 rows):")
    con.sql("SELECT * FROM initial_table LIMIT 5").show()
    processing_pipeline = [
            (filter_sales_by_not_type, [config.TIPOS_A_EXCLUIR]),
            promoid_to_boolean,
            (remove_columns, [config.COLUMNS_TO_REMOVE]),
            covid_flag,
            (filter_by_string_in_column, ['establecimiento', '81']),
            filter_by_materials,
            group_by_week,
            fill_time_series_gaps,
            (filter_by_min_weeks, [12]),
            sort_series_by_volume
        ]


    
    # Process the data through all steps
    final_table = process_data(
        initial_table, 
        processing_pipeline, 
        show_intermediate=True, 
        save_result=True, 
        output_path=config.GOLD_WEEKLY_PATH)


def create_gold_weekly_training_table():
    con = duckdb.connect()
    
    # 1. Load the initial data into a PyArrow Table
    print(f"Loading initial data from: {config.SILVER_VENTAS_PATH}")
    initial_table = con.sql(f"SELECT * FROM read_parquet('{config.GOLD_WEEKLY_PATH}')").fetch_arrow_table()
    print(f"Initial rows: {len(initial_table):,}")
    
    processing_pipeline = [
        (filter_by_min_weeks, [56])
    ]

    final_table = process_data(
        initial_table, 
        processing_pipeline, 
        show_intermediate=True, 
        save_result=True, 
        output_path=config.GOLD_WEEKLY_TRAINING_PATH)


def create_gold_weekly_full_table():
    con = duckdb.connect()
    
    # 1. Load the initial data into a PyArrow Table
    print(f"Loading initial data from: {config.GOLD_WEEKLY_FULL_PATH}")
    initial_table = con.sql(f"SELECT * FROM read_parquet('{config.GOLD_WEEKLY_FULL_PATH}')").fetch_arrow_table()
    print(f"Initial rows: {len(initial_table):,}")
    




    # Register the table for querying
    con.register('initial_table', initial_table)
    # Show first 5 rows
    print("\nInitial Silver Table (first 5 rows):")
    con.sql("SELECT * FROM initial_table LIMIT 5").show()
   

    processing_pipeline = [
        filter_by_materials
    ]

    # Process the data through all steps
    final_table = process_data(
        initial_table, 
        processing_pipeline, 
        show_intermediate=True, 
        save_result=True, 
        output_path=config.GOLD_WEEKLY_FULL_PATH)




if __name__ == "__main__":


    create_gold_weekly_table()
    