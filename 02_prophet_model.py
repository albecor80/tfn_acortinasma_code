import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import config

# Create output directories if they don't exist
MODELS_DIR = Path(config.DATA_DIR) / "models" / "prophet"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FORECASTS_DIR = Path(config.DATA_DIR) / "forecasts" / "prophet"
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = Path(config.DATA_DIR) / "plots" / "prophet"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Create CV test directory
CV_TEST_DIR = Path(config.DATA_DIR) / "cv_test"
CV_TEST_DIR.mkdir(parents=True, exist_ok=True)


def load_time_series_data(file_path: str) -> pa.Table:
    """Load the time series data from a parquet file as a PyArrow table."""
    con = duckdb.connect()
    try:
        result = con.sql(f"SELECT * FROM read_parquet('{file_path}')").fetch_arrow_table()
    finally:
        con.close()
    return result


def get_unique_series(table: pa.Table) -> list:
    """Get list of unique establecimiento-material combinations using DuckDB."""
    con = duckdb.connect()
    try:
        con.register('input_table', table)
        unique_series = con.sql("""
            SELECT DISTINCT establecimiento, material
            FROM input_table
            ORDER BY establecimiento, material
        """).fetch_arrow_table()
        
        # Convert to list of dicts
        series_list = []
        for i in range(len(unique_series)):
            series_list.append({
                'establecimiento': unique_series['establecimiento'][i].as_py(),
                'material': unique_series['material'][i].as_py()
            })
    finally:
        con.close()
    
    return series_list


def prepare_prophet_data(table: pa.Table, establecimiento: str, material: str) -> tuple:
    """
    Prepare data for Prophet model using DuckDB to filter and sort.
    Returns a tuple of PyArrow arrays (dates, values, promo, covid) that can be converted to a Prophet DataFrame.
    """
    con = duckdb.connect()
    try:
        con.register('input_table', table)
        print(table)
        # Filter and sort using DuckDB
        series_data = con.sql(f"""
            SELECT 
                week AS ds,
                weekly_volume AS y,
                has_promo,
                is_covid_period
            FROM input_table
            WHERE establecimiento = '{establecimiento}'
              AND material = '{material}'
            ORDER BY week
        """).fetch_arrow_table()
    finally:
        con.close()
    
    return series_data


def train_prophet_model(table: pa.Table, series_info: dict, use_regressors: bool = True) -> dict:
    """
    Train a Prophet model for a given time series.
    
    Args:
        table: Full dataset with all series as PyArrow table
        series_info: Dict with 'establecimiento' and 'material'
        use_regressors: Whether to use additional regressors (promo, covid)
    
    Returns:
        Dict with trained model and metadata
    """
    establecimiento = series_info['establecimiento']
    material = series_info['material']
    
    # Prepare data for this specific series
    series_data = prepare_prophet_data(table, establecimiento, material)
    
    # If series is too short, skip
    if len(series_data) < 12:  # Require at least 12 data points
        return {
            'success': False,
            'error': 'Insufficient data points',
            'series_info': series_info,
            'data_points': len(series_data)
        }
    
    try:
        # Convert to pandas only at this point - required for Prophet
        prophet_df = series_data.to_pandas()
        
        # Initialize Prophet model with weekly seasonality
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            interval_width=0.95
        )
        
        # Add regressors if specified
        if use_regressors:
            model.add_regressor('has_promo', mode='multiplicative')
            model.add_regressor('is_covid_period', mode='multiplicative')
            
        # Fit the model
        model.fit(prophet_df)
        
        # Get model parameters
        params = model.params
        
        # Convert to simpler dict for JSON serialization
        simplified_params = {
            'k': float(params['k']),
            'm': float(params['m']),
            'sigma_obs': float(params['sigma_obs'])
        }
        
        return {
            'success': True,
            'model': model,
            'series_info': series_info,
            'data_points': len(series_data),
            'params': simplified_params,
            'start_date': prophet_df['ds'].min().strftime('%Y-%m-%d'),
            'end_date': prophet_df['ds'].max().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'series_info': series_info,
            'data_points': len(series_data)
        }


def generate_forecast(model_result: dict, periods: int = 12) -> pa.Table:
    """
    Generate a forecast for the given number of periods.
    
    Args:
        model_result: Dict with trained model and metadata
        periods: Number of future periods to forecast
    
    Returns:
        Forecast as PyArrow table
    """
    if not model_result['success']:
        return None
    
    model = model_result['model']
    
    # Create future dataframe (Prophet requires pandas)
    future = model.make_future_dataframe(periods=periods, freq='W')
    
    # Add regressor values for the future periods
    future['has_promo'] = 0  # Assume no promotions in the future
    future['is_covid_period'] = 0  # Assume no COVID in the future
    
    # Generate forecast
    forecast_df = model.predict(future)
    
    # Convert back to PyArrow table
    forecast_table = pa.Table.from_pandas(forecast_df)
    
    return forecast_table


def calculate_global_metrics(results: list, data_path: str) -> dict:
    """
    Calculate global MAPE and RMSE by aggregating predictions and actuals 
    across all forecasted series. This gives higher weight to higher-volume series.
    
    Args:
        results: List of model results from run_prophet_modeling
        data_path: Path to the original data to identify series
        
    Returns:
        Dictionary with global metrics
    """
    print("\nCalculating global metrics across all series...")
    
    # Load the original data to get all series info
    con = duckdb.connect()
    try:
        table = con.sql(f"SELECT * FROM read_parquet('{data_path}')").fetch_arrow_table()
    finally:
        con.close()
        
    # Keep track of all predictions and actuals by date
    all_predictions = {}  # Dict of date -> total predicted volume
    all_actuals = {}      # Dict of date -> total actual volume
    series_count = 0      # Count of series included in the global metrics
    
    # Get all successful models with cross-validation results
    for result in results:
        if not result.get('success', False):
            continue
            
        series_id = result.get('series_id')
        if not series_id:
            continue
            
        # Parse the series_id to get establecimiento and material
        try:
            establecimiento, material = series_id.split('_', 1)
        except ValueError:
            print(f"Warning: Cannot parse series_id: {series_id}")
            continue
            
        # Look for cross-validation files
        cv_file = Path(config.DATA_DIR) / "models" / "prophet" / f"{series_id}_cv_results.csv"
        
        if not cv_file.exists():
            print(f"Warning: CV results file not found for {series_id}")
            # Try to run cross-validation now
            try:
                # Find the model
                model_metadata_file = Path(config.DATA_DIR) / "models" / "prophet" / f"{series_id}_metadata.json"
                if not model_metadata_file.exists():
                    print(f"Warning: Model metadata not found for {series_id}")
                    continue
                    
                # Load the model
                with open(model_metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                # Get the series data
                con = duckdb.connect()
                try:
                    series_data = con.sql(f"""
                        SELECT 
                            week AS ds,
                            weekly_volume AS y,
                            has_promo,
                            is_covid_period
                        FROM table
                        WHERE establecimiento = '{establecimiento}'
                          AND material = '{material}'
                        ORDER BY week
                    """).fetch_arrow_table().to_pandas()
                finally:
                    con.close()
                    
                # Skip if not enough data
                if len(series_data) < 20:  # Need at least 20 weeks for meaningful CV
                    print(f"Warning: Not enough data for CV for {series_id}")
                    continue
                    
                # Recreate the model (we need to load from file, not re-train)
                # Since we can't directly load Prophet models from file, we'd need to save full state
                # For this example, I'll just note that we would need access to the cv_results
                print(f"Warning: Would need to run CV for {series_id} but skipping for now")
                continue
            except Exception as e:
                print(f"Error trying to run CV for {series_id}: {str(e)}")
                continue
        
        try:
            # Load the cross-validation results CSV
            cv_results = pd.read_csv(cv_file)
            
            # Add this series to our global metrics
            for _, row in cv_results.iterrows():
                ds = row['ds']  # Date
                y = row['y']    # Actual
                yhat = row['yhat']  # Predicted
                
                # Add to our aggregates
                if ds not in all_actuals:
                    all_actuals[ds] = 0
                if ds not in all_predictions:
                    all_predictions[ds] = 0
                    
                all_actuals[ds] += y
                all_predictions[ds] += yhat
            
            series_count += 1
            print(f"Added CV results from {series_id} to global metrics")
        
        except Exception as e:
            print(f"Error processing CV results for {series_id}: {str(e)}")
            continue
    
    # Now calculate the global metrics
    if not all_actuals:
        print("Warning: No CV results found for global metrics calculation")
        return {'error': 'No CV results available'}
        
    # Convert to dataframe for easier calculation
    global_df = pd.DataFrame({
        'ds': list(all_actuals.keys()),
        'y': list(all_actuals.values()),
        'yhat': list(all_predictions.values())
    }).sort_values('ds')
    
    # Calculate metrics
    global_metrics = {}
    
    # Calculate global RMSE
    global_metrics['global_rmse'] = np.sqrt(np.mean((global_df['y'] - global_df['yhat'])**2))
    
    # Calculate global MAE
    global_metrics['global_mae'] = np.mean(np.abs(global_df['y'] - global_df['yhat']))
    
    # Calculate global MAPE (avoiding division by zero)
    non_zero_mask = (global_df['y'] != 0)
    if non_zero_mask.sum() > 0:
        mape_values = np.abs((global_df['y'][non_zero_mask] - global_df['yhat'][non_zero_mask]) / global_df['y'][non_zero_mask]) * 100
        global_metrics['global_mape'] = float(mape_values.mean())
    else:
        global_metrics['global_mape'] = None
        
    # Add some context
    global_metrics['included_series'] = series_count
    global_metrics['total_periods'] = len(global_df)
    global_metrics['total_actual_volume'] = float(global_df['y'].sum())
    global_metrics['total_predicted_volume'] = float(global_df['yhat'].sum())
    global_metrics['volume_accuracy'] = 100 * (1 - abs(global_df['y'].sum() - global_df['yhat'].sum()) / global_df['y'].sum()) if global_df['y'].sum() != 0 else None
    
    print(f"Global metrics calculated over {series_count} series and {len(global_df)} time periods:")
    print(f"Global RMSE: {global_metrics['global_rmse']:.2f}")
    print(f"Global MAE: {global_metrics['global_mae']:.2f}")
    if global_metrics['global_mape'] is not None:
        print(f"Global MAPE: {global_metrics['global_mape']:.2f}%")
    print(f"Volume Accuracy: {global_metrics['volume_accuracy']:.2f}%")
    
    # Save global metrics to file
    global_metrics_file = Path(config.DATA_DIR) / "prophet_global_metrics.json"
    with open(global_metrics_file, 'w') as f:
        json.dump(global_metrics, f, indent=2)
    print(f"Global metrics saved to {global_metrics_file}")
    
    return global_metrics


def evaluate_model(model_result: dict) -> dict:
    """
    Evaluate the model using cross-validation.
    
    Args:
        model_result: Dict with trained model and metadata
    
    Returns:
        Evaluation metrics
    """
    if not model_result['success']:
        print(f"Skipping evaluation for unsuccessful model: {model_result.get('error', 'Unknown error')}")
        return None
    
    model = model_result['model']
    
    try:
        print(f"Starting cross-validation for {model_result['series_info']['establecimiento']}_{model_result['series_info']['material']}...")
        
        # Perform cross-validation (Prophet's CV requires pandas)
        cv_results = cross_validation(
            model, 
            initial='78 w',  # 78 weeks training
            period='12 w',   # 12 weeks period
            horizon='12 w',  # 12 weeks horizon
            parallel='processes'
        )
        
        print(f"Cross-validation completed with {len(cv_results)} cutoffs")
        
        if len(cv_results) == 0:
            print("Warning: Cross-validation returned 0 cutoffs, cannot calculate metrics")
            return {
                'error': 'Cross-validation returned 0 cutoffs',
                'mape': None,
                'rmse': None
            }
        
        # Save CV results for global metrics calculation
        series_id = f"{model_result['series_info']['establecimiento']}_{model_result['series_info']['material']}"
        cv_file = Path(config.DATA_DIR) / "models" / "prophet" / f"{series_id}_cv_results.csv"
        cv_results.to_csv(cv_file, index=False)
        print(f"Saved CV results to {cv_file}")
        
        # Calculate performance metrics
        metrics = performance_metrics(cv_results)
        
        # Log available metrics
        print(f"Available metrics: {list(metrics.columns)}")
        
        # Print available metrics
        metric_msg = "Metrics calculated: "
        if 'mse' in metrics.columns:
            metric_msg += f"MSE={metrics['mse'].mean():.2f}, "
        if 'rmse' in metrics.columns:
            metric_msg += f"RMSE={metrics['rmse'].mean():.2f}, "
        if 'mae' in metrics.columns:
            metric_msg += f"MAE={metrics['mae'].mean():.2f}"
        print(metric_msg)
        
        # Calculate MAPE manually if not provided by Prophet
        mape_value = None
        if 'mape' not in metrics.columns:
            print("MAPE not calculated by Prophet, attempting manual calculation...")
            try:
                # Calculate MAPE manually: mean(abs((y - yhat) / y)) * 100 for y != 0
                y_true = cv_results['y']
                y_pred = cv_results['yhat']
                
                # Filter out zeros in y to avoid division by zero
                non_zero_mask = (y_true != 0)
                
                if non_zero_mask.sum() > 0:
                    # Calculate MAPE only for non-zero y values
                    mape_values = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100
                    mape_value = float(mape_values.mean())
                    print(f"Manually calculated MAPE: {mape_value:.2f}%")
                else:
                    print("Cannot calculate MAPE: all actual values are zero")
            except Exception as e:
                print(f"Failed to calculate MAPE manually: {str(e)}")
        else:
            mape_value = float(metrics['mape'].mean()) if not metrics['mape'].isnull().all() else None
            
        # Simplify metrics for storage
        simplified_metrics = {
            'mse': float(metrics['mse'].mean()) if 'mse' in metrics.columns and not metrics['mse'].isnull().all() else None,
            'rmse': float(metrics['rmse'].mean()) if 'rmse' in metrics.columns and not metrics['rmse'].isnull().all() else None,
            'mae': float(metrics['mae'].mean()) if 'mae' in metrics.columns and not metrics['mae'].isnull().all() else None,
            'mape': mape_value,
            'coverage': float(metrics['coverage'].mean()) if 'coverage' in metrics.columns and not metrics['coverage'].isnull().all() else None
        }
        
        return simplified_metrics
        
    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'mape': None, 
            'rmse': float(metrics['rmse'].mean()) if 'rmse' in locals() and 'rmse' in metrics.columns else None
        }


def save_model_results(model_result: dict, forecast_table: pa.Table, evaluation: dict):
    """
    Save model results, forecasts, and plots using PyArrow.
    
    Args:
        model_result: Dict with trained model and metadata
        forecast_table: Forecast as PyArrow table
        evaluation: Evaluation metrics
    """
    if not model_result['success']:
        return
    
    establecimiento = model_result['series_info']['establecimiento']
    material = model_result['series_info']['material']
    
    # Create a unique identifier for this series
    series_id = f"{establecimiento}_{material}"
    
    # Save model metadata and evaluation
    metadata = {
        'series_id': series_id,
        'establecimiento': establecimiento,
        'material': material,
        'data_points': model_result['data_points'],
        'start_date': model_result['start_date'],
        'end_date': model_result['end_date'],
        'params': model_result['params'],
        'evaluation': evaluation,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(MODELS_DIR / f"{series_id}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save forecast as parquet using PyArrow
    forecast_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'weekly', 'yearly']
    forecast_subset = forecast_table.select([c for c in forecast_cols if c in forecast_table.column_names])
    pq.write_table(forecast_subset, FORECASTS_DIR / f"{series_id}_forecast.parquet")
    
    # Create and save plots - Prophet plotting requires pandas
    model = model_result['model']
    forecast_df = forecast_table.to_pandas()
    
    # Plot forecast
    fig1 = model.plot(forecast_df)
    fig1.savefig(PLOTS_DIR / f"{series_id}_forecast.png")
    
    # Plot components
    fig2 = model.plot_components(forecast_df)
    fig2.savefig(PLOTS_DIR / f"{series_id}_components.png")
    
    # Close figures to avoid memory issues
    plt.close(fig1)
    plt.close(fig2)


def get_top_n_series(table: pa.Table, n: int) -> list:
    """Get top N series by volume using DuckDB."""
    con = duckdb.connect()
    try:
        con.register('input_table', table)
        
        top_series = con.sql(f"""
            WITH series_volumes AS (
                SELECT 
                    establecimiento,
                    material,
                    SUM(weekly_volume) AS total_volume
                FROM input_table
                GROUP BY establecimiento, material
            )
            SELECT 
                establecimiento,
                material,
                total_volume
            FROM series_volumes
            ORDER BY total_volume DESC
            LIMIT {n}
        """).fetch_arrow_table()
        
        # Convert to list of dicts
        series_list = []
        for i in range(len(top_series)):
            series_list.append({
                'establecimiento': top_series['establecimiento'][i].as_py(),
                'material': top_series['material'][i].as_py()
            })
    finally:
        con.close()
    
    return series_list


def save_metrics_summary(results: list, output_path: str = None):
    """
    Extract and save MAPE and RMSE metrics from all successful models to a JSON file.
    
    Args:
        results: List of model results from run_prophet_modeling
        output_path: Path to save the JSON file (default: data_dir/prophet_metrics_summary.json)
    """
    if output_path is None:
        output_path = Path(config.DATA_DIR) / "prophet_metrics_summary.json"
    
    # Extract metrics from successful models only
    metrics_summary = {}
    skipped_count = 0
    
    print(f"\nProcessing {len(results)} model results for metrics summary...")
    
    for result in results:
        if result.get('success', False):
            series_id = result.get('series_id')
            evaluation = result.get('evaluation')
            
            if not series_id:
                print(f"Warning: Missing series_id in result")
                continue
                
            if not evaluation:
                print(f"Warning: Missing evaluation for {series_id}")
                skipped_count += 1
                continue
            
            # Include any model with at least one valid metric
            has_metrics = False
            metrics_dict = {
                'mape': evaluation.get('mape'),
                'rmse': evaluation.get('rmse'),
                'mae': evaluation.get('mae'),
                'data_points': result.get('data_points')
            }
            
            # Check if at least one metric is available
            if any(v is not None for k, v in metrics_dict.items() if k != 'data_points'):
                has_metrics = True
                
            if has_metrics:
                metrics_summary[series_id] = metrics_dict
                print(f"Added metrics for {series_id}: MAPE={metrics_dict['mape']}, RMSE={metrics_dict['rmse']}, MAE={metrics_dict['mae']}")
            else:
                print(f"Warning: No valid metrics for {series_id}, evaluation has error: {evaluation.get('error')}")
                skipped_count += 1
        else:
            print(f"Skipping unsuccessful model: {result.get('error', 'Unknown error')}")
            skipped_count += 1
    
    print(f"Processed {len(metrics_summary)} models with metrics, skipped {skipped_count} models")
    
    if not metrics_summary:
        print("Warning: No metrics found to summarize. Saving empty summary.")
        empty_summary = {
            "_summary": {
                "error": "No valid metrics found",
                "timestamp": datetime.now().isoformat()
            }
        }
        with open(output_path, 'w') as f:
            json.dump(empty_summary, f, indent=2)
        print(f"Empty metrics summary saved to {output_path}")
        return empty_summary
    
    # Sort by a preferred metric (RMSE if MAPE not available)
    def sort_key(item):
        mape = item[1]['mape']
        rmse = item[1]['rmse']
        mae = item[1]['mae']
        
        # First by MAPE if available, then by RMSE, then by MAE
        if mape is not None:
            return (0, mape)
        elif rmse is not None:
            return (1, rmse)
        elif mae is not None:
            return (2, mae)
        else:
            return (3, float('inf'))
    
    sorted_metrics = dict(sorted(metrics_summary.items(), key=sort_key))
    
    # Add summary statistics
    mape_values = [m['mape'] for m in metrics_summary.values() if m['mape'] is not None]
    rmse_values = [m['rmse'] for m in metrics_summary.values() if m['rmse'] is not None]
    mae_values = [m['mae'] for m in metrics_summary.values() if m['mae'] is not None]
    
    overall_summary = {
        'avg_mape': sum(mape_values) / len(mape_values) if mape_values else None,
        'min_mape': min(mape_values) if mape_values else None,
        'max_mape': max(mape_values) if mape_values else None,
        'avg_rmse': sum(rmse_values) / len(rmse_values) if rmse_values else None,
        'min_rmse': min(rmse_values) if rmse_values else None,
        'max_rmse': max(rmse_values) if rmse_values else None,
        'avg_mae': sum(mae_values) / len(mae_values) if mae_values else None,
        'min_mae': min(mae_values) if mae_values else None,
        'max_mae': max(mae_values) if mae_values else None,
        'successful_models': len(metrics_summary),
        'total_models': len(results),
        'timestamp': datetime.now().isoformat()
    }
    
    # Add to the beginning of the dictionary
    sorted_metrics = {
        '_summary': overall_summary,
        **sorted_metrics
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(sorted_metrics, f, indent=2)
    
    print(f"Metrics summary saved to {output_path}")
    
    return sorted_metrics


def find_best_params_with_cv(series_data_df):
    """
    Find the best parameters for Prophet model using cross-validation.
    
    Args:
        series_data_df: Pandas DataFrame with ds, y, and regressor columns
        
    Returns:
        Dict with best parameters
    """
    print("Finding best parameters with cross-validation...")
    
    # Parameter grid to test
    param_grid = {
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }
    
    # All combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in np.array(np.meshgrid(*param_grid.values())).T.reshape(-1, len(param_grid))]
    
    rmses = []  # Store the RMSE for each parameter combination
    
    # Use cross-validation to evaluate all parameter combinations
    for params in all_params:
        print(f"Testing parameters: {params}")
        
        # Create and fit model with these parameters
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
            **params
        )
        
        # Add regressors if they exist in the data
        if 'has_promo' in series_data_df.columns:
            m.add_regressor('has_promo', mode='multiplicative')
        if 'is_covid_period' in series_data_df.columns:
            m.add_regressor('is_covid_period', mode='multiplicative')
            
        try:
            # Fit the model
            m.fit(series_data_df)
            
            # Cross-validation
            df_cv = cross_validation(
                m,
                initial='78 w',  # 78 weeks initial
                period='12 w',   # 12 weeks period
                horizon='12 w',  # 12 weeks horizon
                parallel='processes'
            )
            
            # Calculate metrics
            df_p = performance_metrics(df_cv)
            
            # Store the RMSE
            rmses.append((params, df_p['rmse'].mean()))
            print(f"  RMSE: {df_p['rmse'].mean():.4f}")
        except Exception as e:
            print(f"  Error with parameters {params}: {str(e)}")
            rmses.append((params, float('inf')))
    
    # Find the best parameters
    best_params = min(rmses, key=lambda x: x[1])[0]
    print(f"Best parameters: {best_params}, RMSE: {min(rmses, key=lambda x: x[1])[1]:.4f}")
    
    return best_params


def test_cv_impact(table: pa.Table, series_list: list, num_series: int = 5):
    """
    Test the impact of cross-validation by comparing models trained with and without
    parameter optimization through cross-validation.
    
    Args:
        table: Full dataset with all series as PyArrow table
        series_list: List of series to test
        num_series: Number of series to test
        
    Returns:
        Dict with test results
    """
    print(f"\nTesting CV impact on {min(num_series, len(series_list))} series...")
    
    results = {}
    counter = 0
    
    for series_info in series_list:
        if counter >= num_series:
            break
            
        establecimiento = series_info['establecimiento']
        material = series_info['material']
        series_id = f"{establecimiento}_{material}"
        
        print(f"\n\nTesting CV impact for series: {series_id}")
        
        # Prepare data for this specific series
        series_data = prepare_prophet_data(table, establecimiento, material)
        
        # Skip if not enough data
        if len(series_data) < 80:  # Need at least 80 data points for meaningful test
            print(f"  Skipping {series_id} - insufficient data points ({len(series_data)})")
            continue
        
        # Convert to pandas for prophet
        series_data_df = series_data.to_pandas()
        
        try:
            # Split into train and test (80/20)
            train_size = int(len(series_data_df) * 0.8)
            train_df = series_data_df.iloc[:train_size].copy()
            test_df = series_data_df.iloc[train_size:].copy()
            
            print(f"  Split data: {len(train_df)} train, {len(test_df)} test points")
            
            # 1. Train model with default parameters
            print("  Training model with default parameters...")
            default_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                interval_width=0.95
            )
            
            # Add regressors
            if 'has_promo' in train_df.columns:
                default_model.add_regressor('has_promo', mode='multiplicative')
            if 'is_covid_period' in train_df.columns:
                default_model.add_regressor('is_covid_period', mode='multiplicative')
                
            default_model.fit(train_df)
            
            # 2. Find best parameters with CV on training data
            best_params = find_best_params_with_cv(train_df)
            
            # 3. Train model with CV-optimized parameters
            print("  Training model with CV-optimized parameters...")
            cv_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95,
                **best_params
            )
            
            # Add regressors
            if 'has_promo' in train_df.columns:
                cv_model.add_regressor('has_promo', mode='multiplicative')
            if 'is_covid_period' in train_df.columns:
                cv_model.add_regressor('is_covid_period', mode='multiplicative')
                
            cv_model.fit(train_df)
            
            # 4. Evaluate both models on test data
            # Create a future dataframe for the test period
            future_df = default_model.make_future_dataframe(periods=0, freq='W')
            future_df = future_df.merge(test_df[['ds']], on='ds', how='inner')
            
            # Add regressor values from test data
            if 'has_promo' in test_df.columns:
                future_df = future_df.merge(test_df[['ds', 'has_promo']], on='ds', how='left')
            if 'is_covid_period' in test_df.columns:
                future_df = future_df.merge(test_df[['ds', 'is_covid_period']], on='ds', how='left')
                
            # Predict with both models
            default_forecast = default_model.predict(future_df)
            cv_forecast = cv_model.predict(future_df)
            
            # Merge forecasts with actual values
            eval_df = pd.DataFrame({
                'ds': test_df['ds'],
                'y': test_df['y'],
                'default_yhat': default_forecast['yhat'],
                'cv_yhat': cv_forecast['yhat']
            })
            
            # Calculate metrics
            default_rmse = np.sqrt(np.mean((eval_df['y'] - eval_df['default_yhat'])**2))
            cv_rmse = np.sqrt(np.mean((eval_df['y'] - eval_df['cv_yhat'])**2))
            
            default_mae = np.mean(np.abs(eval_df['y'] - eval_df['default_yhat']))
            cv_mae = np.mean(np.abs(eval_df['y'] - eval_df['cv_yhat']))
            
            # Calculate MAPE for non-zero values
            non_zero_mask = (eval_df['y'] > 0)
            if non_zero_mask.sum() > 0:
                default_mape = 100 * np.mean(np.abs((eval_df['y'][non_zero_mask] - eval_df['default_yhat'][non_zero_mask]) / eval_df['y'][non_zero_mask]))
                cv_mape = 100 * np.mean(np.abs((eval_df['y'][non_zero_mask] - eval_df['cv_yhat'][non_zero_mask]) / eval_df['y'][non_zero_mask]))
            else:
                default_mape = None
                cv_mape = None
            
            # Store results
            series_result = {
                'series_id': series_id,
                'data_points': len(series_data_df),
                'train_points': len(train_df),
                'test_points': len(test_df),
                'best_params': best_params,
                'default_metrics': {
                    'rmse': default_rmse,
                    'mae': default_mae,
                    'mape': default_mape
                },
                'cv_metrics': {
                    'rmse': cv_rmse,
                    'mae': cv_mae,
                    'mape': cv_mape
                },
                'improvement': {
                    'rmse_pct': ((default_rmse - cv_rmse) / default_rmse * 100) if default_rmse > 0 else None,
                    'mae_pct': ((default_mae - cv_mae) / default_mae * 100) if default_mae > 0 else None,
                    'mape_pct': ((default_mape - cv_mape) / default_mape * 100) if default_mape and cv_mape else None
                }
            }
            
            # Print results
            print(f"\n  Results for {series_id}:")
            print(f"  Default model - RMSE: {default_rmse:.2f}, MAE: {default_mae:.2f}, MAPE: {default_mape:.2f}%" if default_mape else f"  Default model - RMSE: {default_rmse:.2f}, MAE: {default_mae:.2f}")
            print(f"  CV model - RMSE: {cv_rmse:.2f}, MAE: {cv_mae:.2f}, MAPE: {cv_mape:.2f}%" if cv_mape else f"  CV model - RMSE: {cv_rmse:.2f}, MAE: {cv_mae:.2f}")
            
            if series_result['improvement']['rmse_pct']:
                print(f"  Improvement - RMSE: {series_result['improvement']['rmse_pct']:.2f}%, MAE: {series_result['improvement']['mae_pct']:.2f}%")
                
            # Create comparison plot
            plt.figure(figsize=(12, 6))
            plt.plot(eval_df['ds'], eval_df['y'], 'o-', label='Actual')
            plt.plot(eval_df['ds'], eval_df['default_yhat'], 'o-', label='Default Model')
            plt.plot(eval_df['ds'], eval_df['cv_yhat'], 'o-', label='CV-Optimized Model')
            plt.title(f"CV Test Results for {series_id}")
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.tight_layout()
            plt.savefig(CV_TEST_DIR / f"{series_id}_cv_test.png")
            plt.close()
            
            results[series_id] = series_result
            counter += 1
            
        except Exception as e:
            print(f"  Error testing CV impact for {series_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # Calculate overall results
    if results:
        # Calculate average improvement
        rmse_improvements = [r['improvement']['rmse_pct'] for r in results.values() if r['improvement']['rmse_pct'] is not None]
        mae_improvements = [r['improvement']['mae_pct'] for r in results.values() if r['improvement']['mae_pct'] is not None]
        mape_improvements = [r['improvement']['mape_pct'] for r in results.values() if r['improvement']['mape_pct'] is not None]
        
        overall_results = {
            'avg_rmse_improvement': sum(rmse_improvements) / len(rmse_improvements) if rmse_improvements else None,
            'avg_mae_improvement': sum(mae_improvements) / len(mae_improvements) if mae_improvements else None,
            'avg_mape_improvement': sum(mape_improvements) / len(mape_improvements) if mape_improvements else None,
            'series_tested': len(results),
            'series_with_improvement': sum(1 for r in results.values() if r['improvement']['rmse_pct'] and r['improvement']['rmse_pct'] > 0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add overall results to the dictionary
        results['_summary'] = overall_results
        
        # Print overall results
        print("\nOverall CV test results:")
        print(f"Series tested: {overall_results['series_tested']}")
        print(f"Series with RMSE improvement: {overall_results['series_with_improvement']} ({overall_results['series_with_improvement'] / overall_results['series_tested'] * 100:.1f}%)")
        
        if overall_results['avg_rmse_improvement']:
            print(f"Average RMSE improvement: {overall_results['avg_rmse_improvement']:.2f}%")
        if overall_results['avg_mae_improvement']:
            print(f"Average MAE improvement: {overall_results['avg_mae_improvement']:.2f}%") 
        if overall_results['avg_mape_improvement']:
            print(f"Average MAPE improvement: {overall_results['avg_mape_improvement']:.2f}%")
            
        # Save results to JSON
        with open(CV_TEST_DIR / "cv_test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nCV test results saved to {CV_TEST_DIR / 'cv_test_results.json'}")
    
    return results


def run_prophet_modeling(data_path: str, top_n: int = None, resume: bool = True, check_outputs: bool = False):
    """
    Run Prophet modeling for all series in the dataset.
    
    Args:
        data_path: Path to the parquet file with time series data
        top_n: Number of top series to process (by volume)
        resume: Whether to resume from where execution was stopped previously
        check_outputs: Whether to use output files to determine what to skip (more reliable)
    """
    # Load data as PyArrow table
    print(f"Loading time series data from {data_path}")
    table = load_time_series_data(data_path)
    
    # Get series to process
    if top_n is not None:
        # Get top N series by volume
        series_list = get_top_n_series(table, top_n)
        print(f"Processing top {top_n} series by volume")
    else:
        # Get all unique series
        series_list = get_unique_series(table)
        print(f"Processing all {len(series_list)} series")
    
    # Check for existing results if resuming
    existing_results = []
    existing_series_ids = set()
    results_path = Path(config.DATA_DIR) / "prophet_modeling_results.json"
    
    if resume:
        # Check previous result file
        if results_path.exists():
            try:
                with open(results_path, 'r') as f:
                    existing_results = json.load(f)
                    
                # Get series IDs that have already been processed
                for result in existing_results:
                    if 'series_id' in result:
                        existing_series_ids.add(result['series_id'])
                        
                print(f"Found {len(existing_results)} existing results, {len(existing_series_ids)} unique series")
            except Exception as e:
                print(f"Error loading existing results: {str(e)}")
                existing_results = []
                existing_series_ids = set()
        
        # Filter out series that are fully processed
        if resume:
            remaining_series = []
            for series_info in series_list:
                series_id = f"{series_info['establecimiento']}_{series_info['material']}"
                
                # Check if this series has all output files
                forecast_plot_path = PLOTS_DIR / f"{series_id}_forecast.png"
                components_plot_path = PLOTS_DIR / f"{series_id}_components.png"
                forecast_path = FORECASTS_DIR / f"{series_id}_forecast.parquet"
                metadata_path = MODELS_DIR / f"{series_id}_metadata.json"
                
                # Only skip if ALL files exist, otherwise reprocess
                if (forecast_plot_path.exists() and 
                    components_plot_path.exists() and 
                    forecast_path.exists() and 
                    metadata_path.exists()):
                    continue
                
                # Include series that need processing
                remaining_series.append(series_info)
                
            skipped = len(series_list) - len(remaining_series)
            print(f"Skipping {skipped} already fully processed series, processing {len(remaining_series)} remaining series")
            series_list = remaining_series
    
    # Track results
    results = []
    
    # Process each series
    for series_info in tqdm(series_list):
        establecimiento = series_info['establecimiento']
        material = series_info['material']
        series_id = f"{establecimiento}_{material}"
        
        print(f"\n\nProcessing series: {series_id}")
        
        # Additional check for specific files before processing
        forecast_plot_path = PLOTS_DIR / f"{series_id}_forecast.png"
        components_plot_path = PLOTS_DIR / f"{series_id}_components.png"
        forecast_path = FORECASTS_DIR / f"{series_id}_forecast.parquet"
        metadata_path = MODELS_DIR / f"{series_id}_metadata.json"
        
        if resume and forecast_plot_path.exists() and components_plot_path.exists() and forecast_path.exists() and metadata_path.exists():
            print(f"Skipping {series_id} - output files already exist")
            
            # Try to load metadata to include in results
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Create a result entry from metadata
                result_entry = {
                    'series_id': series_id,
                    'success': True,
                    'data_points': metadata.get('data_points'),
                    'evaluation': metadata.get('evaluation', {})
                }
                results.append(result_entry)
                continue
            except Exception as e:
                print(f"Error loading metadata for {series_id}: {str(e)}. Will reprocess.")
        
        # Train model
        model_result = train_prophet_model(table, series_info, use_regressors=True)
        
        # If model training was successful
        if model_result['success']:
            print(f"Model training successful for {series_id}")
            
            # Generate forecast
            forecast_table = generate_forecast(model_result)
            
            # Evaluate model
            print(f"Evaluating model for {series_id}...")
            evaluation = evaluate_model(model_result)
            
            if evaluation:
                print(f"Evaluation complete for {series_id}")
                if 'error' in evaluation:
                    print(f"Evaluation had error: {evaluation['error']}")
            else:
                print(f"Evaluation failed for {series_id}")
                evaluation = {'error': 'Evaluation failed', 'mape': None, 'rmse': None}
            
            # Save results
            save_model_results(model_result, forecast_table, evaluation)
            
            # Track result
            results.append({
                'series_id': series_id,
                'success': True,
                'data_points': model_result['data_points'],
                'evaluation': evaluation
            })
        else:
            print(f"Model training failed for {series_id}: {model_result.get('error', 'Unknown error')}")
            
            # Track failure
            results.append({
                'series_id': series_id,
                'success': False,
                'error': model_result.get('error', 'Unknown error')
            })
    
    # Combine with existing results if resuming
    if resume and existing_results:
        combined_results = existing_results + results
        print(f"Combined {len(existing_results)} existing results with {len(results)} new results")
        results = combined_results
    
    # Save overall results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save focused metrics summary
    print("\nGenerating metrics summary...")
    metrics_summary = save_metrics_summary(results)
    
    # Calculate global metrics
    try:
        global_metrics = calculate_global_metrics(results, data_path)
        
        # Add global metrics to the summary
        if '_summary' in metrics_summary and isinstance(metrics_summary['_summary'], dict):
            metrics_summary['_summary']['global_metrics'] = global_metrics
            
            # Save updated summary
            with open(Path(config.DATA_DIR) / "prophet_metrics_summary.json", 'w') as f:
                json.dump(metrics_summary, f, indent=2)
                
    except Exception as e:
        print(f"Error calculating global metrics: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nProcessed {len(results)} series: {successful} successful, {len(results) - successful} failed")
    
    # Print metrics summary if available
    if '_summary' in metrics_summary and metrics_summary['_summary'].get('avg_mape') is not None:
        summary = metrics_summary['_summary']
        print(f"\nModel Performance Summary:")
        print(f"Average MAPE: {summary['avg_mape']:.2f}%")
        print(f"Average RMSE: {summary['avg_rmse']:.2f}")
    else:
        print("\nNo valid metrics summary available. Check log for errors.")
    
    return results


def list_materials_from_parquet(file_path: str):
    """
    List all unique materials from a parquet file.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        List of unique material IDs
    """
    # Create a connection to use for the query
    con = duckdb.connect()
    try:
        # Query to get distinct materials
        materials = con.sql(f"""
            SELECT DISTINCT material
            FROM read_parquet('{file_path}')
            ORDER BY material
        """).fetchall()
        
        # Extract material IDs from result tuples
        material_ids = [m[0] for m in materials]
        return material_ids
    finally:
        con.close()


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Prophet modeling on time series data')
    parser.add_argument('--top', type=int, default=None, help='Process only the top N series by volume')
    parser.add_argument('--no-resume', action='store_true', help='Start from scratch (do not resume from previous run)')
    parser.add_argument('--check-outputs', action='store_true', help='Use output files to determine what to skip (more reliable)')
    parser.add_argument('--cv-test', action='store_true', help='Test impact of cross-validation parameter optimization')
    parser.add_argument('--cv-test-series', type=int, default=5, help='Number of series to test for CV impact (default: 5)')
    args = parser.parse_args()
    
    # Path to weekly time series data
    weekly_data_path = str(Path(config.DATA_DIR) / config.GOLD_WEEKLY_TRAINING_PATH)
    
    # If testing CV impact
    if args.cv_test:
        # Load data
        table = load_time_series_data(weekly_data_path)
        print(table.head(5))
        # Get series to test
        if args.top is not None:
            series_list = get_top_n_series(table, args.top)
        else:
            # Use top 20 series if no limit specified
            series_list = get_top_n_series(table, 20)
            
        # Run CV test
        cv_test_results = test_cv_impact(table, series_list, args.cv_test_series)
        
    else:
        # Run Prophet modeling with resume functionality
        results = run_prophet_modeling(
            weekly_data_path, 
            top_n=args.top, 
            resume=not args.no_resume,
            check_outputs=args.check_outputs
        )
