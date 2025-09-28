"""
Utility functions for plotting, evaluation, and saving results
Adapted for ARIMA modeling in Financial Forecast Demo
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import pickle
from openpyxl.utils.dataframe import dataframe_to_rows

from .config import RESULTS_DIR, SAVED_MODELS_DIR


def plot_predictions(y_true, y_pred, title='ARIMA Model Predictions', dates=None, 
                    confidence_intervals=None):
    """
    Plot actual vs predicted values with optional confidence intervals
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        dates: Optional list of dates for x-axis
        confidence_intervals: Optional tuple of (lower, upper) confidence bounds
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(12, 6))
    
    if dates is not None:
        plt.plot(dates, y_true, 'b-', label='Actual', linewidth=2)
        plt.plot(dates, y_pred, 'r--', label='Predicted', linewidth=2)
        
        # Add confidence intervals if provided
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            plt.fill_between(dates, lower, upper, alpha=0.3, color='red', 
                           label='95% Confidence Interval')
    else:
        plt.plot(y_true, 'b-', label='Actual', linewidth=2)
        plt.plot(y_pred, 'r--', label='Predicted', linewidth=2)
        
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            plt.fill_between(range(len(y_pred)), lower, upper, alpha=0.3, 
                           color='red', label='95% Confidence Interval')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Price ($)', fontsize=12)
    plt.xlabel('Date' if dates is not None else 'Time', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if dates is not None:
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    return plt.gcf()


def plot_forecast(historical_data, forecast_data, forecast_dates, 
                 confidence_intervals=None, title='ARIMA Forecast'):
    """
    Plot historical data with forecast
    
    Args:
        historical_data: Historical price data
        forecast_data: Forecasted values
        forecast_dates: Dates for forecast
        confidence_intervals: Optional confidence intervals
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(14, 8))
    
    # Plot historical data (last 60 days for context)
    hist_data = historical_data.tail(60)
    plt.plot(hist_data.index, hist_data.values, 'b-', label='Historical', 
             linewidth=2)
    
    # Plot forecast
    plt.plot(forecast_dates, forecast_data, 'r--', label='Forecast', 
             linewidth=2, marker='o')
    
    # Add confidence intervals if provided
    if confidence_intervals is not None:
        lower, upper = confidence_intervals
        plt.fill_between(forecast_dates, lower, upper, alpha=0.3, 
                        color='red', label='95% Confidence Interval')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Price ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    return plt.gcf()


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance with multiple metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Ensure arrays are flattened
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Calculate additional financial metrics
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100
    
    # Direction accuracy (for financial predictions)
    true_direction = np.diff(y_true_flat) > 0
    pred_direction = np.diff(y_pred_flat) > 0
    direction_accuracy = np.mean(true_direction == pred_direction) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape,
        'Direction Accuracy (%)': direction_accuracy
    }


def evaluate_model_dual_horizon(y_true_1d, y_pred_1d, y_true_5d=None, y_pred_5d=None):
    """
    Evaluate model performance with dual-horizon metrics (1-day and 5-day ahead)
    
    Args:
        y_true_1d: True values for 1-day ahead
        y_pred_1d: Predicted values for 1-day ahead
        y_true_5d: True values for 5-day ahead (optional)
        y_pred_5d: Predicted values for 5-day ahead (optional)
        
    Returns:
        Dictionary with metrics for both 1-day and 5-day ahead
    """
    results = {}
    
    # 1-day ahead metrics
    metrics_1d = evaluate_model(y_true_1d, y_pred_1d)
    results['1d'] = metrics_1d
    
    # 5-day ahead metrics (if provided)
    if y_true_5d is not None and y_pred_5d is not None:
        metrics_5d = evaluate_model(y_true_5d, y_pred_5d)
        results['5d'] = metrics_5d
    
    return results


def save_dual_horizon_to_excel(metrics_dict, ticker, output_file=None, verbose=True):
    """
    Save dual-horizon evaluation metrics to Excel in a clean format
    
    Args:
        metrics_dict: Dictionary containing dual-horizon metrics
        ticker: Stock ticker symbol
        output_file: Path to output file (if None, uses default)
        verbose: Whether to print confirmation
        
    Returns:
        Path to saved file
    """
    # Create separate rows for each horizon
    rows = []
    
    # Extract horizons from metrics keys
    horizons = set()
    for key in metrics_dict.keys():
        if '_' in key and (key.startswith('1d_') or key.startswith('5d_')):
            horizon = key.split('_')[0]
            horizons.add(horizon)
    
    # If no horizon-specific metrics found, fall back to single row
    if not horizons:
        row = {'Ticker': ticker, 'Horizon': '1d'}  # Default to 1d for Demo
        row.update(metrics_dict)
        rows.append(row)
    else:
        # Create one row per horizon
        for horizon in sorted(horizons):
            row = {'Ticker': ticker, 'Horizon': horizon}
            
            # Add horizon-specific metrics
            for key, value in metrics_dict.items():
                if key.startswith(f'{horizon}_'):
                    metric_name = key[3:]  # Remove horizon prefix
                    row[metric_name] = value
                elif '_' not in key or not any(key.startswith(h + '_') for h in horizons):
                    # Add non-horizon-specific metrics (like trading metrics)
                    row[key] = value
            
            rows.append(row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(rows)
    
    # Use the regular save_to_excel function
    return save_to_excel(results_df, output_file, ticker, verbose, append_rows=True)


def save_to_excel(results_df: pd.DataFrame, output_file=None, ticker=None, 
                 verbose=True, append_rows=False):
    """
    Save results to Excel file
    
    Args:
        results_df: DataFrame with results
        output_file: Path to output file (if None, a default path will be used)
        ticker: Stock ticker symbol (used in default filename)
        verbose: Whether to print the save confirmation
        append_rows: If True, append rows to existing sheet
        
    Returns:
        Path to saved file
    """
    if output_file is None:
        filename = f"{ticker}_results.xlsx" if ticker else "results.xlsx"
        output_file = os.path.join(RESULTS_DIR, filename)
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    sheet_name = ticker if ticker else "Results"
    
    try:
        if append_rows and os.path.exists(output_file):
            # Try to append to existing file
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', 
                              if_sheet_exists='overlay') as writer:
                results_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Create new file or overwrite
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
                results_df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        # Fallback to CSV
        csv_file = output_file.replace('.xlsx', '.csv')
        results_df.to_csv(csv_file, index=False)
        output_file = csv_file
        print(f"Saved as CSV instead: {csv_file}")
    
    if verbose:
        print(f"Results saved to {output_file}")
    return output_file


def format_metrics_for_display(metrics):
    """
    Format metrics dictionary for display
    
    Args:
        metrics: Dictionary of evaluation metrics
        
    Returns:
        Formatted string
    """
    return "\n".join([f"{name}: {value:.6f}" for name, value in metrics.items()])


def save_arima_model(model, model_name, model_params=None):
    """
    Save an ARIMA model using pickle
    
    Args:
        model: Fitted ARIMA model
        model_name: Name for the model (typically ticker symbol)
        model_params: Optional dictionary of model parameters
        
    Returns:
        Path to the saved model
    """
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    
    filename = f"{model_name}_arima.pkl"
    model_path = os.path.join(SAVED_MODELS_DIR, filename)
    
    # Save model and parameters
    model_data = {
        'model': model,
        'params': model_params or {},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"ARIMA model saved at: {model_path}")
    return model_path


def load_arima_model(model_name):
    """
    Load an ARIMA model from pickle file
    
    Args:
        model_name: Name of the model to load (typically ticker symbol)
        
    Returns:
        Tuple of (model, parameters)
    """
    model_path = os.path.join(SAVED_MODELS_DIR, f"{model_name}_arima.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No ARIMA model found for {model_name} at {model_path}")
    
    print(f"Loading ARIMA model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data.get('params', {})


def create_results_summary(ticker, metrics, model_params=None):
    """
    Create a summary DataFrame of results
    
    Args:
        ticker: Stock ticker symbol
        metrics: Dictionary of evaluation metrics
        model_params: Optional model parameters
        
    Returns:
        DataFrame with summary
    """
    summary_data = {
        'Ticker': [ticker],
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        **{k: [v] for k, v in metrics.items()}
    }
    
    if model_params:
        summary_data.update({f"Param_{k}": [v] for k, v in model_params.items()})
    
    return pd.DataFrame(summary_data) 