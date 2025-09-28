#!/usr/bin/env python
"""
Batch processing functions for Financial Forecast Demo
Mirrors the functionality of the original batch processing with ARIMA models
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .train import train_ticker_model
from .evaluate import evaluate_ticker_model
from .predict import predict_future
from ..data.database import get_database, update_ticker_from_yfinance
from ..config import DEFAULT_TICKER, RESULTS_DIR
from ..utils import save_to_excel


def ensure_data_updated(tickers):
    """
    Ensure all tickers have up-to-date data in the database
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Dictionary of update results
    """
    print(f"🔄 PRE-BATCH UPDATE: Checking and updating data for {len(tickers)} tickers...")
    
    update_results = {}
    
    for ticker in tqdm(tickers, desc="Pre-batch data updates"):
        try:
            # Check if ticker needs updating
            db = get_database()
            tickers_in_db = db.list_tickers()
            
            if ticker not in tickers_in_db:
                print(f" {ticker}: New ticker - adding to database")
                success = update_ticker_from_yfinance(ticker, years=10)
                update_results[ticker] = 'added' if success else 'failed'
            else:
                # For existing tickers, we could add logic to check if data is stale
                # For now, assume data is current
                update_results[ticker] = 'current'
            
            db.close()
            
        except Exception as e:
            print(f" Error updating {ticker}: {e}")
            update_results[ticker] = 'error'
    
    successful_updates = sum(1 for result in update_results.values() if result in ['added', 'current'])
    print(f" PRE-BATCH UPDATE COMPLETE: {successful_updates}/{len(tickers)} successful")
    
    return update_results


def process_tickers(tickers, processing_function, max_workers=None, **kwargs):
    """
    Process multiple tickers in parallel using ThreadPoolExecutor
    
    Args:
        tickers: List of ticker symbols
        processing_function: Function to apply to each ticker
        max_workers: Maximum number of parallel workers
        **kwargs: Additional arguments for processing function
        
    Returns:
        Dictionary of results for each ticker
    """
    if max_workers is None:
        max_workers = min(4, len(tickers))  # Conservative default
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(processing_function, ticker, **kwargs): ticker 
            for ticker in tickers
        }
        
        # Process completed tasks
        for future in tqdm(as_completed(future_to_ticker), 
                          total=len(tickers), 
                          desc="Processing tickers"):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                results[ticker] = result
                print(f" Completed {ticker}")
                sys.stdout.flush()
            except Exception as exc:
                print(f" {ticker} generated an exception: {exc}")
                sys.stdout.flush()
                results[ticker] = None
    
    return results


def batch_train(tickers, sequential=False, max_workers=None, years=5, auto_order=True, **kwargs):
    """
    Train ARIMA models for multiple tickers with optimized data loading
    
    Args:
        tickers: List of ticker symbols
        sequential: Boolean, if True process tickers sequentially
        max_workers: Maximum number of parallel workers (None = auto)
        years: Number of years of historical data
        auto_order: Whether to use automatic ARIMA order selection
        **kwargs: Additional arguments to pass to train_ticker_model
        
    Returns:
        Dictionary of training results for each ticker
    """
    print(f" Training models for {len(tickers)} tickers: {', '.join(tickers)}")
    
    # STEP 1: Ensure all data is updated SEQUENTIALLY
    update_results = ensure_data_updated(tickers)
    
    # Add common parameters to kwargs
    kwargs.update({
        'years': years,
        'auto_order': auto_order,
        'save_model': True,
        'plot_results': False  # Disable plots for batch processing
    })
    
    if sequential:
        # Process sequentially
        results = {}
        for ticker in tqdm(tickers, desc="Training models"):
            try:
                result = train_ticker_model(ticker, **kwargs)
                results[ticker] = result
                print(f" Completed training for {ticker}")
                sys.stdout.flush()
            except Exception as exc:
                print(f" {ticker} generated an exception: {exc}")
                sys.stdout.flush()
                results[ticker] = None
    else:
        # STEP 2: Use parallel processing for training
        print("🔄 Starting parallel training...")
        results = process_tickers(tickers, train_ticker_model, max_workers=max_workers, **kwargs)
    
    # STEP 3: Create summary report
    successful_trains = sum(1 for result in results.values() if result is not None)
    print(f"\n BATCH TRAINING SUMMARY:")
    print(f" Successfully trained: {successful_trains}/{len(tickers)} models")
    
    # Create training summary DataFrame
    summary_data = {}
    for ticker, result in results.items():
        if result is not None:
            summary_data[ticker] = {
                'Status': 'Success',
                'Model_Order': str(result.order) if hasattr(result, 'order') else 'Unknown',
                'AIC': result.fitted_model.aic if hasattr(result, 'fitted_model') else 'Unknown',
                'BIC': result.fitted_model.bic if hasattr(result, 'fitted_model') else 'Unknown'
            }
        else:
            summary_data[ticker] = {
                'Status': 'Failed',
                'Model_Order': 'N/A',
                'AIC': 'N/A',
                'BIC': 'N/A'
            }
    
    # Save batch summary
    if summary_data:
        summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
        summary_path = os.path.join(RESULTS_DIR, 'batch_training_summary.xlsx')
        summary_df.to_excel(summary_path)
        print(f"📁 Batch training summary saved to: {summary_path}")
    
    return results


def batch_evaluate(tickers, sequential=False, max_workers=None, years=5, include_simulation=True, **kwargs):
    """
    Evaluate models for multiple tickers with trading simulation
    
    Args:
        tickers: List of ticker symbols
        sequential: Boolean, if True process tickers sequentially
        max_workers: Maximum number of parallel workers (None = auto)
        years: Number of years of historical data
        include_simulation: Whether to include trading simulation
        **kwargs: Additional arguments to pass to evaluate_ticker_model
        
    Returns:
        Dictionary of evaluation metrics for each ticker
    """
    print(f" Evaluating models for {len(tickers)} tickers: {', '.join(tickers)}")
    
    # STEP 1: Ensure all data is updated SEQUENTIALLY
    update_results = ensure_data_updated(tickers)
    
    # Add common parameters to kwargs
    kwargs.update({
        'years': years,
        'include_simulation': include_simulation,
        'plot_results': False,  # Disable plots for batch processing
        'save_results': True
    })
    
    if sequential:
        # Process sequentially
        results = {}
        for ticker in tqdm(tickers, desc="Evaluating models"):
            try:
                result = evaluate_ticker_model(ticker, **kwargs)
                results[ticker] = result
                print(f" Completed evaluation for {ticker}")
                sys.stdout.flush()
            except Exception as exc:
                print(f" {ticker} generated an exception: {exc}")
                sys.stdout.flush()
                results[ticker] = None
    else:
        # STEP 2: Use parallel processing for evaluation
        print("🔄 Starting parallel evaluation...")
        results = process_tickers(tickers, evaluate_ticker_model, max_workers=max_workers, **kwargs)
    
    # STEP 3: Create comprehensive summary report
    successful_evals = sum(1 for result in results.values() if result is not None)
    print(f"\n BATCH EVALUATION SUMMARY:")
    print(f" Successfully evaluated: {successful_evals}/{len(tickers)} models")
    
    # Create evaluation summary DataFrame with dual-horizon support
    summary_data = {}
    for ticker, result in results.items():
        if result is not None and isinstance(result, dict):
            # Check if we have dual-horizon metrics
            has_dual_horizon = any(key.startswith('1d_') or key.startswith('5d_') for key in result.keys())
            
            if has_dual_horizon:
                # Use dual-horizon format
                summary_data[ticker] = {
                    'Status': 'Success',
                    '1d_RMSE': result.get('1d_RMSE', 0),
                    '1d_R²': result.get('1d_R²', 0),
                    '1d_MAPE_%': result.get('1d_MAPE (%)', 0),
                    '1d_Direction_Accuracy_%': result.get('1d_Direction Accuracy (%)', 0),
                    '5d_RMSE': result.get('5d_RMSE', 0),
                    '5d_R²': result.get('5d_R²', 0),
                    '5d_MAPE_%': result.get('5d_MAPE (%)', 0),
                    '5d_Direction_Accuracy_%': result.get('5d_Direction Accuracy (%)', 0),
                    'Strategy_Return_%': result.get('Total_Return', 0) if include_simulation else 'N/A',
                    'BuyHold_Return_%': result.get('Total_Return_BH', 0) if include_simulation else 'N/A',
                    'Alpha_%': result.get('Alpha', 0) if include_simulation else 'N/A',
                    'Sharpe_Ratio': result.get('Sharpe', 0) if include_simulation else 'N/A',
                    'Max_Drawdown_%': result.get('Max_Drawdown_%', 0) if include_simulation else 'N/A',
                    'Trading_Signals': result.get('Trading_Signals', 0) if include_simulation else 'N/A'
                }
            else:
                # Fallback to single-horizon format
                summary_data[ticker] = {
                    'Status': 'Success',
                    'RMSE': result.get('RMSE', 0),
                    'R²': result.get('R²', 0),
                    'MAPE_%': result.get('MAPE (%)', 0),
                    'Direction_Accuracy_%': result.get('Direction Accuracy (%)', 0),
                    'Strategy_Return_%': result.get('Total_Return', 0) if include_simulation else 'N/A',
                    'BuyHold_Return_%': result.get('Total_Return_BH', 0) if include_simulation else 'N/A',
                    'Alpha_%': result.get('Alpha', 0) if include_simulation else 'N/A',
                    'Sharpe_Ratio': result.get('Sharpe', 0) if include_simulation else 'N/A',
                    'Max_Drawdown_%': result.get('Max_Drawdown_%', 0) if include_simulation else 'N/A',
                    'Trading_Signals': result.get('Trading_Signals', 0) if include_simulation else 'N/A'
                }
        else:
            summary_data[ticker] = {
                'Status': 'Failed',
                '1d_RMSE': 'N/A',
                '1d_R²': 'N/A',
                '1d_MAPE_%': 'N/A',
                '1d_Direction_Accuracy_%': 'N/A',
                '5d_RMSE': 'N/A',
                '5d_R²': 'N/A',
                '5d_MAPE_%': 'N/A',
                '5d_Direction_Accuracy_%': 'N/A',
                'Strategy_Return_%': 'N/A',
                'BuyHold_Return_%': 'N/A',
                'Alpha_%': 'N/A',
                'Sharpe_Ratio': 'N/A',
                'Max_Drawdown_%': 'N/A',
                'Trading_Signals': 'N/A'
            }
    
    # Save batch summary
    if summary_data:
        summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
        summary_path = os.path.join(RESULTS_DIR, 'batch_evaluation_results.xlsx')
        summary_df.to_excel(summary_path)
        print(f"📁 Batch evaluation summary saved to: {summary_path}")
        
        # Display summary table
        print(f"\n EVALUATION RESULTS SUMMARY:")
        print("=" * 100)
        print(summary_df.to_string())
    
    return results


def batch_predict(tickers, sequential=False, max_workers=None, days_ahead=5, **kwargs):
    """
    Generate predictions for multiple tickers
    
    Args:
        tickers: List of ticker symbols
        sequential: Boolean, if True process tickers sequentially
        max_workers: Maximum number of parallel workers (None = auto)
        days_ahead: Number of days ahead to predict
        **kwargs: Additional arguments to pass to predict_future
        
    Returns:
        Dictionary of prediction results for each ticker
    """
    print(f" Making predictions for {len(tickers)} tickers: {', '.join(tickers)}")
    
    # STEP 1: Ensure all data is updated SEQUENTIALLY
    update_results = ensure_data_updated(tickers)
    
    # Add common parameters to kwargs
    kwargs.update({
        'days_ahead': days_ahead,
        'plot_results': False,  # Disable plots for batch processing
        'save_results': True,
        'verbose': False  # Reduce output for batch processing
    })
    
    if sequential:
        # Process sequentially
        results = {}
        for ticker in tqdm(tickers, desc="Generating predictions"):
            try:
                forecast, conf_int, dates = predict_future(ticker, **kwargs)
                results[ticker] = {
                    'forecast': forecast,
                    'confidence_intervals': conf_int,
                    'dates': dates,
                    'status': 'success'
                }
                print(f" Completed predictions for {ticker}")
                sys.stdout.flush()
            except Exception as exc:
                print(f" {ticker} generated an exception: {exc}")
                sys.stdout.flush()
                results[ticker] = {'status': 'failed', 'error': str(exc)}
    else:
        # STEP 2: Use parallel processing for predictions
        print("🔄 Starting parallel predictions...")
        
        def predict_wrapper(ticker, **kwargs):
            forecast, conf_int, dates = predict_future(ticker, **kwargs)
            return {
                'forecast': forecast,
                'confidence_intervals': conf_int,
                'dates': dates,
                'status': 'success'
            }
        
        results = process_tickers(tickers, predict_wrapper, max_workers=max_workers, **kwargs)
    
    # STEP 3: Create comprehensive summary report
    successful_preds = sum(1 for result in results.values() 
                          if result is not None and result.get('status') == 'success')
    print(f"\n BATCH PREDICTION SUMMARY:")
    print(f" Successfully generated predictions: {successful_preds}/{len(tickers)} tickers")
    
    # Create prediction summary DataFrame
    summary_data = {}
    for ticker, result in results.items():
        if result is not None and result.get('status') == 'success':
            forecast = result['forecast']
            dates = result['dates']
            conf_int = result['confidence_intervals']
            
            # Calculate prediction statistics
            forecast_mean = np.mean(forecast)
            forecast_std = np.std(forecast)
            trend = "Upward" if forecast[-1] > forecast[0] else "Downward"
            trend_magnitude = abs(forecast[-1] - forecast[0]) / forecast[0] * 100
            conf_range = conf_int.iloc[:, 1] - conf_int.iloc[:, 0]
            avg_conf_range = np.mean(conf_range)
            
            summary_data[ticker] = {
                'Status': 'Success',
                'Forecast_Start': f"${forecast[0]:.2f}",
                'Forecast_End': f"${forecast[-1]:.2f}",
                'Forecast_Mean': f"${forecast_mean:.2f}",
                'Forecast_Std': f"${forecast_std:.2f}",
                'Trend': trend,
                'Trend_Magnitude_%': f"{trend_magnitude:.2f}%",
                'Avg_Confidence_Range': f"${avg_conf_range:.2f}",
                'Forecast_Period': f"{dates[0].date()} to {dates[-1].date()}"
            }
        else:
            summary_data[ticker] = {
                'Status': 'Failed',
                'Forecast_Start': 'N/A',
                'Forecast_End': 'N/A',
                'Forecast_Mean': 'N/A',
                'Forecast_Std': 'N/A',
                'Trend': 'N/A',
                'Trend_Magnitude_%': 'N/A',
                'Avg_Confidence_Range': 'N/A',
                'Forecast_Period': 'N/A'
            }
    
    # Save batch summary
    if summary_data:
        summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
        summary_path = os.path.join(RESULTS_DIR, 'batch_prediction_results.xlsx')
        summary_df.to_excel(summary_path)
        print(f"📁 Batch prediction summary saved to: {summary_path}")
        
        # Display summary table
        print(f"\n PREDICTION RESULTS SUMMARY:")
        print("=" * 120)
        print(summary_df.to_string())
    
    return results


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch processing for Financial Forecast Demo')
    parser.add_argument('command', choices=['train', 'evaluate', 'predict'], 
                       help='Command to run')
    parser.add_argument('tickers', nargs='+', help='Ticker symbols to process')
    parser.add_argument('--sequential', action='store_true', 
                       help='Process sequentially instead of parallel')
    parser.add_argument('--max-workers', type=int, help='Maximum parallel workers')
    parser.add_argument('--years', type=int, default=5, help='Years of data')
    parser.add_argument('--days-ahead', type=int, default=5, help='Days ahead for prediction')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        results = batch_train(
            args.tickers, 
            sequential=args.sequential,
            max_workers=args.max_workers,
            years=args.years
        )
    elif args.command == 'evaluate':
        results = batch_evaluate(
            args.tickers,
            sequential=args.sequential, 
            max_workers=args.max_workers,
            years=args.years
        )
    elif args.command == 'predict':
        results = batch_predict(
            args.tickers,
            sequential=args.sequential,
            max_workers=args.max_workers, 
            days_ahead=args.days_ahead
        )
    
    print(f"\n Batch {args.command} completed!")


if __name__ == '__main__':
    main() 