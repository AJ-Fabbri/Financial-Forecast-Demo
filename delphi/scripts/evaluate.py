#!/usr/bin/env python
"""
Evaluate ARIMA models for stock price prediction
Adapted for demonstration purposes
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..data.loader import load_ticker_data, prepare_data_for_arima
from ..models.arima_model import evaluate_arima_model, evaluate_arima_with_simulation
from ..config import DEFAULT_TICKER, TEST_SIZE, RESULTS_DIR
from ..utils import load_arima_model, save_to_excel, create_results_summary


def evaluate_ticker_model(ticker=None, years=5, target_col='Close', 
                         plot_results=True, save_results=True, include_simulation=True):
    """
    Evaluate a trained ARIMA model for a specific ticker
    
    Args:
        ticker: Stock ticker symbol (default: SPY)
        years: Number of years of historical data
        target_col: Target column for prediction
        plot_results: Whether to create evaluation plots
        save_results: Whether to save evaluation results
        include_simulation: Whether to include trading simulation analysis
        
    Returns:
        Dictionary of evaluation metrics
    """
    if ticker is None:
        ticker = DEFAULT_TICKER
    
    print(f"Evaluating ARIMA model for {ticker}")
    print("=" * 50)
    
    try:
        # Load the trained model
        print("Loading trained model...")
        model_obj, model_params = load_arima_model(ticker)
        
        print(f"Loaded model with order: {model_params.get('order', 'Unknown')}")
        print(f"Training AIC: {model_params.get('aic', 'Unknown')}")
        
        # Load fresh data for evaluation
        print("\nLoading evaluation data...")
        df = load_ticker_data(ticker, years=years)
        
        # Prepare data for evaluation
        train_data, test_data, train_dates, test_dates = prepare_data_for_arima(
            df, target_column=target_col, test_size=TEST_SIZE
        )
        
        print(f"Evaluation set size: {len(test_data)} samples")
        
        if include_simulation:
            print("\n Evaluating model with trading simulation (matching main methodology)...")
            print("   - Generates buy/sell signals based on ARIMA predictions")
            print("   - Backtests trading strategy vs buy-and-hold")
            print("   - Calculates Sharpe ratios and drawdown metrics")
            print()
            
            # Use the new simulation-enhanced evaluation
            metrics = evaluate_arima_with_simulation(
                model_obj, 
                test_data, 
                test_dates=test_dates,
                plot_results=plot_results,
                ticker=ticker,
                verbose=True,
                forecast_horizon=5
            )
        else:
            print("\nEvaluating model performance (basic metrics only)...")
            # Use the original basic evaluation
            metrics = evaluate_arima_model(
                model_obj, 
                test_data, 
                test_dates=test_dates,
                plot_results=plot_results,
                ticker=ticker,
                verbose=True
            )
        
        # Save results if requested
        if save_results:
            print("\nSaving evaluation results...")
            
            # Create comprehensive results summary
            eval_summary = create_results_summary(
                ticker, 
                metrics,
                {
                    **model_params,
                    'evaluation_samples': len(test_data),
                    'evaluation_period': f"{test_dates[0].date()} to {test_dates[-1].date()}",
                    'simulation_included': include_simulation
                }
            )
            
            # Save to Excel with dual-horizon format
            from ..utils import save_dual_horizon_to_excel
            excel_path = os.path.join(RESULTS_DIR, f"{ticker}_dual_horizon_evaluation.xlsx")
            save_dual_horizon_to_excel(metrics, ticker, output_file=excel_path, verbose=True)
            print(f"Dual-horizon evaluation results saved to: {excel_path}")
            
            # Also save traditional format for backward compatibility
            excel_path_traditional = os.path.join(RESULTS_DIR, f"{ticker}_evaluation_results.xlsx")
            save_to_excel(eval_summary, excel_path_traditional, ticker=f"{ticker}_evaluation")
            print(f"Traditional evaluation results saved to: {excel_path_traditional}")
        
        print(f"\nEvaluation completed successfully!")
        if include_simulation:
            print("\n Trading Simulation Summary:")
            print("=" * 35)
            for metric, value in metrics.items():
                if any(key in metric for key in ['Total_Return', 'Alpha', 'Sharpe', 'Max_Drawdown', 'Trading_Signals']):
                    if 'Trading_Signals' in metric:
                        print(f"{metric}: {value}")
                    elif '%' in metric or 'Return' in metric or 'Alpha' in metric:
                        print(f"{metric}: {value:.2f}%")
                    else:
                        print(f"{metric}: {value:.4f}")
        else:
            print("\nBasic Metrics Summary:")
            print("=" * 25)
            for metric, value in metrics.items():
                if 'Accuracy' in metric or '%' in metric:
                    print(f"{metric}: {value:.2f}%")
                else:
                    print(f"{metric}: {value:.6f}")
        
        return metrics
        
    except FileNotFoundError as e:
        print(f"Error: No trained model found for {ticker}")
        print("Please train a model first using the train command")
        raise e
    except Exception as e:
        print(f"Error evaluating model for {ticker}: {e}")
        raise e


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained ARIMA model for stock price prediction',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('ticker', nargs='?', default=DEFAULT_TICKER,
                       help=f'Stock ticker symbol (default: {DEFAULT_TICKER})')
    parser.add_argument('--years', type=int, default=5,
                       help='Number of years of historical data (default: 5)')
    parser.add_argument('--target', type=str, default='Close',
                       help='Target column for prediction (default: Close)')
    parser.add_argument('--no-plot', action='store_false', dest='plot_results',
                       help='Do not create evaluation plots')
    parser.add_argument('--no-save', action='store_false', dest='save_results',
                       help='Do not save evaluation results')
    parser.add_argument('--no-simulation', action='store_false', dest='include_simulation',
                       help='Skip trading simulation analysis')
    
    args = parser.parse_args()
    
    # Evaluate model
    evaluate_ticker_model(
        ticker=args.ticker,
        years=args.years,
        target_col=args.target,
        plot_results=args.plot_results,
        save_results=args.save_results,
        include_simulation=args.include_simulation
    )


if __name__ == '__main__':
    main() 