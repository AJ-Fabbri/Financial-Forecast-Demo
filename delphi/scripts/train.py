#!/usr/bin/env python
"""
Train ARIMA models for stock price prediction
Adapted for demonstration purposes
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..data.loader import load_ticker_data, prepare_data_for_arima
from ..models.arima_model import train_arima_model
from ..config import DEFAULT_TICKER, FORECAST_HORIZON, TEST_SIZE, RESULTS_DIR
from ..utils import save_to_excel, create_results_summary


def train_ticker_model(ticker=None, years=5, target_col='Close', 
                      auto_order=True, save_model=True, plot_results=True):
    """
    Train an ARIMA model for a specific ticker
    
    Args:
        ticker: Stock ticker symbol (default: SPY)
        years: Number of years of historical data
        target_col: Target column for prediction
        auto_order: Whether to automatically find optimal ARIMA order
        save_model: Whether to save the trained model
        plot_results: Whether to create diagnostic plots
        
    Returns:
        Trained ARIMA model
    """
    if ticker is None:
        ticker = DEFAULT_TICKER
    
    print(f"Training ARIMA model for {ticker}")
    print("=" * 50)
    
    try:
        # Load data
        print("Loading data...")
        df = load_ticker_data(ticker, years=years)
        
        # Prepare data for ARIMA
        train_data, test_data, train_dates, test_dates = prepare_data_for_arima(
            df, target_column=target_col, test_size=TEST_SIZE
        )
        
        # Train model
        print("\nTraining ARIMA model...")
        model = train_arima_model(
            train_data, 
            ticker=ticker,
            auto_order=auto_order,
            save_model=save_model,
            verbose=True
        )
        
        # Display model summary
        print("\nModel Summary:")
        print("=" * 50)
        print(model.get_model_summary())
        
        # Create diagnostic plots if requested
        if plot_results:
            print("\nGenerating diagnostic plots...")
            
            # Model diagnostics
            diag_fig = model.plot_diagnostics()
            diag_path = os.path.join(RESULTS_DIR, f"{ticker}_arima_diagnostics.png")
            diag_fig.savefig(diag_path, dpi=300, bbox_inches='tight')
            print(f"Diagnostic plots saved to: {diag_path}")
            plt.show()
        
        # Save training summary
        model_params = {
            'order': str(model.order),
            'aic': model.fitted_model.aic,
            'bic': model.fitted_model.bic,
            'data_points': len(train_data),
            'years': years
        }
        
        summary_df = create_results_summary(
            ticker, 
            {'Training_AIC': model.fitted_model.aic, 'Training_BIC': model.fitted_model.bic},
            model_params
        )
        
        # Save to Excel
        excel_path = os.path.join(RESULTS_DIR, f"{ticker}_training_summary.xlsx")
        save_to_excel(summary_df, excel_path, ticker=ticker)
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved for ticker: {ticker}")
        print(f"Training summary saved to: {excel_path}")
        
        return model
        
    except Exception as e:
        print(f"Error training model for {ticker}: {e}")
        raise e


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Train ARIMA model for stock price prediction',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('ticker', nargs='?', default=DEFAULT_TICKER,
                       help=f'Stock ticker symbol (default: {DEFAULT_TICKER})')
    parser.add_argument('--years', type=int, default=5,
                       help='Number of years of historical data (default: 5)')
    parser.add_argument('--target', type=str, default='Close',
                       help='Target column for prediction (default: Close)')
    parser.add_argument('--no-auto-order', action='store_false', dest='auto_order',
                       help='Disable automatic ARIMA order selection')
    parser.add_argument('--no-save', action='store_false', dest='save_model',
                       help='Do not save the trained model')
    parser.add_argument('--no-plot', action='store_false', dest='plot_results',
                       help='Do not create diagnostic plots')
    
    args = parser.parse_args()
    
    # Train model
    train_ticker_model(
        ticker=args.ticker,
        years=args.years,
        target_col=args.target,
        auto_order=args.auto_order,
        save_model=args.save_model,
        plot_results=args.plot_results
    )


if __name__ == '__main__':
    main() 