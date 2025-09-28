#!/usr/bin/env python
"""
Generate predictions using trained ARIMA models
Adapted for demonstration purposes
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from ..data.loader import load_ticker_data, get_latest_price
from ..models.arima_model import forecast_future
from ..config import DEFAULT_TICKER, FORECAST_HORIZON, RESULTS_DIR
from ..utils import load_arima_model, save_to_excel


def predict_future(ticker=None, days_ahead=None, plot_results=True, 
                  save_results=True, verbose=True):
    """
    Generate future predictions using a trained ARIMA model
    
    Args:
        ticker: Stock ticker symbol (default: SPY)
        days_ahead: Number of days to predict (default from config)
        plot_results: Whether to create forecast plots
        save_results: Whether to save prediction results
        verbose: Whether to print detailed output
        
    Returns:
        Tuple of (forecast_values, confidence_intervals, forecast_dates)
    """
    if ticker is None:
        ticker = DEFAULT_TICKER
    
    if days_ahead is None:
        days_ahead = FORECAST_HORIZON
    
    if verbose:
        print(f"Generating {days_ahead}-day forecast for {ticker}")
        print("=" * 50)
    
    try:
        # Load the trained model
        if verbose:
            print("Loading trained model...")
        model_obj, model_params = load_arima_model(ticker)
        
        if verbose:
            print(f"Loaded model with order: {model_params.get('order', 'Unknown')}")
            print(f"Training AIC: {model_params.get('aic', 'Unknown')}")
        
        # Get current price for context
        try:
            current_price = get_latest_price(ticker)
            if verbose:
                print(f"Current {ticker} price: ${current_price:.2f}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not get current price: {e}")
            current_price = None
        
        # Generate forecast
        if verbose:
            print(f"\nGenerating {days_ahead} day forecast...")
        
        forecast_values, conf_intervals, forecast_dates = forecast_future(
            model_obj,
            steps=days_ahead,
            plot_results=plot_results,
            ticker=ticker
        )
        
        # Display forecast results
        if verbose:
            print("\nForecast Results:")
            print("=" * 50)
            print(f"{'Date':<12} {'Forecast':<10} {'Lower CI':<10} {'Upper CI':<10}")
            print("-" * 50)
            
            for i, date in enumerate(forecast_dates):
                forecast_val = forecast_values[i]
                lower_ci = conf_intervals.iloc[i, 0]
                upper_ci = conf_intervals.iloc[i, 1]
                
                print(f"{date.strftime('%Y-%m-%d'):<12} "
                      f"${forecast_val:<9.2f} "
                      f"${lower_ci:<9.2f} "
                      f"${upper_ci:<9.2f}")
        
        # Calculate forecast statistics
        forecast_mean = np.mean(forecast_values)
        forecast_std = np.std(forecast_values)
        forecast_trend = "Upward" if forecast_values[-1] > forecast_values[0] else "Downward"
        
        if current_price:
            price_change = forecast_values[-1] - current_price
            price_change_pct = (price_change / current_price) * 100
        else:
            price_change = None
            price_change_pct = None
        
        if verbose:
            print(f"\nForecast Summary:")
            print("=" * 30)
            print(f"Forecast period: {days_ahead} days")
            print(f"Average forecast: ${forecast_mean:.2f}")
            print(f"Forecast volatility: ${forecast_std:.2f}")
            print(f"Trend direction: {forecast_trend}")
            
            if current_price and price_change is not None:
                print(f"Expected price change: ${price_change:+.2f} ({price_change_pct:+.2f}%)")
        
        # Save results if requested
        if save_results:
            if verbose:
                print("\nSaving forecast results...")
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast_values,
                'Lower_CI': conf_intervals.iloc[:, 0].values,
                'Upper_CI': conf_intervals.iloc[:, 1].values,
                'Ticker': ticker,
                'Forecast_Horizon': days_ahead,
                'Generated_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Add current price context if available
            if current_price:
                forecast_df['Current_Price'] = current_price
                forecast_df['Price_Change'] = forecast_df['Forecast'] - current_price
                forecast_df['Price_Change_Pct'] = (forecast_df['Price_Change'] / current_price) * 100
            
            # Save to Excel
            excel_path = os.path.join(RESULTS_DIR, f"{ticker}_forecast_{days_ahead}d.xlsx")
            save_to_excel(forecast_df, excel_path, ticker=f"{ticker}_forecast")
            
            if verbose:
                print(f"Forecast results saved to: {excel_path}")
        
        # Additional analysis plot
        if plot_results and verbose:
            print("\nGenerating detailed forecast analysis...")
            
            # Create a comprehensive forecast plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Load recent historical data for context
            try:
                recent_df = load_ticker_data(ticker, years=1)
                recent_data = recent_df['Close'].tail(60)  # Last 60 days
                
                # Top plot: Historical + Forecast
                ax1.plot(recent_data.index, recent_data.values, 'b-', 
                        label='Historical', linewidth=2)
                ax1.plot(forecast_dates, forecast_values, 'r--', 
                        label='Forecast', linewidth=2, marker='o')
                ax1.fill_between(forecast_dates, 
                               conf_intervals.iloc[:, 0].values,
                               conf_intervals.iloc[:, 1].values,
                               alpha=0.3, color='red', label='95% Confidence Interval')
                
                ax1.set_title(f'{ticker} - Price Forecast ({days_ahead} days)')
                ax1.set_ylabel('Price ($)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Bottom plot: Forecast uncertainty
                uncertainty = conf_intervals.iloc[:, 1].values - conf_intervals.iloc[:, 0].values
                ax2.plot(forecast_dates, uncertainty, 'g-', linewidth=2, marker='s')
                ax2.set_title('Forecast Uncertainty (Confidence Interval Width)')
                ax2.set_ylabel('Uncertainty ($)')
                ax2.set_xlabel('Date')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save detailed plot
                detailed_plot_path = os.path.join(RESULTS_DIR, f"{ticker}_forecast_analysis.png")
                plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"Detailed forecast plot saved to: {detailed_plot_path}")
                plt.show()
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not create detailed analysis plot: {e}")
        
        if verbose:
            print(f"\nForecast generation completed successfully!")
        
        return forecast_values, conf_intervals, forecast_dates
        
    except FileNotFoundError as e:
        print(f"Error: No trained model found for {ticker}")
        print("Please train a model first using the train command")
        raise e
    except Exception as e:
        print(f"Error generating forecast for {ticker}: {e}")
        raise e


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Generate predictions using trained ARIMA model',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('ticker', nargs='?', default=DEFAULT_TICKER,
                       help=f'Stock ticker symbol (default: {DEFAULT_TICKER})')
    parser.add_argument('--days', type=int, default=FORECAST_HORIZON,
                       help=f'Number of days to forecast (default: {FORECAST_HORIZON})')
    parser.add_argument('--no-plot', action='store_false', dest='plot_results',
                       help='Do not create forecast plots')
    parser.add_argument('--no-save', action='store_false', dest='save_results',
                       help='Do not save forecast results')
    parser.add_argument('--quiet', action='store_false', dest='verbose',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Generate predictions
    predict_future(
        ticker=args.ticker,
        days_ahead=args.days,
        plot_results=args.plot_results,
        save_results=args.save_results,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main() 