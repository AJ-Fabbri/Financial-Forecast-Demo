#!/usr/bin/env python
"""
Main entry point for Financial Forecast Demo package
Adapted from the original project with ARIMA modeling focus
"""
import argparse
import sys

from .train import train_ticker_model
from .evaluate import evaluate_ticker_model
from .predict import predict_future
from ..config import DEFAULT_TICKER, FORECAST_HORIZON


def main():
    """Main entry point for the Financial Forecast Demo package"""
    parser = argparse.ArgumentParser(
        description='Financial Forecast Demo - Financial Time Series Prediction with ARIMA Models',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  %(prog)s train SPY --years 5
  %(prog)s evaluate SPY
  %(prog)s predict SPY --days 10
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train an ARIMA model for a ticker')
    train_parser.add_argument('ticker', nargs='?', default=DEFAULT_TICKER,
                             help=f'Stock ticker symbol (default: {DEFAULT_TICKER})')
    train_parser.add_argument('--years', type=int, default=5,
                             help='Number of years of historical data (default: 5)')
    train_parser.add_argument('--target', type=str, default='Close',
                             help='Target column for prediction (default: Close)')
    train_parser.add_argument('--no-auto-order', action='store_false', dest='auto_order',
                             help='Disable automatic ARIMA order selection')
    train_parser.add_argument('--no-save', action='store_false', dest='save_model',
                             help='Do not save the trained model')
    train_parser.add_argument('--no-plot', action='store_false', dest='plot_results',
                             help='Do not create diagnostic plots')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained ARIMA model')
    eval_parser.add_argument('ticker', nargs='?', default=DEFAULT_TICKER,
                            help=f'Stock ticker symbol (default: {DEFAULT_TICKER})')
    eval_parser.add_argument('--years', type=int, default=5,
                            help='Number of years of historical data (default: 5)')
    eval_parser.add_argument('--target', type=str, default='Close',
                            help='Target column for prediction (default: Close)')
    eval_parser.add_argument('--no-plot', action='store_false', dest='plot_results',
                            help='Do not create evaluation plots')
    eval_parser.add_argument('--no-save', action='store_false', dest='save_results',
                            help='Do not save evaluation results')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions with a trained model')
    predict_parser.add_argument('ticker', nargs='?', default=DEFAULT_TICKER,
                               help=f'Stock ticker symbol (default: {DEFAULT_TICKER})')
    predict_parser.add_argument('--days', type=int, default=FORECAST_HORIZON,
                               help=f'Number of days ahead to predict (default: {FORECAST_HORIZON})')
    predict_parser.add_argument('--no-plot', action='store_false', dest='plot_results',
                               help='Do not create forecast plots')
    predict_parser.add_argument('--no-save', action='store_false', dest='save_results',
                               help='Do not save forecast results')
    predict_parser.add_argument('--quiet', action='store_false', dest='verbose',
                               help='Reduce output verbosity')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run the appropriate command
    try:
        if args.command == 'train':
            train_ticker_model(
                ticker=args.ticker,
                years=args.years,
                target_col=args.target,
                auto_order=args.auto_order,
                save_model=args.save_model,
                plot_results=args.plot_results
            )
        elif args.command == 'evaluate':
            evaluate_ticker_model(
                ticker=args.ticker,
                years=args.years,
                target_col=args.target,
                plot_results=args.plot_results,
                save_results=args.save_results
            )
        elif args.command == 'predict':
            predict_future(
                ticker=args.ticker,
                days_ahead=args.days,
                plot_results=args.plot_results,
                save_results=args.save_results,
                verbose=args.verbose
            )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 