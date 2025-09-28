#!/usr/bin/env python
"""
Quick demonstration of Financial Forecast Demo capabilities
Shows the complete workflow: train -> evaluate -> predict
"""
import sys
import os

# Add parent directory to path to import delphi package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delphi.scripts.train import train_ticker_model
from delphi.scripts.evaluate import evaluate_ticker_model
from delphi.scripts.predict import predict_future

ticker = 'GOOG'

def run_complete_demo(ticker, years=3):
    """
    Run a complete demonstration of the Financial Forecast Demo capabilities
    
    Args:
        ticker: Stock ticker to analyze
        years: Years of historical data to use
    """
    print("=" * 60)
    print("FINANCIAL FORECAST DEMO - COMPLETE WORKFLOW DEMONSTRATION")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Historical Data: {years} years")
    print()
    
    try:
        # Step 1: Train Model
        print("STEP 1: TRAINING ARIMA MODEL")
        print("-" * 40)
        model = train_ticker_model(
            ticker=ticker,
            years=years,
            auto_order=True,
            save_model=True,
            plot_results=False  # Disable plots for cleaner demo
        )
        print("Training completed successfully!")
        print()
        
        # Step 2: Evaluate Model
        print("STEP 2: EVALUATING MODEL WITH TRADING SIMULATION")
        print("-" * 40)
        print("This now matches the main evaluation methodology:")
        print("   • Generates buy/sell signals from ARIMA predictions")
        print("   • Backtests trading strategy vs buy-and-hold")
        print("   • Calculates Sharpe ratios and drawdown metrics")
        print()
        
        metrics = evaluate_ticker_model(
            ticker=ticker,
            years=years,
            plot_results=False,  # Disable plots for cleaner demo
            save_results=True,
            include_simulation=True  # Enable trading simulation
        )
        print("Evaluation with trading simulation completed successfully!")
        print()
        
        # Display key simulation metrics
        print("Key Trading Simulation Results:")
        print("-" * 35)
        simulation_keys = ['Total_Return', 'Total_Return_BH', 'Alpha', 'Sharpe', 'Sharpe_BH', 'Max_Drawdown_%', 'Trading_Signals']
        for key in simulation_keys:
            if key in metrics:
                value = metrics[key]
                if 'Trading_Signals' in key:
                    print(f"  {key}: {value}")
                elif '%' in key or 'Return' in key or 'Alpha' in key:
                    print(f"  {key}: {value:.2f}%")
                else:
                    print(f"  {key}: {value:.4f}")
        print()
        
        # Step 3: Generate Predictions
        print("STEP 3: GENERATING FUTURE PREDICTIONS")
        print("-" * 40)
        forecast, conf_int, dates = predict_future(
            ticker=ticker,
            days_ahead=7,
            plot_results=False,  # Disable plots for cleaner demo
            save_results=True,
            verbose=True
        )
        print("Predictions generated successfully!")
        print()
        
        # Summary
        print("DEMO SUMMARY")
        print("=" * 40)
        print(f"✓ Trained ARIMA model for {ticker}")
        print(f"✓ Model order: {model.order}")
        print(f"✓ Training AIC: {model.fitted_model.aic:.2f}")
        print(f"✓ Evaluation RMSE: {metrics['RMSE']:.4f}")
        print(f"✓ Direction Accuracy: {metrics['Direction Accuracy (%)']:.2f}%")
        
        # Add simulation summary
        if 'Total_Return' in metrics:
            print(f"✓ Trading Strategy Return: {metrics['Total_Return']:.2f}%")
            print(f"✓ Buy-and-Hold Return: {metrics['Total_Return_BH']:.2f}%")
            print(f"✓ Alpha (Strategy - B&H): {metrics['Alpha']:.2f}%")
            print(f"✓ Trading Signals Generated: {metrics['Trading_Signals']}")
        
        print(f"✓ Generated 7-day forecast")
        print(f"✓ Results saved to delphi/results/")
        print()
        print("Demo completed successfully!")
        print("Now includes full trading simulation analysis")
        print("Check the delphi/results/ directory for detailed outputs.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        raise e


def run_quick_test():
    """Run a quick test to verify the system is working"""
    print("Running quick system test...")
    
    try:
        # Test data loading
        from delphi.data.loader import validate_ticker, get_latest_price
        
        print(f"Testing ticker validation for {ticker}...")
        is_valid = validate_ticker(ticker)
        
        if is_valid:
            print(f"Ticker {ticker} is valid")
            price = get_latest_price(ticker)
            print(f"Current {ticker} price: ${price:.2f}")
        else:
            print(f"Ticker {ticker} validation failed")
            return False
        
        print("Quick test passed!")
        return True
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False


if __name__ == '__main__':
    print("Welcome to Financial Forecast Demo!")
    print()
    
    # Run quick test first
    if run_quick_test():
        print()
        
        # Ask user if they want to run the full demo
        response = input("Run complete demo? This will train a model and may take a few minutes (y/n): ")
        
        if response.lower().startswith('y'):
            print()
            run_complete_demo(ticker)
        else:
            print("Demo cancelled. You can run individual commands using:")
            print("python -m delphi.scripts.run train SPY")
            print("python -m delphi.scripts.run evaluate SPY")
            print("python -m delphi.scripts.run predict SPY")
    else:
        print("System test failed. Please check your installation.") 