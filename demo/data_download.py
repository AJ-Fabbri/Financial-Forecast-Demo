
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf # Import yfinance

def download_and_save_stock_data(ticker, start_date, end_date, output_path):
    """
    Downloads historical stock data using yfinance and saves it to a CSV file.
    """
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    if not df.empty:
        df.to_csv(output_path)
        print(f"Saved {ticker} data to {output_path}")
    else:
        print(f"No data downloaded for {ticker}. Check ticker symbol or date range.")

if __name__ == "__main__":
    # Define date range for demo data (e.g., last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2 * 365) # Approx 2 years of data

    # Generate data for AAPL
    download_and_save_stock_data("AAPL", start_date, end_date, 'demo/aapl_daily.csv')

    # Generate data for GOOG
    download_and_save_stock_data("GOOG", start_date, end_date, 'demo/goog_daily.csv')

    # Generate data for MSFT
    download_and_save_stock_data("MSFT", start_date, end_date, 'demo/msft_daily.csv')