"""
Data loading utilities for Financial Forecast Demo
Uses DuckDB database as primary source with yfinance fallback
Simplified version of the original data loading capabilities
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..config import DEFAULT_HISTORY_YEARS, INPUT_FEATURES
from .database import get_database, update_ticker_from_yfinance


def load_ticker_data(ticker, years=None, features=None, use_database=True):
    """
    Load stock data with preference for database, fallback to yfinance
    
    Args:
        ticker: Stock ticker symbol (e.g., 'SPY')
        years: Number of years of historical data (default from config)
        features: List of features to include (default from config)
        use_database: Whether to try database first (default: True)
        
    Returns:
        DataFrame with stock data and indicators
    """
    if years is None:
        years = DEFAULT_HISTORY_YEARS
    
    if features is None:
        features = INPUT_FEATURES
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    df = None
    
    # Try database first if enabled
    if use_database:
        try:
            print(f"Trying to load {ticker} from database...")
            db = get_database()
            df = db.get_ticker_data(
                ticker, 
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            db.close()
            
            if not df.empty:
                print(f"Loaded {len(df)} rows from database")
            else:
                print(f"No data in database for {ticker}")
                
        except Exception as e:
            print(f"Database load failed: {e}")
    
    # Fallback to yfinance if database didn't work
    if df is None or df.empty:
        print(f"Loading {ticker} data from yfinance...")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            print(f"Loaded {len(df)} rows from yfinance")
            
            # Optionally save to database for next time
            if use_database:
                try:
                    print("Saving to database for future use...")
                    db = get_database()
                    db.upsert_ticker_data(ticker, df)
                    db.close()
                except Exception as e:
                    print(f"Warning: Could not save to database: {e}")
                    
        except Exception as e:
            raise ValueError(f"Failed to load data for {ticker}: {e}")
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Select only requested features that exist
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features {missing_features}. Using available: {available_features}")
    
    # Return only available features
    result_df = df[available_features].copy()
    
    # Remove any rows with NaN values
    result_df = result_df.dropna()
    
    print(f"Final dataset: {len(result_df)} rows with {len(available_features)} features")
    return result_df


def add_technical_indicators(df):
    """
    Add basic technical indicators to the DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    # Ensure we have the basic columns
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        raise ValueError("DataFrame must contain OHLCV columns")
    
    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    
    # MACD (simplified)
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI (simplified)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
    bb_std_dev = df['Close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
    
    # Price returns
    df['Return'] = df['Close'].pct_change()
    df['Return_1d'] = df['Close'].pct_change(1)
    
    # Volatility (rolling standard deviation of returns)
    df['Volatility'] = df['Return'].rolling(window=20).std()
    
    # Average True Range (ATR) - simplified
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df


def prepare_data_for_arima(df, target_column='Close', test_size=0.2):
    """
    Prepare data specifically for ARIMA modeling
    
    Args:
        df: DataFrame with stock data
        target_column: Column to use as target variable
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (train_data, test_data, train_dates, test_dates)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Extract target series
    target_series = df[target_column].copy()
    
    # Split into train/test
    split_idx = int(len(target_series) * (1 - test_size))
    
    train_data = target_series.iloc[:split_idx]
    test_data = target_series.iloc[split_idx:]
    
    train_dates = target_series.index[:split_idx]
    test_dates = target_series.index[split_idx:]
    
    print(f"Data split: {len(train_data)} training samples, {len(test_data)} test samples")
    
    return train_data, test_data, train_dates, test_dates


def get_latest_price(ticker):
    """
    Get the most recent price for a ticker
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Latest closing price
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d')
    
    if data.empty:
        raise ValueError(f"No recent data found for {ticker}")
    
    return data['Close'].iloc[-1]


def validate_ticker(ticker):
    """
    Validate that a ticker exists and has data
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Boolean indicating if ticker is valid
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='5d')
        return not data.empty
    except:
        return False


def populate_database_with_tickers(tickers, years=5):
    """
    Populate database with multiple tickers
    
    Args:
        tickers: List of ticker symbols
        years: Years of historical data to fetch
        
    Returns:
        Dictionary with results for each ticker
    """
    results = {}
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        try:
            success = update_ticker_from_yfinance(ticker, years)
            results[ticker] = {'success': success}
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            results[ticker] = {'success': False, 'error': str(e)}
    
    return results 