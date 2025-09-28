"""
Configuration settings for Financial Forecast Demo
Simplified version focusing on ARIMA modeling capabilities
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "delphi", "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "delphi", "models")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "delphi", "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "delphi", "results")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data parameters
DEFAULT_HISTORY_YEARS = 5  # Shorter for demo
DEFAULT_TICKER = 'SPY'     # Default to SPY for demo

# Feature Selection Configuration
# Simplified feature set for demonstration
INPUT_FEATURES = [
    'Close',           # Target column (always included)
    'Volume',          # Volume data
    'High',            # High price
    'Low',             # Low price
    'Open',            # Open price
    'MACD',            # MACD indicator
    'MACD_Signal',     # MACD signal line
    'MACD_Hist',       # MACD histogram
    'RSI',             # RSI indicator
    'SMA_10',          # 10-day moving average
    'SMA_20',          # 20-day moving average
    'ATR',             # Average True Range
    'Return',          # Price returns
    'Volatility',      # Rolling volatility
]

# ARIMA model parameters (replacing LSTM parameters)
ARIMA_ORDER = (1, 1, 1)    # (p, d, q) - simple ARIMA configuration
FORECAST_HORIZON = 5       # Days ahead to predict
SEASONAL_ORDER = None      # No seasonal component for simplicity

# Training parameters
TEST_SIZE = 0.2            # Larger test set for demo
VALIDATION_SIZE = 0      # No validation set as ARIMA is not a deep learning model
RANDOM_SEED = 42

# Evaluation parameters
CONFIDENCE_LEVEL = 0.95    # For prediction intervals

# Demo mode configuration - always True for demo
DEMO_MODE = True 

# Trading simulation parameters
VOLATILITY_MULTIPLIER = 1.5  # Multiplier for volatility-based signal thresholds
INITIAL_CASH = 1000.0  # Starting cash for trading simulation
BUY_THRESHOLD = 1.0  # Minimum price ratio for buying after selling
SELL_THRESHOLD = 1.0  # Minimum profit ratio for selling after buying
MIN_HOLDING_DAYS = 0  # Minimum days to hold a position 