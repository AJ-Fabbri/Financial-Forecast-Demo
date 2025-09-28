# Financial Time Series Prediction Demo

A demonstration of sophisticated financial modeling infrastructure using ARIMA models. This project showcases engineering capabilities, system architecture, and statistical modeling expertise while maintaining clean, production-ready code.

**Note**: The internal project codename is "Delphi" (hence the file names), but this is simply referred to as "the project" or "the demo" throughout this documentation.

## **IMPORTANT DISCLAIMERS**

- **NOT FINANCIAL ADVICE**: This is a demonstration project only. Nothing here constitutes financial advice.
- **SIMPLIFIED VERSION**: This is a heavily simplified version of a proprietary project. It does not use deep learning or cutting-edge statistical methods.
- **EDUCATIONAL PURPOSE**: Statistical signals here are relatively meaningless and should NOT be used for actual trading decisions.
- **NO WARRANTIES**: Use at your own risk. The authors are not responsible for any financial losses.

## Overview

This demo is a modular framework for financial time series prediction that demonstrates:

- **Professional Software Architecture**: Clean, maintainable code with proper separation of concerns
- **Statistical Modeling Expertise**: ARIMA implementation with automatic order selection and diagnostic tools
- **Production-Ready Infrastructure**: CLI tools, configuration management, error handling, and comprehensive logging
- **Data Engineering**: Real-time data loading from Yahoo Finance with technical indicator calculation
- **Visualization & Reporting**: Professional charts, evaluation metrics, and Excel export capabilities
- **Interactive Web Demo**: A Streamlit application for real-time forecasting demonstrations.

## Architecture

```
financial-demo/ # Repository structure
├── README.md               # High-level overview of the framework
├── demo/                   # Demo examples, including the Streamlit web demo
│   ├── web_demo.py         # Streamlit web application
│   ├── data_download.py   # Script to generate demo data (real yfinance data)
│   ├── aapl_daily.csv      # Sample daily data for AAPL
│   ├── goog_daily.csv      # Sample daily data for GOOG
│   └── msft_daily.csv      # Sample daily data for MSFT
├── delphi/                 # Core framework package (internal codename)
│   ├── config.py           # Configuration management
│   ├── utils.py            # Utility functions
│   ├── data/
│   │   ├── loader.py       # yfinance integration & data processing
│   └── models/
│       └── arima_model.py  # ARIMA model implementation
│   └── scripts/            # CLI tools
│       ├── run.py          # Main CLI entry point
│       ├── train.py        # Training script
│       ├── evaluate.py     # Evaluation script
│       └── predict.py      # Prediction script
├── requirements.txt        # Python dependencies
└── setup.py                # Package setup
```

## Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd financial-demo
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download Demo Data** (Optional):
   The Streamlit app uses pre-generated CSVs for demonstration. To update the sample data:
```bash
python3 demo/data_download.py
```

### Running the Web Demo (RECOMMENDED APPROACH)

Run the run_web_demo.py script, OR:
Navigate to the project root and run the Streamlit application:
```bash
streamlit run demo/web_demo.py
```
This will open the interactive web demo in your browser, allowing you to select datasets, forecast horizons, and view real-time predictions with evaluation metrics.

### Basic CLI Usage (for advanced users/developers)

The framework also provides CLI tools for data management, model training, evaluation, and prediction.

**Example: Train an ARIMA model for SPY data (CLI)**:
```bash
python3 -m delphi.scripts.run train SPY --years 5
```

**Example: Evaluate the trained SPY model (CLI)**:
```bash
python3 -m delphi.scripts.run evaluate SPY
```

**Example: Generate 10-day forecast for SPY (CLI)**:
```bash
python3 -m delphi.scripts.run predict SPY --days 10
```

For more CLI commands, run `python3 -m delphi.scripts.run --help`.

## Features

### Data Management
- **DuckDB database integration** for efficient data storage
- **Real-time data loading** from Yahoo Finance with database caching
- **Technical indicators**: MACD, RSI, Bollinger Bands, ATR, Moving Averages
- **Configurable features**: Easy selection of input variables
- **Data validation**: Automatic handling of missing data and outliers
- **Database management tools** for populating and maintaining data

### Statistical Modeling
- **ARIMA implementation** with automatic order selection
- **Stationarity testing** using Augmented Dickey-Fuller test
- **Model diagnostics**: Residual analysis, ACF/PACF plots
- **Confidence intervals** for predictions

**Important Note**: ARIMA is univariate (uses only Close price), but we calculate multivariate features to demonstrate data engineering capabilities for future models.

### Engineering Excellence
- **Modular design**: Clean separation between data, models, and interfaces
- **Configuration management**: Centralized settings with sensible defaults
- **Error handling**: Comprehensive exception handling and user feedback
- **Logging**: Detailed progress tracking and debugging information
- **CLI interface**: Professional command-line tools with help documentation

### Visualization & Reporting
- **Interactive Web Dashboard**: Powered by Streamlit for live forecasting visualization and evaluation.
- **Professional charts**: High-quality Plotly visualizations within the web app, and Matplotlib for CLI.
- **Diagnostic plots**: Model residuals, ACF/PACF analysis (CLI).
- **Forecast visualization**: Historical context with confidence intervals.
- **Excel export**: Structured results for further analysis (CLI).

## Configuration

The system is highly configurable through the configuration file:

```python
# Data parameters
DEFAULT_HISTORY_YEARS = 5
DEFAULT_TICKER = 'SPY'

# Model parameters
ARIMA_ORDER = (1, 1, 1)
FORECAST_HORIZON = 5
CONFIDENCE_LEVEL = 0.95

# Feature selection
INPUT_FEATURES = [
    'Close', 'Volume', 'High', 'Low', 'Open'
]
```

## Example Workflow

### 1. Train a Model
```bash
python -m delphi.scripts.run train AAPL --years 3
```
Output:
- Model diagnostics plots
- Training summary with AIC/BIC scores
- Saved model for future use

### 2. Evaluate Performance
```bash
python -m delphi.scripts.run evaluate AAPL
```
Output:
- Prediction accuracy metrics (RMSE, MAE, R², MAPE)
- Actual vs predicted visualizations
- Residual analysis plots

### 3. Generate Forecasts
```bash
python -m delphi.scripts.run predict AAPL --days 7
```
Output:
- 7-day price forecasts with confidence intervals
- Trend analysis and summary statistics
- Professional forecast visualizations

### 4. Database Management
```bash
# Populate database with demo data
python3 -m delphi.scripts.manage_database populate-demo

# Update specific ticker
python3 -m delphi.scripts.manage_database update SPY --years 5

# List tickers in database
python3 -m delphi.scripts.manage_database list

# Show database statistics
python3 -m delphi.scripts.manage_database stats
```

## Advanced Usage

### Custom Features
```python
from delphi.data.loader import load_ticker_data

# Load data with custom features
data = load_ticker_data('SPY', years=5, features=['Close', 'Volume', 'MACD'])
```

### Programmatic API
```python
from delphi.models.arima_model import train_arima_model, forecast_future
from delphi.data.loader import load_ticker_data, prepare_data_for_arima

# Load and prepare data
df = load_ticker_data('SPY', years=3)
train_data, test_data, _, _ = prepare_data_for_arima(df)

# Train model
model = train_arima_model(train_data, ticker='SPY', auto_order=True)

# Generate forecast
forecast, conf_int, dates = forecast_future(model, steps=10)
```

## System Requirements

- **Python**: 3.8+
- **Dependencies**: See `requirements.txt`
- **Memory**: ~500MB for typical usage
- **Storage**: ~100MB for models and results

## Testing & Validation

The framework includes comprehensive validation:
- **Stationarity testing** for time series appropriateness
- **Model diagnostics** to verify assumptions
- **Cross-validation** for robust performance estimation
- **Statistical significance** testing for predictions

## Technical Highlights

### Statistical Rigor
- Proper ARIMA methodology with order selection
- Stationarity testing and data preprocessing
- Confidence interval calculation
- Model diagnostic validation

### Software Engineering
- **Object-oriented design** with clear interfaces
- **Error handling** with informative messages
- **Configuration management** for easy customization
- **Modular architecture** for extensibility

### Data Engineering
- **Real-time data integration** with yfinance
- **Technical indicator calculation** with proper handling of edge cases
- **Data validation** and cleaning pipelines
- **Efficient data structures** using pandas

## Use Cases

This framework demonstrates capabilities for:
- **Quantitative Research**: Statistical modeling and backtesting
- **Risk Management**: Volatility forecasting and scenario analysis
- **Portfolio Management**: Asset price prediction and allocation
- **Trading Systems**: Signal generation and strategy development

## Documentation

- **Code Documentation**: Comprehensive docstrings throughout
- **Configuration Guide**: Detailed parameter explanations
- **API Reference**: Complete function and class documentation
- **Examples**: Practical usage scenarios
- **Architecture Diagrams**: System diagrams in `architecture/`
- **Methodology Documentation**: Detailed explanations of models and approaches in `docs/`

## Future Enhancements

The modular architecture allows for easy extension:
- **Additional Models**: GARCH, VAR, Machine Learning models
- **Data Sources**: Bloomberg, Quandl, custom APIs
- **Advanced Features**: Portfolio optimization, risk metrics
- **Trading Simulation**: Integration of more complex trading strategies
- **Scalability**: Distributed computing with Dask/Spark

---

**Note**: This is a demonstration framework showcasing software engineering and statistical modeling capabilities. It uses simplified ARIMA models and sample data for educational and demonstration purposes. It is **not** intended for production use or real financial trading decisions.

**DISCLAIMER**: This is NOT financial advice. This is a heavily simplified educational demo of a proprietary project. The statistical signals are relatively meaningless and should not be used for actual trading decisions.

## License

MIT License - See LICENSE file for details. 