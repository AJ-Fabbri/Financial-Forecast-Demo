"""
ARIMA model implementation for Financial Forecast Demo
Demonstrates time series modeling capabilities using statistical methods
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from ..config import ARIMA_ORDER, FORECAST_HORIZON, CONFIDENCE_LEVEL
from ..utils import evaluate_model, save_arima_model, plot_predictions, plot_forecast


class ARIMAModel:
    """
    ARIMA model wrapper for financial time series prediction
    """
    
    def __init__(self, order=None, seasonal_order=None):
        """
        Initialize ARIMA model
        
        Args:
            order: ARIMA order (p, d, q) tuple
            seasonal_order: Seasonal ARIMA order
        """
        self.order = order or ARIMA_ORDER
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
    def check_stationarity(self, data, verbose=True):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            data: Time series data
            verbose: Whether to print results
            
        Returns:
            Boolean indicating if series is stationary
        """
        result = adfuller(data.dropna())
        
        if verbose:
            print('Augmented Dickey-Fuller Test Results:')
            print(f'ADF Statistic: {result[0]:.6f}')
            print(f'p-value: {result[1]:.6f}')
            print(f'Critical Values:')
            for key, value in result[4].items():
                print(f'\t{key}: {value:.3f}')
        
        # If p-value < 0.05, reject null hypothesis (series is stationary)
        is_stationary = result[1] < 0.05
        
        if verbose:
            status = "stationary" if is_stationary else "non-stationary"
            print(f'Result: The series is {status}')
            
        return is_stationary
    
    def auto_find_order(self, data, max_p=5, max_d=2, max_q=5, verbose=True):
        """
        Automatically find optimal ARIMA order using AIC criterion
        
        Args:
            data: Time series data
            max_p: Maximum AR order to test
            max_d: Maximum differencing order to test
            max_q: Maximum MA order to test
            verbose: Whether to print progress
            
        Returns:
            Optimal (p, d, q) order
        """
        best_aic = np.inf
        best_order = None
        
        if verbose:
            print("Searching for optimal ARIMA order...")
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        aic = fitted.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            
                    except Exception as e:
                        continue
        
        if verbose:
            print(f"Optimal order: {best_order} with AIC: {best_aic:.2f}")
            
        return best_order
    
    def fit(self, data, auto_order=False, verbose=True):
        """
        Fit ARIMA model to data
        
        Args:
            data: Time series data
            auto_order: Whether to automatically find optimal order
            verbose: Whether to print fitting progress
            
        Returns:
            Fitted model
        """
        if auto_order:
            self.order = self.auto_find_order(data, verbose=verbose)
        
        if verbose:
            print(f"Fitting ARIMA{self.order} model...")
        
        try:
            self.model = ARIMA(data, order=self.order, 
                             seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            if verbose:
                print("Model fitted successfully!")
                print(f"AIC: {self.fitted_model.aic:.2f}")
                print(f"BIC: {self.fitted_model.bic:.2f}")
                
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            raise e
            
        return self.fitted_model
    
    def predict(self, steps=None, return_conf_int=True):
        """
        Make predictions using fitted model
        
        Args:
            steps: Number of steps to forecast (default from config)
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Predictions and optionally confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if steps is None:
            steps = FORECAST_HORIZON
        
        # Get forecast
        forecast_result = self.fitted_model.forecast(steps=steps, 
                                                   alpha=1-CONFIDENCE_LEVEL)
        
        predictions = forecast_result
        
        if return_conf_int:
            # Get prediction intervals
            pred_ci = self.fitted_model.get_prediction(
                start=len(self.fitted_model.fittedvalues), 
                end=len(self.fitted_model.fittedvalues) + steps - 1
            ).conf_int(alpha=1-CONFIDENCE_LEVEL)
            
            return predictions, pred_ci
        
        return predictions
    
    def get_model_summary(self):
        """
        Get model summary statistics
        
        Returns:
            Model summary string
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return str(self.fitted_model.summary())
    
    def plot_diagnostics(self, figsize=(15, 10)):
        """
        Plot model diagnostic plots
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Residuals plot
        residuals = self.fitted_model.resid
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].grid(True)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, density=True)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].grid(True)
        
        # ACF of residuals
        plot_acf(residuals, ax=axes[1, 0], lags=20)
        axes[1, 0].set_title('ACF of Residuals')
        
        # PACF of residuals
        plot_pacf(residuals, ax=axes[1, 1], lags=20)
        axes[1, 1].set_title('PACF of Residuals')
        
        plt.tight_layout()
        return fig


def train_arima_model(data, ticker=None, order=None, auto_order=True, 
                     save_model=True, verbose=True):
    """
    Train an ARIMA model on time series data
    
    Args:
        data: Time series data
        ticker: Stock ticker (for saving)
        order: ARIMA order (p, d, q)
        auto_order: Whether to automatically find optimal order
        save_model: Whether to save the trained model
        verbose: Whether to print progress
        
    Returns:
        Trained ARIMAModel instance
    """
    if verbose:
        print(f"Training ARIMA model for {ticker or 'data'}...")
        print(f"Data shape: {data.shape}")
    
    # Initialize model
    arima_model = ARIMAModel(order=order)
    
    # Check stationarity
    if verbose:
        arima_model.check_stationarity(data)
    
    # Fit model
    fitted_model = arima_model.fit(data, auto_order=auto_order, verbose=verbose)
    
    # Save model if requested
    if save_model and ticker:
        model_params = {
            'order': arima_model.order,
            'seasonal_order': arima_model.seasonal_order,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
        save_arima_model(arima_model, ticker, model_params)
    
    return arima_model


def evaluate_arima_model(model, test_data, test_dates=None, plot_results=True, 
                        ticker=None, verbose=True):
    """
    Evaluate ARIMA model performance on test data
    
    Args:
        model: Trained ARIMAModel instance
        test_data: Test time series data
        test_dates: Optional test dates for plotting
        plot_results: Whether to create plots
        ticker: Stock ticker for plot titles
        verbose: Whether to print results
        
    Returns:
        Dictionary of evaluation metrics
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before evaluation")
    
    # Make predictions for test period
    n_test = len(test_data)
    predictions = []
    
    # For ARIMA, we need to make step-by-step predictions
    # to properly evaluate on test data
    for i in range(n_test):
        if i == 0:
            # First prediction uses the fitted model
            forecast_result = model.fitted_model.forecast(steps=1)
            pred = forecast_result.iloc[0] if hasattr(forecast_result, 'iloc') else forecast_result.values[0]
        else:
            # Subsequent predictions: refit with additional data point
            extended_data = pd.concat([
                model.fitted_model.model.data.orig_endog,
                test_data.iloc[:i]
            ])
            temp_model = ARIMA(extended_data, order=model.order)
            temp_fitted = temp_model.fit()
            forecast_result = temp_fitted.forecast(steps=1)
            pred = forecast_result.iloc[0] if hasattr(forecast_result, 'iloc') else forecast_result.values[0]
        
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    metrics = evaluate_model(test_data.values, predictions)
    
    if verbose:
        print("\nModel Evaluation Results:")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
    
    # Create plots if requested
    if plot_results:
        title = f"ARIMA Model Evaluation - {ticker}" if ticker else "ARIMA Model Evaluation"
        
        # Plot predictions vs actual
        plot_predictions(
            test_data.values, 
            predictions, 
            title=title,
            dates=test_dates
        )
        
        plt.tight_layout()
        plt.show()
    
    return metrics


def evaluate_arima_5day_ahead(model, full_data, test_size=0.2, forecast_horizon=5, 
                             ticker=None, verbose=True, plot_results=True):
    """
    Evaluate ARIMA model's 5-day ahead prediction accuracy
    Similar to the main Project_Delphi evaluation methodology
    
    Args:
        model: Trained ARIMAModel instance
        full_data: Complete time series data (pandas Series with datetime index)
        test_size: Proportion of data to use for testing
        forecast_horizon: Number of days ahead to forecast (default 5)
        ticker: Stock ticker for plot titles
        verbose: Whether to print results
        plot_results: Whether to create plots
        
    Returns:
        Dictionary of evaluation metrics and results DataFrame
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before evaluation")
    
    # Calculate test period
    n_total = len(full_data)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test
    
    # We need additional data points to make 5-day ahead predictions
    # So we start evaluation from train_end - forecast_horizon
    evaluation_start = n_train - forecast_horizon
    evaluation_end = n_total - forecast_horizon
    
    predictions_5day = []
    actual_values_5day = []
    evaluation_dates = []
    prediction_dates = []
    
    if verbose:
        print(f"Evaluating 5-day ahead predictions...")
        print(f"Evaluation period: {evaluation_start} to {evaluation_end} (total: {evaluation_end - evaluation_start} predictions)")
    
    # For each point in the evaluation period, make a 5-day ahead prediction
    for i in range(evaluation_start, evaluation_end):
        # Get data up to point i for training the temporary model
        train_data_temp = full_data.iloc[:i+1]
        
        # Create and fit temporary ARIMA model
        try:
            temp_model = ARIMA(train_data_temp.values, order=model.order)
            temp_fitted = temp_model.fit()
            
            # Make 5-day ahead forecast
            forecast_result = temp_fitted.forecast(steps=forecast_horizon)
            if hasattr(forecast_result, 'iloc'):
                pred_5day = forecast_result.iloc[-1]  # Get the 5th day prediction
            else:
                pred_5day = forecast_result[-1]
            
            # Get actual value 5 days later
            actual_idx = i + forecast_horizon
            if actual_idx < len(full_data):
                actual_5day = full_data.iloc[actual_idx]
                
                predictions_5day.append(pred_5day)
                actual_values_5day.append(actual_5day)
                evaluation_dates.append(full_data.index[i])
                prediction_dates.append(full_data.index[actual_idx])
                
        except Exception as e:
            if verbose:
                print(f"Skipping prediction at index {i}: {e}")
            continue
    
    if len(predictions_5day) == 0:
        raise ValueError("No valid 5-day ahead predictions could be made")
    
    predictions_5day = np.array(predictions_5day)
    actual_values_5day = np.array(actual_values_5day)
    
    # Calculate 5-day ahead metrics
    metrics_5day = evaluate_model(actual_values_5day, predictions_5day)
    
    # Calculate direction accuracy (did we predict the right direction?)
    actual_changes = actual_values_5day - np.array([full_data.iloc[full_data.index.get_loc(date)] 
                                                   for date in evaluation_dates])
    pred_changes = predictions_5day - np.array([full_data.iloc[full_data.index.get_loc(date)] 
                                               for date in evaluation_dates])
    
    direction_correct = (np.sign(actual_changes) == np.sign(pred_changes)).sum()
    direction_accuracy = (direction_correct / len(actual_changes)) * 100
    metrics_5day['Direction_Accuracy_5D'] = direction_accuracy
    
    if verbose:
        print("\n5-Day Ahead Prediction Evaluation Results:")
        print("=" * 50)
        for metric, value in metrics_5day.items():
            if 'Accuracy' in metric or '%' in metric:
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value:.6f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Evaluation_Date': evaluation_dates,
        'Prediction_Date': prediction_dates,
        'Actual_5D': actual_values_5day,
        'Predicted_5D': predictions_5day,
        'Error_5D': actual_values_5day - predictions_5day,
        'Percent_Error_5D': ((actual_values_5day - predictions_5day) / actual_values_5day) * 100
    })
    
    # Create plots if requested
    if plot_results:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Predictions vs Actual (5-day ahead)
        axes[0].plot(prediction_dates, actual_values_5day, 'b-', label='Actual Values', linewidth=2)
        axes[0].plot(prediction_dates, predictions_5day, 'r--', label='5-Day Ahead Predictions', linewidth=2)
        axes[0].set_title(f'5-Day Ahead Predictions vs Actual - {ticker or "ARIMA Model"}')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction errors over time
        axes[1].plot(prediction_dates, results_df['Percent_Error_5D'], 'g-', linewidth=1)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1].set_title('5-Day Ahead Prediction Errors (%)')
        axes[1].set_ylabel('Percent Error (%)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return metrics_5day, results_df


def forecast_future(model, steps=None, plot_results=True, ticker=None):
    """
    Generate future forecasts using trained ARIMA model
    
    Args:
        model: Trained ARIMAModel instance
        steps: Number of steps to forecast
        plot_results: Whether to create forecast plot
        ticker: Stock ticker for plot titles
        
    Returns:
        Tuple of (forecast_values, confidence_intervals, forecast_dates)
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before forecasting")
    
    if steps is None:
        steps = FORECAST_HORIZON
    
    # Get forecast with confidence intervals
    forecast, conf_int = model.predict(steps=steps, return_conf_int=True)
    
    # Generate future dates (assuming daily frequency)
    # Try to get the last date from model data, fallback to using the original data index
    try:
        if model.fitted_model.model.data.dates is not None:
            last_date = model.fitted_model.model.data.dates[-1]
        else:
            # Fallback: get from original endog data index if it's a pandas Series/DataFrame
            orig_data = model.fitted_model.model.data.orig_endog
            if hasattr(orig_data, 'index'):
                last_date = orig_data.index[-1]
            else:
                # Final fallback: use today's date
                last_date = pd.Timestamp.now().normalize()
    except (AttributeError, IndexError, TypeError):
        # Fallback: use today's date
        last_date = pd.Timestamp.now().normalize()
    
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                  periods=steps, freq='D')
    
    if plot_results:
        # Get historical data for context
        orig_data = model.fitted_model.model.data.orig_endog
        try:
            if model.fitted_model.model.data.dates is not None:
                historical_data = pd.Series(orig_data, index=model.fitted_model.model.data.dates)
            else:
                # If dates are None, use the original data as-is if it already has an index
                if hasattr(orig_data, 'index'):
                    historical_data = orig_data
                else:
                    # Create a simple integer index
                    historical_data = pd.Series(orig_data)
        except (AttributeError, TypeError):
            # Fallback: create series with integer index
            historical_data = pd.Series(orig_data)
        
        title = f"ARIMA Forecast - {ticker}" if ticker else "ARIMA Forecast"
        confidence_intervals = (conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values)
        
        plot_forecast(
            historical_data, 
            forecast.values, 
            forecast_dates,
            confidence_intervals=confidence_intervals,
            title=title
        )
        
        plt.tight_layout()
        plt.show()
    
    return forecast.values, conf_int, forecast_dates 


def evaluate_arima_with_simulation(model, test_data, test_dates=None, plot_results=True, 
                                 ticker=None, verbose=True, forecast_horizon=5):
    """
    Evaluate ARIMA model with trading simulation (matching main Project_Delphi methodology)
    
    Args:
        model: Trained ARIMAModel instance
        test_data: Test time series data
        test_dates: Test dates for evaluation
        plot_results: Whether to create plots
        ticker: Stock ticker for plot titles
        verbose: Whether to print results
        forecast_horizon: Number of days ahead for predictions
        
    Returns:
        Dictionary of evaluation metrics including trading simulation results
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from ..config import VOLATILITY_MULTIPLIER, INITIAL_CASH, BUY_THRESHOLD, SELL_THRESHOLD, MIN_HOLDING_DAYS
    from ..utils import evaluate_model
    
    if not model.is_fitted:
        raise ValueError("Model must be fitted before evaluation")
    
    # Make step-by-step predictions for the test period
    n_test = len(test_data)
    predictions = []
    
    for i in range(n_test):
        if i == 0:
            # First prediction uses the fitted model
            forecast_result = model.fitted_model.forecast(steps=1)
            pred = forecast_result.iloc[0] if hasattr(forecast_result, 'iloc') else forecast_result.values[0]
        else:
            # Subsequent predictions: refit with additional data point
            extended_data = pd.concat([
                model.fitted_model.model.data.orig_endog,
                test_data.iloc[:i]
            ])
            from statsmodels.tsa.arima.model import ARIMA
            temp_model = ARIMA(extended_data, order=model.order)
            temp_fitted = temp_model.fit()
            forecast_result = temp_fitted.forecast(steps=1)
            pred = forecast_result.iloc[0] if hasattr(forecast_result, 'iloc') else forecast_result.values[0]
        
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate basic metrics for 1-day ahead
    basic_metrics_1d = evaluate_model(test_data.values, predictions)
    
    # Generate 5-day ahead predictions for dual-horizon evaluation
    predictions_5d = []
    actual_values_5d = []
    
    # Generate 5-day ahead predictions using the same approach as evaluate_arima_5day_ahead
    for i in range(min(n_test - 5, len(test_data) - 5)):
        if i == 0:
            # First 5-day prediction uses the fitted model
            forecast_result = model.fitted_model.forecast(steps=5)
            pred_5d = forecast_result.iloc[4] if hasattr(forecast_result, 'iloc') else forecast_result.values[4]
        else:
            # Subsequent predictions: refit with additional data point
            extended_data = pd.concat([
                model.fitted_model.model.data.orig_endog,
                test_data.iloc[:i]
            ])
            temp_model = ARIMA(extended_data, order=model.order)
            temp_fitted = temp_model.fit()
            forecast_result = temp_fitted.forecast(steps=5)
            pred_5d = forecast_result.iloc[4] if hasattr(forecast_result, 'iloc') else forecast_result.values[4]
        
        predictions_5d.append(pred_5d)
        actual_values_5d.append(test_data.values[i + 4])  # Actual value 5 days ahead
    
    predictions_5d = np.array(predictions_5d)
    actual_values_5d = np.array(actual_values_5d)
    
    # Calculate basic metrics for 5-day ahead
    basic_metrics_5d = evaluate_model(actual_values_5d, predictions_5d)
    
    # Combine dual-horizon metrics
    from ..utils import evaluate_model_dual_horizon
    basic_metrics = evaluate_model_dual_horizon(
        test_data.values, predictions,  # 1-day ahead
        actual_values_5d, predictions_5d  # 5-day ahead
    )
    
    # Generate trading signals using ARIMA predictions
    # For ARIMA, we'll simulate multi-step ahead predictions for signal generation
    signals = []
    vol_mult = VOLATILITY_MULTIPLIER
    
    # Create a DataFrame with price data and dates
    if test_dates is None:
        test_dates = pd.date_range(start='2024-01-01', periods=len(test_data), freq='D')
    
    # Calculate rolling volatility (30-day window)
    price_series = pd.Series(test_data.values, index=test_dates)
    returns = price_series.pct_change()
    rolling_vol = returns.rolling(window=min(30, len(returns))).std().fillna(returns.std()) * 100
    
    # Generate signals by comparing predictions at different time points
    for i in range(forecast_horizon, len(predictions)):
        date_now = test_dates[i]
        
        # Simulate the signal generation logic:
        # Compare current prediction with a prediction made forecast_horizon days ago
        if i >= forecast_horizon:
            price_past = predictions[i - forecast_horizon]  # Prediction from forecast_horizon days ago
            price_future = predictions[i]  # Current prediction
            
            pct_change = (price_future - price_past) / price_past * 100
            vol_pct = rolling_vol.iloc[i]
            
            buy_thr = vol_mult * vol_pct
            sell_thr = -buy_thr
            
            if pct_change >= buy_thr:
                sig = 'Buy'
            elif pct_change <= sell_thr:
                sig = 'Sell'
            else:
                continue  # No actionable signal
            
            signals.append({
                'Date': date_now,
                'Signal': sig,
                'Predicted_Price_Past': price_past,
                'Predicted_Price_Future': price_future,
                'Predicted_Pct_Change': pct_change,
                'Volatility': vol_pct
            })
    
    # Create signals DataFrame
    signals_df = pd.DataFrame(signals)
    
    if signals_df.empty:
        if verbose:
            print("No trading signals generated")
        return {**basic_metrics, 'trading_signals': 0}
    
    # Merge signals with price data
    price_df = pd.DataFrame({
        'Date': test_dates,
        'Close': test_data.values,
        'Volatility': rolling_vol.values
    })
    
    df_signals = price_df.merge(signals_df[['Date', 'Signal']], on='Date', how='left')
    
    # Backtest trading strategy
    initial_cash = INITIAL_CASH
    cash = initial_cash
    holdings = 0.0
    last_buy_price = None
    last_sell_price = None
    last_trade_date = None
    buy_threshold = BUY_THRESHOLD
    sell_threshold = SELL_THRESHOLD
    min_holding_days = MIN_HOLDING_DAYS
    positions = []
    
    for _, row in df_signals.iterrows():
        date, price, signal = row['Date'], row['Close'], row['Signal']
        
        # Check minimum holding period
        if last_trade_date is not None and (date - last_trade_date).days <= min_holding_days:
            positions.append({'Date': date, 'Cash': cash, 'Holdings': holdings, 'Price': price, 'Signal': signal})
            continue
        
        # Buy logic
        if signal == 'Buy' and holdings == 0:
            if last_sell_price is None or price < last_sell_price * buy_threshold:
                holdings = cash / price
                cash = 0.0
                last_buy_price = price
                last_trade_date = date
                if verbose:
                    print(f"{date.date()}: BUY  → {holdings:.4f} @ ${price:.2f}")
        
        # Sell logic
        elif signal == 'Sell' and holdings > 0 and last_buy_price is not None:
            if price > last_buy_price * sell_threshold:
                cash = holdings * price
                if verbose:
                    print(f"{date.date()}: SELL → {holdings:.4f} @ ${price:.2f}, Cash=${cash:.2f}")
                holdings = 0.0
                last_sell_price = price
                last_trade_date = date
        
        positions.append({'Date': date, 'Cash': cash, 'Holdings': holdings, 'Price': price, 'Signal': signal})
    
    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame(positions)
    portfolio_df['Portfolio_Value'] = portfolio_df['Cash'] + portfolio_df['Holdings'] * portfolio_df['Price']
    
    # Calculate total return
    total_return = ((portfolio_df['Portfolio_Value'].iloc[-1] - initial_cash) / initial_cash) * 100
    
    # Buy-and-hold strategy comparison
    first_buy = df_signals[df_signals['Signal'] == 'Buy']
    if not first_buy.empty:
        first_buy_date = first_buy['Date'].iloc[0]
        first_buy_price = first_buy['Close'].iloc[0]
        bh_slice = df_signals[df_signals['Date'] >= first_buy_date].copy()
        bh_slice['Buy_and_Hold'] = (initial_cash / first_buy_price) * bh_slice['Close']
        total_return_bh = ((bh_slice['Buy_and_Hold'].iloc[-1] - initial_cash) / initial_cash) * 100
        
        # Calculate Sharpe ratios
        portfolio_df['Return'] = portfolio_df['Portfolio_Value'].pct_change()
        bh_slice['Return'] = bh_slice['Buy_and_Hold'].pct_change()
        
        sharpe_trading = portfolio_df['Return'].mean() / portfolio_df['Return'].std() if portfolio_df['Return'].std() > 0 else 0
        sharpe_bh = bh_slice['Return'].mean() / bh_slice['Return'].std() if bh_slice['Return'].std() > 0 else 0
        sharpe_trading_annual = sharpe_trading * np.sqrt(252)
        sharpe_bh_annual = sharpe_bh * np.sqrt(252)
        
        # Calculate drawdowns
        portfolio_df['Trading_CumMax'] = portfolio_df['Portfolio_Value'].cummax()
        portfolio_df['Trading_Drawdown'] = portfolio_df['Portfolio_Value'] / portfolio_df['Trading_CumMax'] - 1
        mdd_trading = portfolio_df['Trading_Drawdown'].min()
        
        bh_slice['BH_CumMax'] = bh_slice['Buy_and_Hold'].cummax()
        bh_slice['BH_Drawdown'] = bh_slice['Buy_and_Hold'] / bh_slice['BH_CumMax'] - 1
        mdd_bh = bh_slice['BH_Drawdown'].min()
        
        # Trading simulation metrics
        simulation_metrics = {
            'Total_Return': total_return,
            'Total_Return_BH': total_return_bh,
            'Alpha': total_return - total_return_bh,
            'Sharpe': sharpe_trading_annual,
            'Sharpe_BH': sharpe_bh_annual,
            'Sharpe_Diff': sharpe_trading_annual - sharpe_bh_annual,
            'Max_Drawdown_%': mdd_trading * 100,
            'Max_Drawdown_BH_%': mdd_bh * 100,
            'Max_Drawdown_Diff': (mdd_trading - mdd_bh) * 100,
            'Trading_Signals': len(signals_df)
        }
        
        # Create strategy vs buy-and-hold plot
        if plot_results:
            # Merge portfolio and buy-and-hold results
            combined_df = portfolio_df[['Date', 'Portfolio_Value']].merge(
                bh_slice[['Date', 'Buy_and_Hold']], on='Date', how='inner'
            )
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Top plot: Strategy vs Buy-and-Hold
            ax1.plot(combined_df['Date'], combined_df['Portfolio_Value'], 
                    label='ARIMA Strategy', linewidth=2, color='blue')
            ax1.plot(combined_df['Date'], combined_df['Buy_and_Hold'], 
                    label=f"Buy-&-Hold (from {first_buy_date.date()})", linewidth=2, color='orange')
            
            # Add buy and sell signals
            buy_signals = portfolio_df[portfolio_df['Signal'] == 'Buy']
            sell_signals = portfolio_df[portfolio_df['Signal'] == 'Sell']
            
            if not buy_signals.empty:
                buy_signals_with_bh = buy_signals.merge(combined_df[['Date', 'Buy_and_Hold']], on='Date', how='left')
                ax1.scatter(buy_signals_with_bh['Date'], buy_signals_with_bh['Buy_and_Hold'], 
                           marker='^', color='green', s=100, label='Buy Signal', zorder=5)
            
            if not sell_signals.empty:
                sell_signals_with_bh = sell_signals.merge(combined_df[['Date', 'Buy_and_Hold']], on='Date', how='left')
                ax1.scatter(sell_signals_with_bh['Date'], sell_signals_with_bh['Buy_and_Hold'], 
                           marker='v', color='red', s=100, label='Sell Signal', zorder=5)
            
            ax1.set_title(f'{ticker} - ARIMA Strategy vs Buy-and-Hold')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Predictions vs Actual
            ax2.plot(test_dates, test_data.values, 'b-', label='Actual', linewidth=2)
            ax2.plot(test_dates, predictions, 'r--', label='ARIMA Predictions', linewidth=2)
            ax2.set_title(f'{ticker} - ARIMA Predictions vs Actual Prices')
            ax2.set_ylabel('Price ($)')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    else:
        # No buy signals generated
        simulation_metrics = {
            'Total_Return': 0,
            'Total_Return_BH': 0,
            'Alpha': 0,
            'Sharpe': 0,
            'Sharpe_BH': 0,
            'Sharpe_Diff': 0,
            'Max_Drawdown_%': 0,
            'Max_Drawdown_BH_%': 0,
            'Max_Drawdown_Diff': 0,
            'Trading_Signals': len(signals_df)
        }
    
    # Combine all metrics with dual-horizon structure
    all_metrics = {}
    
    # Add dual-horizon prediction metrics with prefixed keys
    for horizon, horizon_metrics in basic_metrics.items():
        for metric_name, metric_value in horizon_metrics.items():
            all_metrics[f'{horizon}_{metric_name}'] = metric_value
    
    # Add simulation metrics
    all_metrics.update(simulation_metrics)
    
    # For backward compatibility, also include the 1-day metrics without prefix
    if '1d' in basic_metrics:
        all_metrics.update(basic_metrics['1d'])
    
    if verbose:
        print("\nARIMA Model with Trading Simulation Results:")
        print("=" * 50)
        print("Dual-Horizon Prediction Metrics:")
        for horizon, horizon_metrics in basic_metrics.items():
            print(f"\n{horizon} ahead metrics:")
            for metric, value in horizon_metrics.items():
                if 'Accuracy' in metric or '%' in metric:
                    print(f"  {metric}: {value:.2f}%")
                else:
                    print(f"  {metric}: {value:.6f}")
        
        print("\nTrading Simulation Metrics:")
        for metric, value in simulation_metrics.items():
            if 'Trading_Signals' in metric:
                print(f"  {metric}: {value}")
            elif '%' in metric or 'Return' in metric or 'Alpha' in metric:
                print(f"  {metric}: {value:.2f}%")
            else:
                print(f"  {metric}: {value:.4f}")
    
    return all_metrics 