# RUN WITH: streamlit run demo/web_demo.py

#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from delphi.models.arima_model import ARIMAModel # Import your actual ARIMA model
from delphi.utils import evaluate_model # Import evaluate_model

# Streamlit App
st.set_page_config(layout="wide", page_title="Financial Forecast Demo")
st.title("Financial Forecast: Stock Price Forecasting Demo")

st.markdown("""
This is a demonstration of the financial forecast framework, showcasing its capabilities in stock price forecasting using a simple ARIMA model.

**Disclaimer:** This demo uses mock data or limited real-time data for illustrative purposes only and should not be used for actual trading decisions.
""")

# Sidebar for controls
st.sidebar.header("Configuration")

# Dataset selection
data_options = {
    "AAPL Daily Data": "aapl_daily.csv",
    "GOOG Daily Data": "goog_daily.csv",
    "MSFT Daily Data": "msft_daily.csv"
}
selected_data_name = st.sidebar.selectbox("Select Dataset", list(data_options.keys()))
selected_data_file = data_options[selected_data_name]

# Forecast Horizon selection
forecast_horizon = st.sidebar.slider("Select Forecast Horizon (days)", 1, 30, 7)

# Auto-ARIMA option
auto_arima = st.sidebar.checkbox("Auto-tune ARIMA order (slower)", value=False)

"""
### How to Use:

1.  **Select a Dataset:** Choose one of the pre-loaded mock stock datasets from the dropdown in the sidebar.
2.  **Choose Forecast Horizon:** Use the slider to select how many days into the future you want to forecast.
3.  **Run Forecast:** Click the "Run Forecast" button.
4.  **View Results:** The app will display:
    *   Historical evaluation metrics (RMSE, MAE, R², etc.) calculated on a hidden test set from the selected dataset.
    *   An interactive plot showing the historical stock prices and the future forecast with a 95% confidence interval.
    *   A table with the detailed forecasted prices and their confidence bounds.

Feel free to experiment with different datasets and forecast horizons!
"""

# Load data (placeholder for actual data loading)
@st.cache_data
def load_data(file_path):
    # Read the CSV, skipping the initial metadata rows and explicitly naming columns
    df = pd.read_csv(
        f"demo/{file_path}",
        skiprows=3,  # Skip the first 3 lines (Price, Ticker, Date header lines)
        names=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], # Explicitly name columns
        parse_dates=['Date'] # Parse the 'Date' column as dates
    )
    df.set_index('Date', inplace=True) # Set 'Date' as the DataFrame index
    return df['Close'] # We only need the Close price for ARIMA

data = load_data(selected_data_file)

if st.sidebar.button("Run Forecast"):
    if data.empty:
        st.warning("Please select a dataset to run the forecast.")
    else:
        st.subheader(f"Forecast for {selected_data_name}")
        
        # Split data for historical evaluation (backtesting)
        train_size = int(len(data) * 0.8)
        train_data_for_eval = data.iloc[:train_size]
        test_data_for_eval = data.iloc[train_size:]

        # Main Model Training for Future Forecast
        with st.spinner("Training ARIMA model for future forecast..."):
            try:
                # Use auto_arima if checked, otherwise use fixed order
                if auto_arima:
                    st.info("Auto-tuning ARIMA order...")
                    model = ARIMAModel()
                    model.fit(data, auto_order=True) # Train on full data for future forecast
                    st.success(f"Auto-tuned ARIMA order: {model.order}")
                else:
                    model = ARIMAModel(order=(5,1,0)) # Using a default order, can be made configurable
                    model.fit(data, auto_order=False) # Train on full data for future forecast
                
                # Display the model's order to the user
                st.write(f"**ARIMA Model Order:** {model.order}")
                
                # Display model summary in an expander
                with st.expander("View ARIMA Model Summary"):
                    st.text(model.get_model_summary())
                
            except Exception as e:
                st.error(f"Error during model training: {e}")
                st.warning("Model could not be trained. Please check data or adjust parameters.")
                st.stop() # Stop execution if model training fails
        
        st.markdown("--- ") # Separator

        # Historical Evaluation (Backtesting)
        col1, col2 = st.columns(2) # Create two columns for metrics and info
        
        with col1:
            with st.spinner("Evaluating model on historical test data..."):
                try:
                    # Make predictions on the test set using the *trained* model
                    # For simplicity, we re-fit on a subset of data for evaluation, or predict directly
                    # Given ARIMA's nature, a simple re-fit for the evaluation segment is often done
                    
                    # Retrain a separate model on the train_data_for_eval to simulate backtesting
                    eval_model = ARIMAModel(order=model.order) # Use the same order as the main model
                    eval_model.fit(train_data_for_eval, auto_order=False) # No auto-order here, use fixed

                    start_index = len(train_data_for_eval)
                    end_index = len(data) - 1 # Corresponds to the end of test_data_for_eval
                    
                    predictions_on_test = eval_model.fitted_model.get_prediction(
                        start=start_index, end=end_index
                    ).predicted_mean
                    
                    # Ensure predictions_on_test has the same index as test_data_for_eval
                    predictions_on_test.index = test_data_for_eval.index
                    
                    metrics = evaluate_model(test_data_for_eval.values, predictions_on_test.values)
                    
                    st.success("Historical Evaluation Complete!")
                    st.write("### Historical Evaluation Metrics (on Test Data)")
                    metrics_df = pd.DataFrame([metrics]).T.rename(columns={0: "Value"})
                    
                    with st.expander("View Historical Evaluation Metrics"):
                        st.dataframe(metrics_df)
                    
                except Exception as e:
                    st.error(f"Error during historical evaluation: {e}")
                    st.warning("Could not compute evaluation metrics.")
                    
        with col2:
            st.info("The historical evaluation metrics above are calculated by training an ARIMA model on 80% of the selected data and testing its predictive accuracy on the remaining 20%. This simulates how the model would have performed on unseen historical data.")
            st.info("These metrics provide insight into how well the model has performed on past data, using the same ARIMA order as determined for the main forecast.")
            if auto_arima:
                st.info(f"Note: When 'Auto-tune ARIMA order' is enabled, the model automatically selects an optimal order (e.g., {model.order}) which may result in relatively flat forecasts if the data does not exhibit strong, predictable trends or seasonality.")
        
        st.markdown("--- ") # Separator
        
        # Future Forecast Generation
        with st.spinner("Generating future forecast..."):
            try:
                forecast_values, conf_int = model.predict(steps=forecast_horizon, return_conf_int=True)
                
                # Debugging: Print raw forecast and confidence intervals
                st.write("### Raw Forecast Values (for debugging):")
                st.write(forecast_values)
                st.write("### Raw Confidence Intervals (for debugging):")
                st.write(conf_int)
                # End Debugging
                
                # Generate future dates for the forecast
                last_date = data.index[-1]
                forecast_dates = pd.date_range(start=last_date + timedelta(days=1),
                                               periods=forecast_horizon, freq='D')
                # Ensure forecast_series uses the actual values of forecast_values
                forecast_series = pd.Series(forecast_values.values, index=forecast_dates)
                
                st.success("Future Forecast Generated!")
                st.write("### Historical Data and Forecast")
                
                # Create Plotly figure
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines',
                    name='Historical Close',
                    line=dict(color='blue')
                ))
                
                # Add forecast data
                fig.add_trace(go.Scatter(
                    x=forecast_series.index,
                    y=forecast_series.values,
                    mode='lines',
                    name=f'Forecast ({forecast_horizon} days)',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_series.index.tolist() + forecast_series.index.tolist()[::-1],
                    y=conf_int.iloc[:, 1].tolist() + conf_int.iloc[:, 0].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ))
                
                fig.update_layout(
                    title=f'{selected_data_name} Stock Price: Historical vs Forecast',
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # st.write("### Forecast Details")
                forecast_df = forecast_series.to_frame(name='Predicted Close')
                forecast_df['Lower Bound (95% CI)'] = conf_int.iloc[:, 0].values
                forecast_df['Upper Bound (95% CI)'] = conf_int.iloc[:, 1].values
                
                with st.expander("View Forecast Details"):
                    st.dataframe(forecast_df)
                
            except Exception as e:
                st.error(f"Error during forecasting: {e}")
                st.warning("Consider adjusting the forecast horizon or selecting a different dataset.")