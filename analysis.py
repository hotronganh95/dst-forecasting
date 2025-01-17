import datetime
from datetime import datetime
import traceback
import httpx
import pytz
import asyncio
from fiber.logging_utils import get_logger
from geomag import parse_data, clean_data
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft, fftfreq
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fetch import fetch_yearly_data
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = get_logger(__name__)

async def fetch_data(years = range(2005, 2026), out_path="all_dst_data.csv"):
    raw_data = await fetch_yearly_data(years)
    parsed_df = parse_data(raw_data)
    del raw_data
    cleaned_df = clean_data(parsed_df)
    del parsed_df
    cleaned_df.to_csv(out_path, index=False)

def cal_stats(df: pd.DataFrame):
    result = adfuller(df['Dst'].dropna())
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])

    if result[1] < 0.05:
        print("The time series is stationary (p-value < 0.05).")
    else:
        print("The time series is not stationary (p-value >= 0.05).")
    return result

def AFC_plot(df: pd.DataFrame, lags=50):
    # Plot ACF
    plot_acf(df['Dst'], lags=lags)  # Adjust lags based on your data
    plt.show()

def PAFC_plot(df: pd.DataFrame, lags=50):
    # Plot ACF
    plot_pacf(df['Dst'], lags=lags)  # Adjust lags based on your data
    plt.show()

def FT(df: pd.DataFrame):
    # Perform Fourier Transform
    N = len(df['Dst'])  # Number of data points
    T = 24.0  # Sampling interval (e.g., hourly data -> 1/24)

    # Compute FFT
    yf = fft(df['Dst'])
    xf = fftfreq(N, T)[:N//2]  # Positive frequencies only

    # Plot the frequency spectrum
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.xlabel('Frequency (1/hour)')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform - Frequency Spectrum')
    plt.show()

def visualize(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Dst'], label='Dst Values')
    plt.xlabel('Timestamp')
    plt.ylabel('Dst Value')
    plt.title('Hourly Equatorial Dst Values')
    plt.legend()
    plt.show()

def Autocorrelation(df: pd.DataFrame):
    plot_acf(df['Dst'].dropna(), lags=50)
    plt.show()

    plot_pacf(df['Dst'].dropna(), lags=50)
    plt.show()

def distribution(df: pd.DataFrame):
    df['Dst'].hist(bins=100, figsize=(10, 6))
    plt.xlabel('Dst Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Dst Values')
    plt.show()

def check_trend(df: pd.DataFrame):
    result = seasonal_decompose(df['Dst'], model='additive', period=24*30*12)  # Adjust 'period' if needed
    result.plot()
    plt.show()

def ARIMA_forecast(df: pd.DataFrame, order=(3, 0, 3)):
    model = ARIMA(df['Dst'], order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    forecast = model_fit.forecast(steps=24)
    print(forecast)

def SARIMA_forecast(df: pd.DataFrame, seasonal_order = (1, 0, 1, 24), order=(1, 0, 1)):
    model = SARIMAX(df['Dst'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=True, maxiter=1000, tol=1e-7)
    print(model_fit.summary())
    forecast = model_fit.forecast(steps=24)
    print(forecast)

def automatic_lag_suggestion(df: pd.DataFrame):
    acf_vals = acf(df['Dst'], nlags=100)  # Test 100 lags
    pacf_vals = pacf(df['Dst'], nlags=100)

    print("ACF Significant Lags:", [i for i, val in enumerate(acf_vals) if abs(val) > 0.2])
    print("PACF Significant Lags:", [i for i, val in enumerate(pacf_vals) if abs(val) > 0.2])

def robust_scale_dst(dst_data, split_ratio=0.8):
    """
    Robustly scale Dst data and prepare it for time series modeling.
    
    Parameters:
    dst_data (array-like): Array of Dst index values
    split_ratio (float): Ratio for train/test split
    
    Returns:
    dict: Dictionary containing scaled data and scaler object
    """
    # Reshape data for sklearn
    dst_array = np.array(dst_data).reshape(-1, 1)
    
    # Calculate split point
    split_point = int(len(dst_array) * split_ratio)
    
    # Split into train and test sets
    train_data = dst_array[:split_point]
    test_data = dst_array[split_point:]
    
    # Initialize and fit RobustScaler on training data only
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaled_train = scaler.fit_transform(train_data)
    
    # Transform test data using the same scaler
    scaled_test = scaler.transform(test_data)
    
    # Calculate scaling parameters
    median = scaler.center_
    iqr = scaler.scale_
    
    return {
        'scaled_train': scaled_train,
        'scaled_test': scaled_test,
        'scaler': scaler,
        'train_data': train_data,
        'test_data': test_data,
        'median': median[0],
        'iqr': iqr[0]
    }

def visualize_scaling(original_data, scaled_data, title_prefix=""):
    """
    Visualize original vs scaled data distributions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Original data distribution
    ax1.hist(original_data, bins=30, color='blue', alpha=0.7)
    ax1.axvline(np.median(original_data), color='red', linestyle='dashed', 
                label=f'Median: {np.median(original_data):.2f}')
    ax1.set_title(f"{title_prefix} Original Dst Distribution")
    ax1.set_xlabel("Dst (nT)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    
    # Scaled data distribution
    ax2.hist(scaled_data, bins=30, color='green', alpha=0.7)
    ax2.axvline(np.median(scaled_data), color='red', linestyle='dashed',
                label=f'Median: {np.median(scaled_data):.2f}')
    ax2.set_title(f"{title_prefix} Scaled Dst Distribution")
    ax2.set_xlabel("Scaled Dst")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Example usage
def demonstrate_robust_scaling():
    np.random.seed(42)
    normal_conditions = np.random.normal(-25, 15, 1000)  # Normal conditions
    storms = np.random.normal(-150, 50, 200)  # Storm conditions
    severe_storms = np.random.normal(-350, 100, 50)  # Severe storms
    dst_data = np.concatenate([normal_conditions, storms, severe_storms])
    
    # Apply robust scaling
    scaling_results = robust_scale_dst(dst_data)
    fig = visualize_scaling(
        scaling_results['train_data'], 
        scaling_results['scaled_train'],
        "Training Data"
    )
    return scaling_results, fig

def scale(df: pd.DataFrame):
    # Assuming your data is in a DataFrame called df with a column 'Dst'
    scaler = StandardScaler()
    df['Dst'] = scaler.fit_transform(df[['Dst']])
    return df

def log_transform(df: pd.DataFrame):
    df['Dst'] = df['Dst'] + 400
    df['Dst'] = np.log1p(df['Dst'])
    return df

def robust_scale(df: pd.DataFrame):
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    df['Dst'] = scaler.fit_transform(df[['Dst']])
    return df

def diff_transform(df: pd.DataFrame):
    df['Dst'] = df['Dst'].diff().dropna()
    return df

def ana(data_path):
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df[:-24*30]
    print(df)
    # df = log_transform(df)
    # cal_stats(df)
    # visualize(df)
    # check_trend(df)
    # distribution(df)
    # FT(df)
    AFC_plot(df, lags=100)
    # automatic_lag_suggestion(df)
    # print(df)
    # SARIMA_forecast(df)
    # ARIMA_forecast(df)

if(__name__ == '__main__'):
    # Fetch newest data
    dataset = 'dst_data.csv'
    years = range(2005, 2026)
    asyncio.run(fetch_data(years=years, out_path=dataset))

    # Do some analysis
    ana(dataset)