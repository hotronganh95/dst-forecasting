import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import signal
from scipy.fft import fft, fftfreq
from read_dst_data import read_dst_csv, split_by_gaps

def detect_outliers(df, column='Dst', method='zscore', threshold=3):
    """
    Detect outliers in the time series data.
    
    Args:
        df (pandas.DataFrame): DataFrame with the data
        column (str): Column name to analyze
        method (str): Method to use for outlier detection ('zscore', 'iqr', 'modified_zscore')
        threshold (float): Threshold for outlier detection
        
    Returns:
        pandas.DataFrame: DataFrame with outlier information
    """
    data = df.copy()
    
    if method == 'zscore':
        # Z-score method
        z_scores = np.abs(stats.zscore(data[column]))
        data['is_outlier'] = z_scores > threshold
        data['outlier_score'] = z_scores
        
    elif method == 'iqr':
        # IQR method
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        data['is_outlier'] = (data[column] < lower_bound) | (data[column] > upper_bound)
        
        # Calculate how far outside the bounds the value is
        data['outlier_score'] = data[column].apply(lambda x: 
            max(0, (x - upper_bound) / IQR if x > upper_bound else (lower_bound - x) / IQR if x < lower_bound else 0)
        )
        
    elif method == 'modified_zscore':
        # Modified Z-score (more robust to outliers)
        median = data[column].median()
        mad = np.median(np.abs(data[column] - median))
        modified_z_scores = 0.6745 * np.abs(data[column] - median) / mad if mad > 0 else 0
        data['is_outlier'] = modified_z_scores > threshold
        data['outlier_score'] = modified_z_scores
    
    return data

def plot_outliers(df, column='Dst', outlier_col='is_outlier'):
    """
    Plot the time series with outliers highlighted.
    
    Args:
        df (pandas.DataFrame): DataFrame with outlier information
        column (str): Column name with the data
        outlier_col (str): Column indicating outliers
    """
    plt.figure(figsize=(14, 7))
    
    # Plot regular data points
    plt.plot(df['timestamp'], df[column], 'b-', alpha=0.6, label='Normal')
    
    # Plot outliers
    outliers = df[df[outlier_col]]
    plt.scatter(outliers['timestamp'], outliers[column], color='red', s=50, label='Outliers')
    
    plt.title(f'Dst Time Series with Outliers ({len(outliers)} outliers found)')
    plt.xlabel('Time')
    plt.ylabel('Dst Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print outlier statistics
    if not outliers.empty:
        print(f"\nNumber of outliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
        print(f"Outlier statistics:")
        print(f"  Min: {outliers[column].min()}")
        print(f"  Max: {outliers[column].max()}")
        print(f"  Mean: {outliers[column].mean():.2f}")
        print(f"  Median: {outliers[column].median():.2f}")

def plot_distribution(df, column='Dst'):
    """
    Plot the distribution of values with outlier analysis.
    
    Args:
        df (pandas.DataFrame): DataFrame with the data
        column (str): Column to analyze
    """
    plt.figure(figsize=(14, 10))
    
    # Create a 2x2 subplot
    plt.subplot(2, 2, 1)
    sns.histplot(df[column], kde=True)
    plt.title('Distribution of Dst Values')
    plt.xlabel('Dst Value')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    stats.probplot(df[column], plot=plt)
    plt.title('Q-Q Plot')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    sns.boxplot(y=df[column])
    plt.title('Box Plot of Dst Values')
    plt.ylabel('Dst Value')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    df[column].plot.density()
    plt.title('Kernel Density Estimate')
    plt.xlabel('Dst Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print distribution statistics
    print("\nDistribution statistics:")
    print(f"  Mean: {df[column].mean():.2f}")
    print(f"  Median: {df[column].median():.2f}")
    print(f"  Std Dev: {df[column].std():.2f}")
    print(f"  Skewness: {df[column].skew():.2f}")
    print(f"  Kurtosis: {df[column].kurtosis():.2f}")

def analyze_spectrum(df, column='Dst', sampling_freq=1/3600):
    """
    Perform spectral analysis on the time series data.
    
    Args:
        df (pandas.DataFrame): DataFrame with the data
        column (str): Column to analyze
        sampling_freq (float): Sampling frequency in Hz
    """
    # Get the data and remove any NaN values
    data = df[column].dropna().values
    
    # Compute FFT
    fft_vals = fft(data)
    n = len(data)
    freq = fftfreq(n, 1/sampling_freq)
    
    # Keep only the positive frequencies and normalize
    positive_freq_idx = np.arange(1, n//2)
    freqs = freq[positive_freq_idx]
    fft_power = 2.0/n * np.abs(fft_vals[positive_freq_idx])
    
    # Plot the power spectrum
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, fft_power)
    plt.title('FFT Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.xscale('log')
    
    # Identify dominant frequencies
    peak_indices = signal.find_peaks(fft_power, height=np.percentile(fft_power, 95))[0]
    
    if len(peak_indices) > 0:
        peak_freqs = freqs[peak_indices]
        peak_powers = fft_power[peak_indices]
        
        # Sort by power in descending order
        sorted_indices = np.argsort(peak_powers)[::-1]
        peak_freqs = peak_freqs[sorted_indices]
        peak_powers = peak_powers[sorted_indices]
        
        # Convert Hz to hours for interpretability
        peak_periods_hours = (1/peak_freqs) / 3600
        
        # Highlight top peaks
        top_n = min(5, len(peak_indices))
        for i in range(top_n):
            plt.axvline(x=peak_freqs[i], color='r', linestyle='--', alpha=0.7)
        
        # Print top frequencies
        print("\nDominant Frequencies:")
        for i in range(top_n):
            print(f"  Peak {i+1}: {peak_freqs[i]:.8f} Hz (Period: {peak_periods_hours[i]:.2f} hours)")
    
    # Spectrogram
    plt.subplot(2, 1, 2)
    
    # Apply windowing for better frequency resolution
    f, t, Sxx = signal.spectrogram(data, fs=sampling_freq, nperseg=1024, noverlap=512)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [seconds]')
    plt.colorbar(label='Power [dB]')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

def analyze_seasonality(df, column='Dst'):
    """
    Analyze potential seasonality patterns in the data
    
    Args:
        df (pandas.DataFrame): DataFrame with timestamp and data
        column (str): Column to analyze
    """
    # Create derived time features
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Create plots for different time periods
    plt.figure(figsize=(14, 18))
    
    # Hourly pattern
    plt.subplot(4, 1, 1)
    hourly_avg = df.groupby('hour')[column].mean()
    plt.plot(hourly_avg.index, hourly_avg.values, 'o-')
    plt.title('Average Dst by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Average Dst')
    plt.grid(True)
    plt.xticks(range(0, 24, 2))
    
    # Daily pattern
    plt.subplot(4, 1, 2)
    daily_avg = df.groupby('day')[column].mean()
    plt.plot(daily_avg.index, daily_avg.values, 'o-')
    plt.title('Average Dst by Day of Month')
    plt.xlabel('Day')
    plt.ylabel('Average Dst')
    plt.grid(True)
    plt.xticks(range(1, 32, 2))
    
    # Monthly pattern
    plt.subplot(4, 1, 3)
    monthly_avg = df.groupby('month')[column].mean()
    plt.plot(monthly_avg.index, monthly_avg.values, 'o-')
    plt.title('Average Dst by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Dst')
    plt.grid(True)
    plt.xticks(range(1, 13))
    
    # Day of week pattern
    plt.subplot(4, 1, 4)
    weekday_avg = df.groupby('dayofweek')[column].mean()
    plt.plot(weekday_avg.index, weekday_avg.values, 'o-')
    plt.title('Average Dst by Day of Week')
    plt.xlabel('Day of Week (0 = Monday)')
    plt.ylabel('Average Dst')
    plt.grid(True)
    plt.xticks(range(0, 7))
    
    plt.tight_layout()
    plt.show()

def perform_comprehensive_analysis(file_path, column='Dst', outlier_method='iqr', outlier_threshold=1.5):
    """
    Perform a comprehensive analysis of the time series data
    
    Args:
        file_path (str): Path to the CSV file
        column (str): Column to analyze
        outlier_method (str): Method for outlier detection
        outlier_threshold (float): Threshold for outlier detection
    """
    # Read data
    print("Reading data...")
    df = read_dst_csv(file_path)
    
    # Get continuous segments
    segments = split_by_gaps(df, gap_threshold='2H')
    longest_segment = max(segments, key=len)
    print(f"Using longest continuous segment with {len(longest_segment)} records")
    
    # Distribution analysis
    print("\n===== Distribution Analysis =====")
    plot_distribution(longest_segment, column)
    
    # Outlier analysis
    print("\n===== Outlier Detection =====")
    outlier_df = detect_outliers(longest_segment, column, method=outlier_method, threshold=outlier_threshold)
    plot_outliers(outlier_df, column)
    
    # Spectral analysis
    print("\n===== Spectral Analysis =====")
    analyze_spectrum(longest_segment, column)
    
    # Seasonality analysis
    print("\n===== Seasonality Analysis =====")
    analyze_seasonality(df, column)
    
    return outlier_df, segments

if __name__ == "__main__":
    file_path = "/mnt/e/dst-forecasting/dst_data.csv"
    
    # Perform comprehensive analysis
    outlier_df, segments = perform_comprehensive_analysis(file_path)
    
    # Export outliers if needed
    outliers = outlier_df[outlier_df['is_outlier']]
    if not outliers.empty:
        print(f"\nExporting {len(outliers)} outliers to CSV...")
        outliers.to_csv('/mnt/e/dst-forecasting/dst_outliers.csv', index=False)
        print("Export complete!")
