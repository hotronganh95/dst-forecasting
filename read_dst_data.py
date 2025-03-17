import pandas as pd
import datetime

def read_dst_csv(file_path):
    """
    Read Dst data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame with parsed timestamp and Dst values
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert timestamp string to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure Dst values are numeric
    df['Dst'] = pd.to_numeric(df['Dst'])
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values(by='timestamp')
    
    return df

def identify_time_gaps(df, expected_interval='1H'):
    """
    Identify gaps in time series data.
    
    Args:
        df (pandas.DataFrame): DataFrame with timestamp column
        expected_interval (str): Expected time interval between records
        
    Returns:
        list: List of tuples with (gap_start, gap_end, gap_duration)
    """
    # Calculate time differences
    df = df.copy()
    df['time_diff'] = df['timestamp'].diff()
    
    # Convert the expected interval to timedelta
    expected_td = pd.Timedelta(expected_interval)
    
    # Find gaps (where diff is greater than expected interval)
    gaps = df[df['time_diff'] > expected_td].copy()
    
    if gaps.empty:
        return []
    
    # Prepare gap information
    gap_info = []
    for idx, row in gaps.iterrows():
        prev_timestamp = df.loc[idx-1, 'timestamp'] if idx > 0 else None
        gap_start = prev_timestamp
        gap_end = row['timestamp']
        gap_duration = row['time_diff']
        gap_info.append((gap_start, gap_end, gap_duration))
    
    return gap_info

def split_by_gaps(df, gap_threshold='1H'):
    """
    Split DataFrame into segments separated by time gaps.
    
    Args:
        df (pandas.DataFrame): DataFrame with timestamp column
        gap_threshold (str): Threshold for considering a gap
        
    Returns:
        list: List of DataFrames, each representing a continuous segment
    """
    # Calculate time differences
    df = df.copy()
    df['time_diff'] = df['timestamp'].diff()
    
    # Convert the threshold to timedelta
    threshold_td = pd.Timedelta(gap_threshold)
    
    # Find indices where gaps occur
    gap_indices = df[df['time_diff'] > threshold_td].index.tolist()
    
    # Create segments
    segments = []
    start_idx = 0
    
    for idx in gap_indices:
        segment = df.iloc[start_idx:idx].copy()
        segments.append(segment)
        start_idx = idx
    
    # Add the last segment
    if start_idx < len(df):
        segments.append(df.iloc[start_idx:].copy())
    
    # Remove the time_diff column from each segment
    for i in range(len(segments)):
        segments[i] = segments[i].drop(columns=['time_diff'])
    
    return segments

if __name__ == "__main__":
    # Example usage
    file_path = "/mnt/e/dst-forecasting/dst_data.csv"
    try:
        dst_data = read_dst_csv(file_path)
        print(f"Successfully loaded {len(dst_data)} rows of Dst data")
        print(dst_data.head())
        
        # Basic statistics
        print("\nBasic statistics:")
        print(f"Date range: {dst_data['timestamp'].min()} to {dst_data['timestamp'].max()}")
        print(f"Dst range: {dst_data['Dst'].min()} to {dst_data['Dst'].max()}")
        print(f"Average Dst: {dst_data['Dst'].mean():.2f}")
        
        # Identify time gaps
        gaps = identify_time_gaps(dst_data)
        if gaps:
            print("\nFound time gaps:")
            for gap_start, gap_end, gap_duration in gaps:
                print(f"Gap from {gap_start} to {gap_end} (duration: {gap_duration})")
        else:
            print("\nNo time gaps found. Data is continuous with 1-hour intervals.")
        
        # Split data by gaps (for gaps > 2 hours)
        segments = split_by_gaps(dst_data, '2H')
        print(f"\nData split into {len(segments)} continuous segments")
        for i, segment in enumerate(segments):
            print(f"Segment {i+1}: {len(segment)} rows from {segment['timestamp'].min()} to {segment['timestamp'].max()}")
        
    except Exception as e:
        print(f"Error reading the file: {e}")
