import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import serialization
import matplotlib.pyplot as plt
import asyncio
from datetime import datetime, timedelta, timezone
import argparse

# Import custom modules
from read_dst_data import read_dst_csv, split_by_gaps
from enhanced_lstm_model import add_cyclical_features, prepare_enhanced_data
from transformer_model import load_trained_model
from geomag import fetch_yearly_data, parse_data, clean_data

class DstForecaster:
    """
    Class for forecasting Dst values using trained models
    """
    def __init__(self, model_paths, data_path=None):
        """
        Initialize DstForecaster with model paths and data path
        
        Args:
            model_paths (dict): Dictionary mapping model types to their paths
            data_path: Path to historical data CSV (optional)
        """
        self.model_paths = model_paths
        self.data_path = data_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize models and scalers as None
        self.models = {}
        self.scaler = None
        self.feature_cols = None
        
        # Track if models are loaded
        self.loaded_models = set()
        
        print(f"DstForecaster initialized. Using device: {self.device}")
    
    def load_models(self):
        """Load all specified models"""
        print("Loading models...")
        
        for model_type, model_path in self.model_paths.items():
            try:
                if os.path.exists(model_path):
                    # Use the load_trained_model function from transformer_model.py
                    model = load_trained_model(
                        model_path=model_path,
                        model_type=model_type,
                        device=self.device
                    )
                    
                    if model is not None:
                        self.models[model_type] = model
                        self.loaded_models.add(model_type)
                        print(f"{model_type.upper()} model loaded from {model_path}")
                    else:
                        print(f"Failed to load {model_type.upper()} model from {model_path}")
                else:
                    print(f"{model_type.upper()} model not found at {model_path}")
            except Exception as e:
                print(f"Error loading {model_type.upper()} model: {str(e)}")
        
        # Try to load scaler (give priority to LSTM model's scaler if available)
        if 'lstm' in self.model_paths:
            scaler_path = self.model_paths['lstm'].replace('.pt', '_scaler.pkl')
            if os.path.exists(scaler_path):
                import pickle
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"Loaded scaler from {scaler_path}")
        
        # If no scaler was found, create one from the full dataset
        if self.scaler is None:
            try:
                print("Warning: No scaler found. Will create one from full dataset using prepare_enhanced_data.")
                # Load the full dataset
                full_data_path = '/mnt/e/dst-forecasting/dst_data.csv'
                if os.path.exists(full_data_path):
                    full_df = read_dst_csv(full_data_path)
                    print(f"Loaded {len(full_df)} records for creating scaler")
                    
                    # Use prepare_enhanced_data to get a properly calibrated scaler
                    _, _, _, _, self.scaler, _ = prepare_enhanced_data(
                        full_df,
                        target_col='Dst',
                        seq_length=24,  # Typical sequence length
                        remove_outliers=True,
                        outlier_method='iqr',
                        outlier_threshold=2.0,
                        add_cycles=False,  # Don't need the cyclical features for the scaler
                        scale_method='robust'  # Use RobustScaler
                    )
                    print("Created and fit scaler using prepare_enhanced_data on the complete dataset")
                else:
                    print(f"Warning: Full dataset not found at {full_data_path}")
            except Exception as e:
                print(f"Error creating scaler from full dataset: {e}")
                print("Creating a basic RobustScaler as fallback")
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
    
    async def download_latest_data(self):
        """
        Download the latest Dst data
        
        Returns:
            pandas.DataFrame: DataFrame with timestamp and Dst values
        """
        print("Downloading latest data...")
        try:
            # Get current year and previous year
            current_year = datetime.now(timezone.utc).year
            years = [current_year - 1, current_year]
            
            # Fetch data using the fetch_yearly_data function from geomag.py
            raw_data = await fetch_yearly_data(years=years)
            
            # Parse and clean the data
            parsed_df = parse_data(raw_data)
            cleaned_df = clean_data(parsed_df)
            
            print(f"Downloaded {len(cleaned_df)} records from {cleaned_df['timestamp'].min()} to {cleaned_df['timestamp'].max()}")
            cleaned_df.to_csv('/mnt/e/dst-forecasting/dst_data_new.csv', index=False)
            return cleaned_df
        
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            return None
    
    def load_historical_data(self):
        """
        Load historical data from CSV file
        
        Returns:
            pandas.DataFrame: DataFrame with timestamp and Dst values
        """
        if not self.data_path or not os.path.exists(self.data_path):
            print(f"Data path not provided or file not found: {self.data_path}")
            return None
        
        try:
            df = read_dst_csv(self.data_path)
            print(f"Loaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
        except Exception as e:
            print(f"Error loading historical data: {str(e)}")
            return None
    
    def prepare_forecast_data(self, df, seq_length=24, remove_outliers=True, outlier_threshold=2.0):
        """
        Prepare data for forecasting
        
        Args:
            df (pandas.DataFrame): DataFrame with timestamp and Dst columns
            seq_length (int): Sequence length for model input
            remove_outliers (bool): Whether to remove or cap outliers
            outlier_threshold (float): Threshold for outlier detection
            
        Returns:
            tuple: (X, timestamps, feature_cols)
        """
        # Make sure we have enough data
        if len(df) < seq_length + 3:  # Need extra points for previous, current, next
            raise ValueError(f"Not enough data points. Need at least {seq_length + 3}, got {len(df)}")
        
        # Save timestamps for reference - convert to numpy array of Python datetime objects
        timestamps = np.array([ts.to_pydatetime() for ts in df['timestamp']])
        print(timestamps)
        # Handle data preparation for forecasting
        print("Preparing data for forecasting...")
        
        # Create a small portion with the last sequence_length + 3 points
        forecast_data = df.tail(seq_length + 3).reset_index(drop=True)
        
        # If we don't have a scaler, create one from the full dataset
        if self.scaler is None:
            try:
                print("No scaler available. Creating one from the full dataset...")
                full_data_path = '/mnt/e/dst-forecasting/dst_data.csv'
                if os.path.exists(full_data_path):
                    full_df = read_dst_csv(full_data_path)
                    # Use prepare_enhanced_data to get a properly calibrated scaler
                    _, _, _, _, self.scaler, _ = prepare_enhanced_data(
                        full_df,
                        target_col='Dst',
                        seq_length=24,
                        remove_outliers=remove_outliers,
                        outlier_method='iqr',
                        outlier_threshold=outlier_threshold,
                        add_cycles=False,
                        scale_method='robust'
                    )
                    print("Created scaler from full dataset")
                else:
                    print("Full dataset not found. Creating scaler from current data.")
                    # Fall back to a simple RobustScaler on the current data
                    from sklearn.preprocessing import RobustScaler
                    self.scaler = RobustScaler()
                    dst_values = forecast_data['Dst'].values.reshape(-1, 1)
                    self.scaler.fit(dst_values)
            except Exception as e:
                print(f"Error creating scaler: {e}")
                print("Proceeding without scaling")
        
        # Process the forecast data for prediction
        
        # Handle outliers if requested (consistent with prepare_enhanced_data)
        if remove_outliers:
            print(f"Handling outliers using IQR method with threshold {outlier_threshold}...")
            # Calculate IQR stats
            Q1 = forecast_data['Dst'].quantile(0.25)
            Q3 = forecast_data['Dst'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            
            # Count outliers before capping
            n_outliers = ((forecast_data['Dst'] < lower_bound) | (forecast_data['Dst'] > upper_bound)).sum()
            
            if n_outliers > 0:
                print(f"Detected {n_outliers} outliers ({n_outliers/len(forecast_data)*100:.2f}%)")
                print(f"Capping outliers to range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                # Cap outliers to the bounds
                forecast_data['Dst'] = forecast_data['Dst'].clip(lower=lower_bound, upper=upper_bound)
        
        # Add cyclical time features
        df_with_features = add_cyclical_features(forecast_data)
        
        # Get feature columns (all except timestamp and Dst)
        feature_cols = [col for col in df_with_features.columns if col not in ['timestamp', 'Dst']]
        self.feature_cols = feature_cols
        print(df_with_features['Dst'],'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        # Apply scaling using our scaler
        if self.scaler:
            dst_values = df_with_features['Dst'].values.reshape(-1, 1)
            scaled_dst = self.scaler.transform(dst_values)
            df_with_features['Dst'] = scaled_dst
        
        # Create input sequences for previous, current, and next time points
        X_prev = np.expand_dims(self.create_input_sequence(df_with_features[:-2], seq_length), axis=0)
        X_current = np.expand_dims(self.create_input_sequence(df_with_features[:-1], seq_length), axis=0)
        X_next = np.expand_dims(self.create_input_sequence(df_with_features, seq_length), axis=0)
        # Stack them together
        X = np.vstack([X_prev, X_current, X_next])
        
        # Generate the next future timestamp (last timestamp + 1 hour)
        last_timestamp = timestamps[-1]
        next_timestamp = last_timestamp + timedelta(hours=1)
        
        # Create array with timestamps for previous, current, and future prediction
        forecast_timestamps = np.array([timestamps[-2], timestamps[-1], next_timestamp])
        
        return X, forecast_timestamps, feature_cols
    
    def create_input_sequence(self, df, seq_length):
        """
        Create a single input sequence for the model
        
        Args:
            df (pandas.DataFrame): DataFrame with features
            seq_length (int): Sequence length
            
        Returns:
            numpy.array: Input sequence
        """
        # Ensure we have enough data
        if len(df) < seq_length:
            raise ValueError(f"Not enough data points for sequence. Need {seq_length}, got {len(df)}")
        
        # Extract the last seq_length points
        sequence_df = df.tail(seq_length)
        
        # Combine Dst and feature columns
        cols = ['Dst'] + self.feature_cols if self.feature_cols else ['Dst']
        sequence = sequence_df[cols].values
        
        return sequence
    
    def forecast(self, X):
        """
        Generate forecasts using all loaded models
        
        Args:
            X (numpy.array): Input sequences for previous, current, and next time points
            
        Returns:
            dict: Dictionary mapping model types to predictions
        """
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Initialize predictions dictionary
        predictions = {model_type: np.zeros(3) for model_type in self.loaded_models}  # [prev, current, next]
        
        # Generate predictions with each model
        for model_type, model in self.models.items():
            try:
                model.eval()
                with torch.no_grad():
                    outputs = model(X_tensor).cpu().numpy()
                
                # Inverse transform if scaler is available
                if self.scaler:
                    preds = self.scaler.inverse_transform(outputs).flatten()
                else:
                    preds = outputs.flatten()
                
                predictions[model_type] = preds
            except Exception as e:
                print(f"Error generating {model_type.upper()} predictions: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return predictions
    
    def print_results(self, timestamps, ground_truth, predictions):
        """
        Print forecasting results
        
        Args:
            timestamps: Array of timestamps for previous, current, and future predictions
            ground_truth: Array of actual Dst values
            predictions: Dictionary mapping model types to prediction arrays
        """
        # Create a nice table header
        print("\n" + "="*80)
        print("DST FORECASTING RESULTS")
        print("="*80)
        print(ground_truth)
        print(predictions)
        # Convert numpy.datetime64 to Python datetime if needed
        formatted_timestamps = []
        for ts in timestamps[-3:]:
            if isinstance(ts, np.datetime64):
                ts = pd.Timestamp(ts).to_pydatetime()
            formatted_timestamps.append(ts)
        
        # Format timestamps
        time_strs = [ts.strftime('%Y-%m-%d %H:00') for ts in formatted_timestamps]
        
        # Prepare results table data
        table_data = []
        
        # Add each model's results
        for time_idx, time_str in enumerate(time_strs):
            # Determine if this is a past, present, or future prediction
            is_forecast = time_idx == 2  # Index 2 is the future prediction
            time_label = time_str + (' (FORECAST)' if is_forecast else '')
            
            # Get actual value if available (only for past and present)
            actual_value = None
            if not is_forecast:
                # For past and present (indices 0 and 1), get actual values from ground truth
                actual_idx = -2 + time_idx  # -2 for past, -1 for present
                if abs(actual_idx) <= len(ground_truth):
                    actual_value = ground_truth[actual_idx]
            
            # Add each model's prediction
            for model_type in sorted(predictions.keys()):
                model_preds = predictions[model_type]
                pred_value = model_preds[time_idx]
                
                # Calculate error if actual value is available
                error = pred_value - actual_value if actual_value is not None else None
                
                row = {
                    'Model': model_type.upper(),
                    'Time': time_label,
                    'Actual': f"{actual_value:.2f}" if actual_value is not None else "Unknown",
                    'Predicted': f"{pred_value:.2f}",
                    'Error': f"{error:.2f}" if error is not None else "N/A"
                }
                table_data.append(row)
        
        # Create a DataFrame for nice display
        results_df = pd.DataFrame(table_data)
        print(results_df.to_string(index=False))
        
        # Print ensemble forecast for next hour
        if len(predictions) > 1:
            ensemble_values = np.array([preds[2] for preds in predictions.values()])
            ensemble_mean = ensemble_values.mean()
            print("\n" + "="*80)
            print(f"ENSEMBLE FORECAST FOR {time_strs[2]}: {ensemble_mean:.2f}")
            print(f"Model forecasts: {', '.join([f'{m.upper()}: {p[2]:.2f}' for m, p in predictions.items()])}")
            print("="*80)
        
        # Visualize the predictions
        self.plot_forecasts(timestamps[-24:], ground_truth[-24:], predictions)
    
    def plot_forecasts(self, timestamps, ground_truth, predictions):
        """
        Plot the forecast results
        
        Args:
            timestamps: Array of timestamps (should include past, present, and future)
            ground_truth: Array of actual values (past data only)
            predictions: Dictionary mapping model types to prediction arrays
        """
        plt.figure(figsize=(14, 7))
        
        # Ensure timestamps and ground_truth have the same dimension
        # Both arrays should be the same length for plotting
        if len(timestamps) != len(ground_truth):
            print(f"Warning: Timestamps ({len(timestamps)}) and ground truth ({len(ground_truth)}) dimensions don't match")
            # Use the smaller length to avoid dimension mismatch
            min_len = min(len(timestamps), len(ground_truth))
            historical_data_timestamps = timestamps[-min_len:]
            historical_data_values = ground_truth[-min_len:]
        else:
            historical_data_timestamps = timestamps
            historical_data_values = ground_truth
        
        # Plot the historical data
        plt.plot(historical_data_timestamps, historical_data_values, 'k-', label='Actual', linewidth=2)
        
        # Get timestamps for the predictions (past, present, future)
        # Make sure we're using the last timestamp from our array and adding one hour for the future
        if len(timestamps) >= 2:
            pred_times = np.array([timestamps[-2], timestamps[-1], timestamps[-1] + timedelta(hours=1)])
        else:
            # Fallback if we don't have enough timestamps
            last_time = timestamps[-1]
            pred_times = np.array([last_time - timedelta(hours=1), last_time, last_time + timedelta(hours=1)])
        
        # Color map for different models
        colors = {'lstm': 'r', 'transformer': 'b', 'tcn': 'g', 'nbeats': 'c'}
        markers = {'lstm': 'o', 'transformer': 's', 'tcn': '^', 'nbeats': 'D'}
        
        # Plot each model's predictions
        for model_type, preds in predictions.items():
            color = colors.get(model_type, 'y')  # Default to yellow if model not in color map
            marker = markers.get(model_type, 'x')  # Default to x if model not in marker map
            
            # Plot past and present predictions
            plt.plot(pred_times[:2], preds[:2], f'{color}-', 
                    label=f'{model_type.upper()} Prediction', 
                    alpha=0.8, linewidth=2, markersize=7)
            
            # Plot future prediction (empty marker)
            plt.plot(pred_times[2], preds[2], f'{color}{marker}', 
                    markersize=9, markerfacecolor='none', markeredgewidth=2)
        
        # Add ensemble forecast point if multiple models are available
        if len(predictions) > 1:
            ensemble_values = np.array([preds[2] for preds in predictions.values()])
            ensemble_mean = ensemble_values.mean()
            plt.plot(pred_times[2], ensemble_mean, 'm*', 
                    label='Ensemble Forecast', markersize=14)
        
        # Add vertical line to separate history from future
        plt.axvline(x=timestamps[-1], color='gray', linestyle='--', alpha=0.7)
        
        # Format the plot
        plt.title('Dst Forecasting Results')
        plt.xlabel('Time')
        plt.ylabel('Dst Value (nT)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Format x-axis with dates
        plt.gcf().autofmt_xdate()
        
        # Save the plot
        plt.tight_layout()
        forecast_dir = "/mnt/e/dst-forecasting/forecasts"
        if not os.path.exists(forecast_dir):
            os.makedirs(forecast_dir)
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"{forecast_dir}/dst_forecast_{timestamp_str}.png", dpi=300)
        print(f"\nForecast plot saved to {forecast_dir}/dst_forecast_{timestamp_str}.png")
        plt.close()

async def main():
    """Main function to run the forecaster"""
    # Add safe globals for numpy scalar types
    import numpy as np
    print("Registering numpy scalar types as safe globals for PyTorch loading...")
    try:
        serialization.add_safe_globals([np.float64, np.float32, np.int64, np.int32, 
                                      np._core.multiarray.scalar])
    except (AttributeError, ImportError) as e:
        print(f"Warning: Could not register safe globals: {e}")
        print("This is normal for older PyTorch versions and shouldn't affect functionality")
    
    parser = argparse.ArgumentParser(description='Forecast Dst values using trained models')
    parser.add_argument('--lstm', default='/mnt/e/dst-forecasting/checkpoints/dst_lstm_pytorch.pt',
                        help='Path to trained LSTM model')
    parser.add_argument('--transformer', default='/mnt/e/dst-forecasting/checkpoints/dst_transformer_model.pt',
                        help='Path to trained Transformer model')
    parser.add_argument('--tcn', default='/mnt/e/dst-forecasting/checkpoints/dst_tcn_model.pt',
                        help='Path to trained TCN model')
    parser.add_argument('--nbeats', default='/mnt/e/dst-forecasting/checkpoints/dst_nbeats_model.pt',
                        help='Path to trained N-BEATS model')
    parser.add_argument('--data', default='/mnt/e/dst-forecasting/dst_data.csv',
                        help='Path to historical data CSV (optional, will download if not provided)')
    parser.add_argument('--download', action='store_true',
                        help='Force download latest data even if historical data is provided')
    parser.add_argument('--no-scaling', action='store_true',
                        help='Disable data scaling even if no scaler is found')
    parser.add_argument('--models', default='lstm,transformer,tcn,nbeats',
                        help='Comma-separated list of models to use (default: all)')
    parser.add_argument('--full-data', default='/mnt/e/dst-forecasting/dst_data.csv',
                        help='Path to full historical data for creating scaler')
    args = parser.parse_args()
    
    # Determine which models to use based on the --models argument
    requested_models = args.models.lower().split(',')
    model_paths = {}
    
    # Add paths only for requested models
    if 'lstm' in requested_models:
        model_paths['lstm'] = args.lstm
    if 'transformer' in requested_models:
        model_paths['transformer'] = args.transformer
    if 'tcn' in requested_models:
        model_paths['tcn'] = args.tcn
    if 'nbeats' in requested_models:
        model_paths['nbeats'] = args.nbeats
    
    if not model_paths:
        print("Error: No valid models specified. Please provide at least one model.")
        return
    
    # Initialize forecaster with specified models
    forecaster = DstForecaster(
        model_paths=model_paths,
        data_path=args.data if not args.download else None
    )
    
    # Load models
    forecaster.load_models()
    
    if not forecaster.loaded_models:
        print("Error: No models were loaded successfully. Cannot continue with forecasting.")
        return
    
    # Load or download data
    if args.download or not args.data:
        df = await forecaster.download_latest_data()
        if df is None or len(df) < 27:  # Need at least seq_length + 3 points
            print("Not enough data downloaded. Falling back to historical data if available.")
            df = forecaster.load_historical_data()
    else:
        df = forecaster.load_historical_data()
    
    if df is None or len(df) < 27:
        print("Error: Not enough data available for forecasting.")
        return
    # print(df)
    # Prepare data for forecasting
    try:
        X, timestamps, feature_cols = forecaster.prepare_forecast_data(df, seq_length=24)
        
        # Generate forecasts
        predictions = forecaster.forecast(X)
        
        # Get ground truth values
        ground_truth = df['Dst'].values
        
        # Print and visualize results
        forecaster.print_results(timestamps, ground_truth, predictions)
        
    except Exception as e:
        print(f"Error during forecasting: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
