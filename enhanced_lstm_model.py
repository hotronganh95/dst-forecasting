import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Import from existing modules
from read_dst_data import read_dst_csv, split_by_gaps
from analyze_dst_data import detect_outliers, analyze_seasonality

# Keep the existing functions for data preparation
def add_cyclical_features(df):
    """
    Add cyclical features based on timestamp
    
    Args:
        df (pandas.DataFrame): DataFrame with timestamp column
        
    Returns:
        pandas.DataFrame: DataFrame with added cyclical features
    """
    df = df.copy()
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    
    # Create cyclical features using sin and cos transformations
    # Hour of day (24 hours)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of month (assuming 31 days max)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # Month of year (12 months)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Day of week (7 days)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Day of year (365/366 days)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Drop the original time components
    df = df.drop(['hour', 'day', 'month', 'dayofweek', 'dayofyear'], axis=1)
    
    return df

def prepare_enhanced_data(df, target_col='Dst', seq_length=24, 
                         remove_outliers=True, outlier_method='iqr', outlier_threshold=2.0,
                         add_cycles=True, scale_method='robust'):
    """
    Prepare data for LSTM model with enhanced preprocessing
    
    Args:
        df (pandas.DataFrame): DataFrame with timestamp and target column
        target_col (str): Target column name
        seq_length (int): Sequence length for LSTM input
        remove_outliers (bool): Whether to remove outliers
        outlier_method (str): Method for outlier detection
        outlier_threshold (float): Threshold for outlier detection
        add_cycles (bool): Whether to add cyclical features
        scale_method (str): Scaling method ('minmax', 'robust', or 'none')
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler, feature_names)
    """
    print("Preparing enhanced data...")
    df = df.copy()
    
    # Handle outliers if requested
    if remove_outliers:
        print(f"Detecting outliers using {outlier_method} method with threshold {outlier_threshold}...")
        outlier_df = detect_outliers(df, target_col, method=outlier_method, threshold=outlier_threshold)
        n_outliers = outlier_df['is_outlier'].sum()
        print(f"Detected {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")
        
        if n_outliers > 0:
            # Option 1: Remove outliers
            # df = outlier_df[~outlier_df['is_outlier']].drop(['is_outlier', 'outlier_score'], axis=1)
            
            # Option 2: Cap outliers (more data-preserving)
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            
            # Cap outliers
            df[target_col] = df[target_col].clip(lower=lower_bound, upper=upper_bound)
            print(f"Capped outliers to range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Add cyclical features if requested
    feature_columns = []
    if add_cycles:
        print("Adding cyclical features...")
        df = add_cyclical_features(df)
        # All columns except timestamp and target
        feature_columns = [col for col in df.columns if col not in [target_col, 'timestamp']]
    
    # Split into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Scale the data if requested
    scaler = None
    feature_scaler = None
    if scale_method != 'none':
        if scale_method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif scale_method == 'robust':
            scaler = RobustScaler()  # More robust to outliers
            
        # Scale target column
        train_target = train_df[target_col].values.reshape(-1, 1)
        scaled_train_target = scaler.fit_transform(train_target)
        
        test_target = test_df[target_col].values.reshape(-1, 1)
        scaled_test_target = scaler.transform(test_target)
        
        # Replace original values with scaled values
        train_df[target_col] = scaled_train_target
        test_df[target_col] = scaled_test_target
        
        # Also scale feature columns if they exist
        if feature_columns:
            feature_scaler = RobustScaler()
            train_features = train_df[feature_columns].values
            scaled_train_features = feature_scaler.fit_transform(train_features)
            
            test_features = test_df[feature_columns].values
            scaled_test_features = feature_scaler.transform(test_features)
            
            for i, col in enumerate(feature_columns):
                train_df[col] = scaled_train_features[:, i]
                test_df[col] = scaled_test_features[:, i]
    
    # Create sequences
    if feature_columns:
        # Multivariate case
        print(f"Creating multivariate sequences with {len(feature_columns)+1} features...")
        X_train, y_train = create_multivariate_sequences(train_df, 
                                                     seq_length=seq_length,
                                                     target_col=target_col, 
                                                     feature_cols=feature_columns)
        
        X_test, y_test = create_multivariate_sequences(test_df,
                                                   seq_length=seq_length,
                                                   target_col=target_col,
                                                   feature_cols=feature_columns)
    else:
        # Univariate case
        print("Creating univariate sequences...")
        X_train, y_train = create_sequences(train_df[target_col].values, seq_length)
        X_test, y_test = create_sequences(test_df[target_col].values, seq_length)
    
    print(f"Training sequences: {X_train.shape}, Test sequences: {X_test.shape}")
    return X_train, y_train, X_test, y_test, scaler, feature_columns

def create_sequences(data, seq_length):
    """
    Create input sequences and targets for LSTM model
    
    Args:
        data (numpy.array): Time series data
        seq_length (int): Length of input sequence
        
    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def create_multivariate_sequences(df, seq_length, target_col, feature_cols):
    """
    Create sequences for multivariate time series data
    
    Args:
        df (pandas.DataFrame): DataFrame with target and feature columns
        seq_length (int): Sequence length
        target_col (str): Target column name
        feature_cols (list): List of feature column names
        
    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values
    """
    X, y = [], []
    
    # Combine target and features
    data = df[[target_col] + feature_cols].values
    
    # Create sequences
    for i in range(len(df) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Target is the first column
    
    return np.array(X), np.array(y)

# New PyTorch LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the last time step output
        out = self.fc(out[:, -1, :])
        return out

def train_pytorch_model(X_train, y_train, X_test, y_test, 
                       input_size, hidden_size=64, 
                       num_layers=2, dropout=0.3,
                       batch_size=32, learning_rate=0.001,
                       num_epochs=100, patience=20,
                       device=None, model_save_path=None):
    """
    Train a PyTorch LSTM model
    
    Args:
        X_train, y_train, X_test, y_test: Training and test data
        input_size: Number of features
        hidden_size: Size of LSTM hidden units
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience
        device: PyTorch device (default: auto-detect)
        model_save_path: Path to save the best model during training (default: None)
        
    Returns:
        tuple: (model, training_history, best_epoch)
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=10, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0
    best_epoch = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        with tqdm(train_loader, unit="batch") as train_pbar:
            train_pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
            
            for X_batch, y_batch in train_pbar:
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                train_pbar.set_postfix(loss=loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                val_loss = criterion(outputs, y_batch)
                val_losses.append(val_loss.item())
        
        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.8f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0
            best_epoch = epoch + 1
            print(f"New best model at epoch {best_epoch} with validation loss: {best_val_loss:.6f}")
            
            # Save the best model during training if a path is provided
            if model_save_path:
                # Create checkpoints directory if it doesn't exist
                import os
                checkpoint_dir = "/mnt/e/dst-forecasting/checkpoints"
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                
                # Create checkpoint paths with the checkpoints directory
                model_name = os.path.basename(model_save_path)
                checkpoint_path = f"{checkpoint_dir}/{model_name}_epoch{best_epoch}.pt"
                best_path = f"{checkpoint_dir}/{model_name}_best.pt"
                
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path}")
                
                # Also save as 'best_model.pt' (overwrite)
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, best_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch}.")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_epoch

def evaluate_pytorch_model(model, X_test, y_test, scaler=None, df=None, target_col='Dst', device=None, save_path=None):
    """
    Evaluate the PyTorch LSTM model
    
    Args:
        model: Trained PyTorch model
        X_test, y_test: Test data
        scaler: Scaler used for the target variable
        df: Original DataFrame for additional analysis
        target_col: Target column name
        device: PyTorch device (default: same as model)
        save_path: Base path to save visualizations (default: /mnt/e/dst-forecasting/outputs)
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Set default save path if not provided
    if save_path is None:
        save_path = "/mnt/e/dst-forecasting/outputs"
    
    # Create directory if it doesn't exist
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get device from model if not provided
    if device is None:
        device = next(model.parameters()).device
    
    # Convert data to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.cpu().numpy()
    
    # If scaler is provided, inverse transform predictions and actual values
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_pred = y_pred.flatten()
    
    # Calculate errors
    errors = y_test - y_pred.flatten()
    abs_errors = np.abs(errors)
    
    # Calculate metrics
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_errors)
    mape = np.mean(np.abs(errors / (y_test + 1e-10))) * 100  # Add small epsilon to avoid division by zero
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot predictions vs actual and save to file
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual', alpha=0.7)
    plt.plot(y_pred.flatten(), label='Predicted', alpha=0.7)
    plt.title(f'Dst Prediction Performance (RMSE: {rmse:.2f}, MAE: {mae:.2f})')
    plt.xlabel('Time Steps')
    plt.ylabel('Dst Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/prediction_performance.png", dpi=300)
    plt.close()
    print(f"Prediction performance plot saved to {save_path}/prediction_performance.png")
    
    # Plot error distribution and save to file
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Limit the number of points to display (max 5000)
    max_points = 5000
    if len(y_test) > max_points:
        # Random sampling if too many points
        idx = np.random.choice(len(y_test), max_points, replace=False)
        y_test_sample = y_test[idx]
        y_pred_sample = y_pred.flatten()[idx]
    else:
        y_test_sample = y_test
        y_pred_sample = y_pred.flatten()
    
    plt.scatter(y_test_sample, y_pred_sample, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/error_analysis.png", dpi=300)
    plt.close()
    print(f"Error analysis plot saved to {save_path}/error_analysis.png")
    
    # If original DataFrame is provided, analyze error by time patterns
    if df is not None:
        try:
            # Select test portion of DataFrame (adjusted for sequence length)
            test_size = len(y_test)
            train_size = len(df) - test_size - X_test.shape[1]  # Adjust for sequence length
            
            # Get timestamps for test predictions
            test_timestamps = df['timestamp'].iloc[train_size + X_test.shape[1]:train_size + X_test.shape[1] + test_size]
            
            # Ensure predicted values are 1-dimensional
            predictions_1d = y_pred.flatten()
            
            # Create error DataFrame with 1-dimensional arrays
            error_df = pd.DataFrame()
            error_df['timestamp'] = test_timestamps.reset_index(drop=True)
            error_df['actual'] = y_test
            error_df['predicted'] = predictions_1d
            error_df['error'] = errors
            error_df['abs_error'] = abs_errors
            
            # Extract time components
            error_df['hour'] = error_df['timestamp'].dt.hour
            error_df['day'] = error_df['timestamp'].dt.day
            error_df['month'] = error_df['timestamp'].dt.month
            error_df['year'] = error_df['timestamp'].dt.year
            error_df['dayofweek'] = error_df['timestamp'].dt.dayofweek
            
            # Plot error by time patterns and save to file
            plt.figure(figsize=(14, 16))
            
            plt.subplot(4, 1, 1)
            hourly_error = error_df.groupby('hour')['abs_error'].mean()
            plt.bar(hourly_error.index, hourly_error.values)
            plt.title('Mean Absolute Error by Hour of Day')
            plt.xlabel('Hour')
            plt.ylabel('MAE')
            plt.grid(True)
            plt.xticks(range(0, 24, 2))
            
            plt.subplot(4, 1, 2)
            daily_error = error_df.groupby('day')['abs_error'].mean()
            plt.bar(daily_error.index, daily_error.values)
            plt.title('Mean Absolute Error by Day of Month')
            plt.xlabel('Day')
            plt.ylabel('MAE')
            plt.grid(True)
            
            plt.subplot(4, 1, 3)
            monthly_error = error_df.groupby('month')['abs_error'].mean()
            plt.bar(monthly_error.index, monthly_error.values)
            plt.title('Mean Absolute Error by Month')
            plt.xlabel('Month')
            plt.ylabel('MAE')
            plt.grid(True)
            plt.xticks(range(1, 13))
            
            plt.subplot(4, 1, 4)
            weekday_error = error_df.groupby('dayofweek')['abs_error'].mean()
            plt.bar(weekday_error.index, weekday_error.values)
            plt.title('Mean Absolute Error by Day of Week')
            plt.xlabel('Day of Week (0 = Monday)')
            plt.ylabel('MAE')
            plt.grid(True)
            plt.xticks(range(0, 7))
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/temporal_error_patterns.png", dpi=300)
            plt.close()
            print(f"Temporal error patterns plot saved to {save_path}/temporal_error_patterns.png")
        except Exception as e:
            print(f"Warning: Could not analyze temporal error patterns: {e}")
            import traceback
            traceback.print_exc()
    
    return metrics, (y_test, y_pred.flatten())

def plot_training_history(history, save_path=None):
    """
    Plot the training history of a PyTorch model
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot (default: /mnt/e/dst-forecasting/outputs)
    """
    # Set default save path if not provided
    if save_path is None:
        save_path = "/mnt/e/dst-forecasting/outputs"
    
    # Create directory if it doesn't exist
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(history['learning_rate'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_history.png", dpi=300)
    plt.close()
    print(f"Training history plot saved to {save_path}/training_history.png")

def save_pytorch_model(model, file_path, feature_cols=None, scaler=None):
    """
    Save PyTorch model and associated metadata
    
    Args:
        model: PyTorch model
        file_path: Path to save the model
        feature_cols: List of feature columns
        scaler: Scaler used for data
    """
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_columns': feature_cols,
        'input_size': model.lstm.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers
    }, file_path)
    
    print(f"Model saved to {file_path}")
    
    # Save scaler separately if provided
    if scaler is not None:
        scaler_path = file_path.replace('.pt', '_scaler.pkl')
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    # Load data
    file_path = "/mnt/e/dst-forecasting/dst_data.csv"
    dst_data = read_dst_csv(file_path)
    
    # Use only continuous segments of data
    segments = split_by_gaps(dst_data, gap_threshold='2H')
    
    # Use the longest segment for demonstration
    longest_segment = max(segments, key=len)
    print(f"Using longest continuous segment with {len(longest_segment)} records")
    
    # Display seasonality analysis for reference
    print("\nSeasonality analysis (for reference):")
    analyze_seasonality(longest_segment, 'Dst')
    
    # Prepare enhanced data with cyclical features and outlier handling
    X_train, y_train, X_test, y_test, scaler, feature_cols = prepare_enhanced_data(
        longest_segment,
        target_col='Dst',
        seq_length=24,  # Use 24 hours of history
        remove_outliers=True,
        outlier_method='iqr', 
        outlier_threshold=2.0,
        add_cycles=True,
        scale_method='robust'
    )
    
    # Set device for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get input size (number of features)
    input_size = X_train.shape[2]  # shape: [batch, sequence, features]
    
    # Build and train the PyTorch model
    model, history, best_epoch = train_pytorch_model(
        X_train, y_train,
        X_test, y_test,
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=100,
        patience=20,
        device=device,
        model_save_path='lstm_model'  # Changed to just the model name, not full path
    )
    
    # Create output directory
    import os
    output_dir = "/mnt/e/dst-forecasting/outputs/lstm"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot training history
    plot_training_history(history, save_path=output_dir)
    
    # Evaluate model
    metrics, (y_true, y_pred) = evaluate_pytorch_model(model, X_test, y_test, scaler, longest_segment, save_path=output_dir)
    
    # Save the model
    save_pytorch_model(
        model, 
        '/mnt/e/dst-forecasting/dst_lstm_pytorch.pt', 
        feature_cols, 
        scaler
    )
    
    print(f"\nBest model was at epoch {best_epoch}")
    print("\nPyTorch model training and evaluation complete")
