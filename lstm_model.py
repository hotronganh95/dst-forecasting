import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from read_dst_data import read_dst_csv, split_by_gaps

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

def prepare_data(df, feature_col='Dst', seq_length=24, scale=True, diff=False):
    """
    Prepare data for LSTM model with options for scaling and differencing
    
    Args:
        df (pandas.DataFrame): DataFrame with timestamp and feature columns
        feature_col (str): Name of the feature column
        seq_length (int): Length of input sequence (past values to predict from)
        scale (bool): Whether to scale the data
        diff (bool): Whether to use differencing (helps with stationary data)
        
    Returns:
        tuple: (X, y, scaler, original_data) 
    """
    data = df[feature_col].values.reshape(-1, 1)
    original_data = data.copy()
    
    # Apply differencing if requested
    if diff:
        data = np.diff(data, axis=0)
        # Add back the first value that was lost in differencing
        data = np.vstack([original_data[0], data])
    
    # Apply scaling if requested
    scaler = None
    if scale:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data, seq_length)
    
    return X, y, scaler, original_data

def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2):
    """
    Build an LSTM model for time series prediction
    
    Args:
        input_shape (tuple): Shape of input data (seq_length, features)
        lstm_units (int): Number of LSTM units
        dropout_rate (float): Dropout rate
        
    Returns:
        tensorflow.keras.Model: Compiled LSTM model
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=lstm_units, 
                  return_sequences=True, 
                  input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def inverse_transform_predictions(predictions, scaler=None, original_data=None, diff=False):
    """
    Transform predictions back to original scale and undo differencing if needed
    
    Args:
        predictions (numpy.array): Model predictions
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler used for data
        original_data (numpy.array): Original data before transformations
        diff (bool): Whether differencing was applied
        
    Returns:
        numpy.array: Predictions in original scale
    """
    # Reshape predictions to 2D if needed
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    
    # Inverse scaling if applicable
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
    
    # Undo differencing if applicable
    if diff:
        # To undo differencing, we need to add each prediction to the previous actual value
        undifferenced = np.zeros_like(predictions)
        for i in range(len(predictions)):
            undifferenced[i] = original_data[i + 1 - 1] + predictions[i]
        return undifferenced
    
    return predictions

def train_and_evaluate(df, feature_col='Dst', seq_length=24, test_split=0.2, 
                      use_diff=True, use_scale=True, epochs=50, batch_size=32):
    """
    Train and evaluate an LSTM model on the given data
    
    Args:
        df (pandas.DataFrame): DataFrame with timestamp and feature columns
        feature_col (str): Name of the feature column
        seq_length (int): Length of input sequence
        test_split (float): Fraction of data to use for testing
        use_diff (bool): Whether to use differencing
        use_scale (bool): Whether to scale the data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (model, history, evaluation_metrics, predictions)
    """
    # Prepare data
    X, y, scaler, original_data = prepare_data(
        df, feature_col, seq_length, use_scale, use_diff
    )
    
    # Split into train and test sets
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build and train model
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions on test set
    test_predictions = model.predict(X_test)
    
    # Transform predictions back to original scale
    original_test_preds = inverse_transform_predictions(
        test_predictions, scaler, original_data[split_idx:], use_diff
    )
    original_y_test = inverse_transform_predictions(
        y_test, scaler, original_data[split_idx:], use_diff
    )
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(original_y_test, original_test_preds),
        'rmse': np.sqrt(mean_squared_error(original_y_test, original_test_preds)),
        'mae': mean_absolute_error(original_y_test, original_test_preds)
    }
    
    return model, history, metrics, (original_y_test, original_test_preds)

def plot_results(history, actual, predictions, title='LSTM Model Predictions'):
    """
    Plot training history and prediction results
    
    Args:
        history (tensorflow.keras.callbacks.History): Training history
        actual (numpy.array): Actual values
        predictions (numpy.array): Predicted values
        title (str): Title for the prediction plot
    """
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot training history
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot predictions vs actual
    ax2.plot(actual, label='Actual', color='blue')
    ax2.plot(predictions, label='Predicted', color='red', alpha=0.7)
    ax2.set_title(title)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Dst Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data
    file_path = "/mnt/e/dst-forecasting/dst_data.csv"
    dst_data = read_dst_csv(file_path)
    
    # Use only continuous segments of data
    segments = split_by_gaps(dst_data, gap_threshold='2H')
    
    # Use the longest segment for demonstration
    longest_segment = max(segments, key=len)
    print(f"Using longest continuous segment with {len(longest_segment)} records")
    
    # Train model with differencing (helps with non-stationary data)
    model, history, metrics, (actual, predictions) = train_and_evaluate(
        longest_segment,
        feature_col='Dst',
        seq_length=24,  # Use 24 hours of history
        test_split=0.2,
        use_diff=True,  # Use differencing to focus on changes
        use_scale=True, # Scale the data
        epochs=50
    )
    
    # Print metrics
    print("\nModel Evaluation:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    
    # Plot results
    plot_results(history, actual, predictions, 'LSTM Dst Predictions')
    
    # Save model if needed
    model.save('/mnt/e/dst-forecasting/lstm_dst_model.h5')
    print("\nModel saved to '/mnt/e/dst-forecasting/lstm_dst_model.h5'")
