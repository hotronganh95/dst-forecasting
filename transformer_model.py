import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from read_dst_data import read_dst_csv, split_by_gaps
from analyze_dst_data import detect_outliers, analyze_seasonality
from enhanced_lstm_model import add_cyclical_features, prepare_enhanced_data, plot_training_history

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model
    """
    def __init__(self, d_model, max_seq_length=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter but should be saved and loaded with state_dict)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1), :]

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model adapted for time series forecasting
    """
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead, 
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection (forecasting head)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Use the encoding of the last time step for prediction
        x = x[:, -1, :]
        
        # Project to output
        output = self.output_projection(x)
        return output

class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        
        # Calculate padding to maintain sequence length (causal padding)
        padding = (kernel_size - 1) * dilation
        
        # Dilated causal convolution
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        ))
        
        # Second convolution
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        ))
        
        # Residual connection if input and output dimensions don't match
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Store the padding size for forward pass
        self.padding_size = padding
        
    def forward(self, x):
        # Save input for residual connection
        residual = self.residual(x)
        
        # First convolution block
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Remove the padding at the end to maintain causality
        # Make sure we get the exact same size by explicitly calculating the indices
        if self.padding_size > 0:
            out = out[:, :, :-self.padding_size]
            
        # Make sure residual has the same size as output
        if residual.size(2) > out.size(2):
            residual = residual[:, :, :out.size(2)]
        elif residual.size(2) < out.size(2):
            # Zero-pad residual if needed (should not happen with proper padding)
            zeros = torch.zeros(residual.size(0), residual.size(1), 
                            out.size(2) - residual.size(2)).to(residual.device)
            residual = torch.cat([residual, zeros], dim=2)
        
        # Add residual connection
        return self.relu(out + residual)

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for time series forecasting
    """
    def __init__(self, input_size, hidden_channels, kernel_size=3, dropout=0.2, num_layers=4):
        super(TemporalConvNet, self).__init__()
        
        self.input_size = input_size
        layers = []
        num_channels = [hidden_channels] * num_layers
        
        # Create TCN blocks with increasing dilation
        for i in range(num_layers):
            dilation = 2 ** i  # Exponentially increasing dilation
            in_channels = input_size if i == 0 else num_channels[i-1]
            layers.append(TCNBlock(
                in_channels=in_channels,
                out_channels=num_channels[i],
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        # TCN expects [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        
        try:
            # Pass through TCN blocks
            out = self.network(x)
            
            # Take the last time step for prediction
            out = out[:, :, -1]
            
            # Project to output
            out = self.output_layer(out)
            return out
        except Exception as e:
            print(f"Error in TCN forward pass: {e}")
            print(f"Input shape: {x.shape}")
            # Re-raise the exception to help with debugging
            raise

class NBeatsBlock(nn.Module):
    """
    Basic building block of N-BEATS model
    """
    def __init__(self, input_dim, theta_dim, hidden_dim, layers):
        super(NBeatsBlock, self).__init__()
        
        self.input_dim = input_dim
        self.theta_dim = theta_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        
        # Fully connected stack
        fc_layers = []
        for i in range(layers):
            if i == 0:
                fc_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
        
        self.fc_stack = nn.Sequential(*fc_layers)
        
        # Output layers for backcast and forecast
        self.theta_b = nn.Linear(hidden_dim, theta_dim)
        self.theta_f = nn.Linear(hidden_dim, theta_dim)
        
        # Backcast and forecast projection
        self.backcast_proj = nn.Linear(theta_dim, input_dim, bias=False)
        self.forecast_proj = nn.Linear(theta_dim, 1, bias=False)
        
    def forward(self, x):
        # Pass through fully connected stack
        out = self.fc_stack(x)
        
        # Get parameters for backcast and forecast
        theta_b = self.theta_b(out)
        theta_f = self.theta_f(out)
        
        # Get backcast and forecast
        backcast = self.backcast_proj(theta_b)
        forecast = self.forecast_proj(theta_f)
        
        return backcast, forecast

class NBeatsModel(nn.Module):
    """
    N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
    """
    def __init__(self, input_dim, hidden_dim=64, theta_dim=8, num_blocks=3, layers=4):
        super(NBeatsModel, self).__init__()
        
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_dim, theta_dim, hidden_dim, layers) 
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        # Flatten the input for fully connected layers (batch, seq_len, features) -> (batch, seq_len*features)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        residuals = x
        forecasts = 0
        
        # Apply double residual stacking
        for block in self.blocks:
            # Get backcast and forecast from the block
            backcast, block_forecast = block(residuals)
            
            # Update residuals and forecast
            residuals = residuals - backcast
            forecasts = forecasts + block_forecast
            
        return forecasts

def train_model(model, X_train, y_train, X_test, y_test, 
               batch_size=32, learning_rate=0.001, num_epochs=100, 
               patience=20, device=None, model_type=None, model_save_path=None):
    """
    Generic training function for different model architectures
    
    Args:
        model: Model to train
        X_train, y_train, X_test, y_test: Training and test data
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience
        device: PyTorch device (default: auto-detect)
        model_type: Type of model ('lstm', 'transformer', etc.)
        model_save_path: Base path to save model checkpoints (default: None)
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
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
                
                # Create a specific path for this model type
                if model_type:
                    model_name = f"{model_save_path}_{model_type}"
                else:
                    model_name = model_save_path
                    
                checkpoint_path = f"{checkpoint_dir}/{model_name}_epoch{best_epoch}.pt"
                best_path = f"{checkpoint_dir}/{model_name}_best.pt"
                
                # Save checkpoint with epoch number
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'model_type': model_type
                }, checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path}")
                
                # Also save as best model (overwrite)
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'model_type': model_type
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

def evaluate_model(model, X_test, y_test, scaler=None, df=None, device=None, model_type=None):
    """
    Evaluate model performance
    """
    # Create output directory for this specific model
    import os
    model_name = model_type if model_type else "model"
    save_path = f"/mnt/e/dst-forecasting/outputs/{model_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Use the enhanced evaluation function but with specific save path
    from enhanced_lstm_model import evaluate_pytorch_model
    return evaluate_pytorch_model(model, X_test, y_test, scaler, df, device=device, save_path=save_path)

def save_model(model, file_path, model_type, feature_cols=None, scaler=None):
    """
    Save model and metadata
    """
    # Create checkpoints directory if it doesn't exist
    import os
    checkpoint_dir = "/mnt/e/dst-forecasting/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Ensure the file path is in the checkpoints directory
    if not file_path.startswith(checkpoint_dir):
        # Extract just the filename from the path
        file_name = os.path.basename(file_path)
        file_path = os.path.join(checkpoint_dir, file_name)
    
    # Save model state
    metadata = {
        'model_state_dict': model.state_dict(),
        'feature_columns': feature_cols,
        'model_type': model_type
    }
    
    if model_type == 'transformer':
        metadata.update({
            'input_size': model.input_size,
            'd_model': model.d_model
        })
    elif model_type == 'tcn':
        metadata.update({
            'input_size': model.input_size
        })
    
    torch.save(metadata, file_path)
    print(f"Model saved to {file_path}")
    
    # Save scaler separately if provided
    if scaler is not None:
        scaler_path = file_path.replace('.pt', '_scaler.pkl')
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")

def load_trained_model(model_path, model_type, input_size=None, device=None):
    """
    Load a previously trained PyTorch model
    
    Args:
        model_path (str): Path to the saved model
        model_type (str): Type of model to load ('lstm', 'transformer', etc.)
        input_size (int): Input feature size (needed for some models)
        device (torch.device): Device to load the model on
        
    Returns:
        torch.nn.Module: The loaded model
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        # First try loading without weights_only for PyTorch 2.6+ compatibility
        try:
            print(f"Loading model with weights_only=False (safer for PyTorch 2.6+)...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            print("Successfully loaded checkpoint with weights_only=False")
        except (TypeError, ValueError) as e:
            # For older PyTorch versions that don't support weights_only
            if "got an unexpected keyword argument 'weights_only'" in str(e):
                print("weights_only parameter not supported, using standard loading")
                checkpoint = torch.load(model_path, map_location=device)
            else:
                raise e
        
        # Print checkpoint keys to debug
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Extract model parameters from checkpoint
        hidden_size = checkpoint.get('hidden_size', 64)
        num_layers = checkpoint.get('num_layers', 2)
        d_model = checkpoint.get('d_model', 64)
        
        # Try to get input_size from the checkpoint if not provided
        if input_size is None:
            # Different ways models might store input_size
            if 'input_size' in checkpoint:
                input_size = checkpoint['input_size']
            elif 'lstm.input_size' in checkpoint:
                input_size = checkpoint['lstm.input_size']
            elif 'lstm.weight_ih_l0' in checkpoint:
                # Extract from the weight shape if possible
                weight_shape = checkpoint['lstm.weight_ih_l0'].shape
                if len(weight_shape) >= 1:
                    input_size = weight_shape[1]  # Second dimension is usually input size
            else:
                # Default if we can't determine
                print("Warning: Could not determine input_size from checkpoint, using default value of 11")
                input_size = 11  # Common value for Dst forecasting (1 target + 10 cyclical features)
        
        print(f"Initializing {model_type} model with input_size={input_size}")
        
        # Initialize the appropriate model architecture
        if model_type == 'lstm':
            from enhanced_lstm_model import LSTMModel
            model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=0.3
            )
        elif model_type == 'transformer':
            model = TimeSeriesTransformer(
                input_size=input_size,
                d_model=d_model,
                nhead=4,
                num_layers=2,
                dropout=0.1
            )
        elif model_type == 'tcn':
            model = TemporalConvNet(
                input_size=input_size,
                hidden_channels=64,
                kernel_size=3,
                dropout=0.2,
                num_layers=4
            )
        elif model_type == 'nbeats':
            # Assuming we know the input dimension for NBeats
            input_dim = checkpoint.get('input_dim', 24 * input_size)  # seq_length * features
            model = NBeatsModel(
                input_dim=input_dim,
                hidden_dim=128,
                theta_dim=16,
                num_blocks=3,
                layers=4
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get the state dict from the checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint
        
        # Check if keys match
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        # Print any missing or extra keys for debugging
        missing_keys = model_keys - checkpoint_keys
        extra_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"Warning: Keys in model but not in checkpoint: {missing_keys}")
        if extra_keys:
            print(f"Warning: Keys in checkpoint but not in model: {extra_keys}")
            
            # Try to fix common prefix issues (e.g., 'module.' prefix from DataParallel)
            if any(k.startswith('module.') for k in state_dict):
                print("Removing 'module.' prefix from state dict keys...")
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Attempt to load state dict
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded state dict with strict=False")
        except Exception as e:
            print(f"Error loading state dict strictly: {str(e)}")
            print("Attempting to load with key matching...")
            
            # Create a new state dict that matches the model's keys
            new_state_dict = {}
            for model_key in model.state_dict().keys():
                # Look for a matching key in the checkpoint
                if model_key in state_dict:
                    new_state_dict[model_key] = state_dict[model_key]
                else:
                    print(f"No match found for key: {model_key}")
            
            # Try to load the matched keys
            if new_state_dict:
                model.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded {len(new_state_dict)}/{len(model.state_dict())} keys with matching")
            else:
                print("No keys could be matched between model and checkpoint")
        
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        print(f"Successfully loaded {model_type} model from {model_path}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_models(data_path, models=None, skip_models=None, load_models=None):
    """
    Train and compare multiple models on the same dataset
    
    Args:
        data_path (str): Path to the data file
        models (list): List of model types to compare, defaults to ['lstm', 'transformer', 'tcn', 'nbeats']
        skip_models (list): List of models to skip (already trained)
        load_models (dict): Dictionary mapping model_types to saved model paths to load instead of training
    """
    if models is None:
        models = ['lstm', 'transformer', 'tcn', 'nbeats']
    
    if skip_models is None:
        skip_models = []
    
    if load_models is None:
        load_models = {}
    
    # Determine which models to train, skip, or load
    models_to_train = []
    for model in models:
        if model in skip_models:
            print(f"Skipping {model} model (in skip_models list)")
        elif model in load_models:
            print(f"Will load pre-trained {model} model (in load_models dict)")
        else:
            models_to_train.append(model)
    
    print(f"Models to train: {models_to_train}")
    print(f"Models to load: {list(load_models.keys())}")
    
    # Early exit if no models to train or load
    if not models_to_train and not load_models:
        print("No models to train or load. All specified models have been skipped.")
        return None, None
    
    # Load data
    dst_data = read_dst_csv(data_path)
    segments = split_by_gaps(dst_data, gap_threshold='2H')
    longest_segment = max(segments, key=len)
    print(f"Using longest continuous segment with {len(longest_segment)} records")
    
    # Prepare data once for all models
    X_train, y_train, X_test, y_test, scaler, feature_cols = prepare_enhanced_data(
        longest_segment,
        target_col='Dst',
        seq_length=24,
        remove_outliers=True,
        outlier_method='iqr', 
        outlier_threshold=2.0,
        add_cycles=True,
        scale_method='robust'
    )
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Track results for each model
    results = {}
    
    # Load pre-trained models
    input_size = X_train.shape[2]  # [batch, seq_len, features]
    seq_length = X_train.shape[1]
    
    for model_type, model_path in load_models.items():
        print(f"\n{'='*50}")
        print(f"Loading pre-trained {model_type.upper()} model from {model_path}")
        print(f"{'='*50}")
        
        # Load the model
        model = load_trained_model(
            model_path=model_path, 
            model_type=model_type,
            input_size=input_size,
            device=device
        )
        
        if model is not None:
            # Evaluate the loaded model
            print(f"Evaluating loaded {model_type} model...")
            metrics, predictions = evaluate_model(
                model, X_test, y_test, scaler, longest_segment, device, model_type
            )
            
            # Store results
            results[model_type] = {
                'metrics': metrics,
                'predictions': predictions,
                'loaded': True,
                'model': model
            }
    
    # Create output directory
    import os
    output_dir = "/mnt/e/dst-forecasting/outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Train the models that need training
    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        input_size = X_train.shape[2]  # [batch, seq_len, features]
        seq_length = X_train.shape[1]
        
        # Initialize appropriate model
        if model_type == 'lstm':
            from enhanced_lstm_model import LSTMModel
            model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3)
        
        elif model_type == 'transformer':
            model = TimeSeriesTransformer(
                input_size=input_size,
                d_model=64,
                nhead=4,
                num_layers=2,
                dropout=0.1
            )
        
        elif model_type == 'tcn':
            model = TemporalConvNet(
                input_size=input_size,
                hidden_channels=64,
                kernel_size=3,
                dropout=0.2,
                num_layers=4
            )
        
        elif model_type == 'nbeats':
            model = NBeatsModel(
                input_dim=seq_length * input_size,
                hidden_dim=128,
                theta_dim=16,
                num_blocks=3,
                layers=4
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        trained_model, history, best_epoch = train_model(
            model, X_train, y_train, X_test, y_test,
            batch_size=32, 
            learning_rate=0.001,
            num_epochs=100,
            patience=20,
            device=device,
            model_type=model_type,
            model_save_path='model'  # Changed to just the model name, not full path
        )
        
        # Plot training history
        model_output_dir = f"{output_dir}/{model_type}"
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        plot_training_history(history, save_path=model_output_dir)
        
        # Evaluate model
        metrics, predictions = evaluate_model(trained_model, X_test, y_test, 
                                               scaler, longest_segment, device, model_type)
        
        # Save model
        save_model(
            trained_model, 
            f'/mnt/e/dst-forecasting/dst_{model_type}_model.pt',
            model_type, 
            feature_cols,
            scaler
        )
        
        # Store results
        results[model_type] = {
            'metrics': metrics,
            'predictions': predictions,
            'best_epoch': best_epoch,
            'history': history
        }
        
        # Save metrics to file for future reference
        metrics_file = f"{model_output_dir}/metrics.json"
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        print(f"Saved metrics to {metrics_file}")
    
    # Compare model performance
    print("\n\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    # Create a comparison table
    comparison_table = []
    for model_type, result in results.items():
        metrics = result['metrics']
        comparison_table.append({
            'Model': model_type.upper(),
            'MSE': metrics['mse'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'MAPE (%)': metrics['mape']
        })
    
    # Convert to DataFrame for nice display
    comparison_df = pd.DataFrame(comparison_table)
    print(comparison_df.to_string(index=False))
    
    # Plot predictions from all models on the same axis
    plt.figure(figsize=(14, 8))
    
    # Plot actual values - use first available model's predictions
    first_model = next(iter(results.keys()))
    y_actual = results[first_model]['predictions'][0]
    plt.plot(y_actual, 'k-', label='Actual', linewidth=2)
    
    # Plot predictions for each model
    for model_type, result in results.items():
        y_pred = result['predictions'][1]
        plt.plot(y_pred, '--', label=f'{model_type.upper()} Prediction', alpha=0.7)
    
    plt.title('Model Comparison: Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Dst Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300)
    plt.close()
    print(f"Model comparison plot saved to {output_dir}/model_comparison.png")
    
    return results, comparison_df

if __name__ == "__main__":
    # For debugging purposes, let's add a try-except block to catch and print errors
    try:
        # Load the pre-trained LSTM model and train the other models
        results, comparison_df = compare_models(
            data_path='/mnt/e/dst-forecasting/dst_data.csv',
            # Include all models you want to compare in the final results
            models=['lstm', 'transformer', 'tcn', 'nbeats'],  # Let's focus on just the TCN model for now
            skip_models=[],
            load_models={
                }
            # load_models={
            #     'lstm': '/mnt/e/dst-forecasting/checkpoints/model_lstm_best.pt',
            #     'transformer': '/mnt/e/dst-forecasting/checkpoints/model_transformer_best.pt',
            #     'tcn': '/mnt/e/dst-forecasting/checkpoints/model_tcn_best.pt'
            #     }
        )
        
        # Save comparison results if available
        if comparison_df is not None:
            comparison_df.to_csv('/mnt/e/dst-forecasting/model_comparison.csv', index=False)
            print("\nComparison results saved to '/mnt/e/dst-forecasting/model_comparison.csv'")
        else:
            print("\nNo comparison available - all models were skipped or no models were specified.")
            
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()