# Dst Forecasting System

A machine learning system for forecasting Disturbance Storm Time (Dst) index values, which are critical for monitoring and predicting geomagnetic storm activity.

## Overview

This system uses multiple machine learning models (LSTM, Transformer, TCN, N-BEATS) to forecast Dst values one hour into the future based on historical geomagnetic data. It can download the latest data or use existing historical data for predictions.

## Project Idea & Motivation

This project was developed to improve geomagnetic storm prediction capabilities through machine learning approaches. My key ideas include:

- **Multi-Model Approach**: Instead of relying on a single model architecture, I implemented multiple models to compare their performance and develop an ensemble approach.
- **Time Series Features**: Advanced temporal feature engineering to capture cyclical patterns in geomagnetic data at different time scales.
- **Real-Time Forecasting**: Creating a system that can automatically download the latest data and generate forecasts with minimal latency.
- **Ensemble Prediction**: Combining the strengths of different model architectures to produce more reliable forecasts.
- **Interpretability**: Visualizations and metrics that help understand model performance and capture meaningful patterns in the data.

The goal is to provide a tool that can assist researchers and operators in predicting geomagnetic disturbances that may impact technological systems on Earth.

## Features

- **Multiple Models**: Supports LSTM, Transformer, TCN, and N-BEATS architectures
- **Ensemble Forecasting**: Combines predictions from multiple models
- **Real-time Updates**: Can download the latest Dst data automatically
- **Visualization**: Generates plots of historical data and forecasts
- **Evaluation**: Compares model performance using standard metrics

## Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8 or higher
- NumPy, Pandas, Matplotlib
- scikit-learn

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/dst-forecasting.git
   cd dst-forecasting
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download pre-trained models:
   - Download the checkpoint file from [Google Drive](https://drive.google.com/file/d/1lxDw5fR2DcVKoygFN9sG0DZL06ty_t02/view?usp=sharing)
   - Place the downloaded file in the `checkpoints/` directory

4. Or train your own models using the provided scripts.

## Usage

### Basic Forecasting

Run the forecasting script with default parameters:

```bash
python forecast_dst.py
```

This will:
1. Load the pre-trained models from the default locations
2. Use historical data from the default path
3. Generate a forecast for the next hour
4. Display results in the console and save a visualization

### Advanced Options

```bash
python forecast_dst.py --models lstm,transformer --download --data /path/to/custom/data.csv
```

#### Command Line Arguments

- `--lstm`: Path to trained LSTM model (default: `/mnt/e/dst-forecasting/checkpoints/dst_lstm_pytorch.pt`)
- `--transformer`: Path to trained Transformer model
- `--tcn`: Path to trained TCN model
- `--nbeats`: Path to trained N-BEATS model
- `--data`: Path to historical data CSV (optional, will download if not provided)
- `--download`: Force download latest data even if historical data is provided
- `--models`: Comma-separated list of models to use (default: all)
- `--no-scaling`: Disable data scaling even if no scaler is found
- `--full-data`: Path to full historical data for creating scaler

### Training Models

To train or retrain the models:

```bash
python transformer_model.py
```

This will train all model types (LSTM, Transformer, TCN, N-BEATS) using the dataset specified in the script.

## File Structure

- `forecast_dst.py`: Main forecasting script
- `enhanced_lstm_model.py`: LSTM model implementation and training
- `transformer_model.py`: Implementation of all models (Transformer, TCN, N-BEATS)
- `read_dst_data.py`: Utilities for reading and processing Dst data
- `analyze_dst_data.py`: Data analysis utilities
- `geomag.py`: Functions for downloading geomagnetic data
- `checkpoints/`: Pre-trained model files
- `forecasts/`: Output directory for forecast visualizations

## Model Performance

The system evaluates models using the following metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

## Example Output

![Dst Forecast Example](/mnt/e/dst-forecasting/forecasts/example_forecast.png)

The above image shows a sample forecast with historical data (black line) and predictions from different models. The vertical dashed line separates historical data from forecasts.

## License

[Include license information here]

## Contributors

[List contributors here]

## Acknowledgments

- World Data Center for Geomagnetism, Kyoto for providing the Dst index data
- [Other acknowledgments]