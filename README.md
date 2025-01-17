# Geomagnetic Data Prediction System

This system fetches, processes, and predicts geomagnetic disturbance (Dst) values using real-time data from the World Data Center for Geomagnetism, Kyoto.

## Features

- Real-time Dst data fetching and processing
- Historical data analysis
- Predictive modeling with:
  - Base model (Prophet-based fallback)
  - Support for custom model integration
- Automated hourly predictions
- Ground truth validation

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main prediction service:
```bash
python geomag.py
```

The system will:
- Sleep until the next hour
- Fetch latest geomagnetic data
- Process historical values
- Make predictions for the next hour
- Validate against ground truth when available

## Custom Model Integration

We encourage contributors to develop custom models that improve upon the base model's performance. To integrate your custom model:

1. Follow the implementation guide in [CUSTOMMODELS.md](CUSTOMMODELS.md)
2. Ensure your model follows the required naming conventions and interfaces
3. Test against the base model performance

See [CUSTOMMODELS.md](CUSTOMMODELS.md) for detailed specifications and examples.

## Components

- `geomag.py`: Main script handling data fetching and processing
- `geomag_basemodel.py`: Base model implementation and model loading logic
- `CUSTOMMODELS.md`: Guide for implementing custom prediction models

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt
- Internet connection for data fetching

## Data Source

Real-time Dst data is sourced from the [World Data Center for Geomagnetism, Kyoto](https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/).