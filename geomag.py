import datetime
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import traceback
import httpx
import pytz
import asyncio
from geomag_basemodel import GeoMagBaseModel, FallbackGeoMagModel
from fiber.logging_utils import get_logger
from fetch import fetch_yearly_data
from sklearn.preprocessing import RobustScaler

logger = get_logger(__name__)

PLACEHOLDER_VALUE = "999999999999999" 
TIME_RANGE=24*30 # 30 days (numner of sample to feed to the model) | 'None' will feed all data available
STEP_BACK=20 # predict STEP_BACK in the past and compare to true value (setting 0  will predict lastest value)
YEARS = range(2023, 2026) # list of year to fetch data

async def main():
    current_time = datetime.now(timezone.utc)
    next_hour = current_time.replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=0)
    sleep_duration = (next_hour - current_time).total_seconds()

    logger.info(
        f"Sleeping until the next top of the hour: {next_hour.isoformat()} (in {sleep_duration} seconds)"
    )
    await asyncio.sleep(sleep_duration)

    logger.info("Starting GeomagneticTask execution...")

    # Step 2: Fetch Latest Geomagnetic Data
    timestamp, dst_value, historical_data = await _fetch_geomag_data(years=YEARS)
    historical_records = []

    if historical_data is not None:
        for _, row in historical_data.iterrows():
            historical_records.append(
                {"timestamp": row["timestamp"].isoformat(), "Dst": row["Dst"]}
            )

    data = {
        "data": {
            "name": "Geomagnetic Data",
            "timestamp": timestamp.isoformat(),
            "value": dst_value,
            "historical_values": historical_records,
        },
    }
    
    if data and data.get("data"):
        # Process current data
        input_data = pd.DataFrame(
            {
                "timestamp": [pd.to_datetime(data["data"]["timestamp"])],
                "value": [float(data["data"]["value"])],
            }
        )

        # Check and process historical data if available
        if data["data"].get("historical_values"):
            historical_df = pd.DataFrame(data["data"]["historical_values"])
            historical_df = historical_df.rename(
                columns={"Dst": "value"}
            )  # Rename Dst to value
            historical_df["timestamp"] = pd.to_datetime(
                historical_df["timestamp"]
            )
            historical_df = historical_df[
                ["timestamp", "value"]
            ]  # Ensure correct columns
            combined_df = pd.concat(
                [historical_df], ignore_index=True
            )
        else:
            combined_df = input_data

        # Preprocess combined data
        processed_data, target_timestamp, target_value  = process_data(combined_df, time_range=TIME_RANGE, step_back=STEP_BACK)

        print(f"processed_data: {processed_data}")

        base_raw_prediction = run_model_inference(processed_data)
        raw_predictions = {
            "predicted_value": float(base_raw_prediction),
            "prediction_time": target_timestamp
        }
        print(f"Base raw prediction: {raw_predictions}")

        await _process_scores(float(base_raw_prediction), target_value, target_timestamp)
    else:
        print("No data provided in request")
        return None
    
async def _process_scores(value, ground_truth_value, target_timestamp):
    """Process and archive scores for the previous hour."""
    print(f"Ground truth value: {ground_truth_value}")
    print(f"Ground truth timestamp: {target_timestamp}")
    print(f"Predicted raw value: {value}")
    print(f"Predicted value: {value*100}")
    # print(f"Predicted value: {reverse_log_transform_value(value)}")

async def fetch_ground_truth():
    """
    Fetches the ground truth DST value for the current UTC hour.

    Returns:
        int: The real-time DST value, or None if fetching fails.
    """
    try:
        # Get the current UTC time
        current_time = datetime.now(timezone.utc)
        logger.info(f"Fetching ground truth for UTC hour: {current_time.hour}")

        # Fetch the most recent geomagnetic data
        timestamp, dst_value = await get_latest_geomag_data(
            include_historical=False
        )

        if timestamp == "N/A" or dst_value == "N/A":
            logger.warning("No ground truth data available for the current hour.")
            return None

        logger.info(f"Ground truth value for hour {current_time.hour}: {dst_value}")
        return dst_value

    except Exception as e:
        logger.error(f"Error fetching ground truth: {e}")
        logger.error(f"{traceback.format_exc()}")
        return None
    
async def _fetch_geomag_data(years=[datetime.now(pytz.UTC).year]):
    timestamp, dst_value, historical_data = await get_latest_geomag_data(
        years=years,
        include_historical=True
    )
    print(
        f"Fetched latest geomagnetic data: timestamp={timestamp}, value={dst_value}"
    )
    if historical_data is not None:
        print(
            f"Fetched historical data for the current month: {len(historical_data)} records"
        )
    else:
        print("No historical data available for the current month.")
    return timestamp, dst_value, historical_data

def run_model_inference(processed_data):
    """
    Run the GeoMag model inference.

    Args:
        processed_data (pd.DataFrame): Preprocessed input data for the model.

    Returns:
        float: Predicted value.
    """
    try:
        model = FallbackGeoMagModel()
        # model = GeoMagBaseModel()
        # Perform prediction using the model
        prediction = model.predict(processed_data)
        # Handle NaN or infinite values
        if np.isnan(prediction) or np.isinf(prediction):
            print("Model returned NaN/Inf, using fallback value")
            return float(
                processed_data["value"].iloc[-1]
            )  # Use input value as fallback

        return float(prediction)  # Ensure we return a Python float

    except Exception as e:
        print(f"Error during model inference: {e}")
        return float(
            processed_data["value"].iloc[-1]
        )  # Return input value as fallback

def process_data(data: pd.DataFrame, time_range:int|None = 100, step_back:int=0) -> pd.DataFrame:
    # Validate
    if time_range is not None:
        time_range = min(time_range, len(data))
    else:
        time_range = len(data)
    step_back = max(0, step_back)
    step_back = min(time_range, step_back)

    try:
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Create a copy and convert timestamp to naive UTC datetime
        processed_df = data.copy()
        # Convert timestamps to pandas datetime and remove timezone
        processed_df["timestamp"] = pd.to_datetime(processed_df["timestamp"])
        if processed_df["timestamp"].dt.tz is not None:
            processed_df["timestamp"] = (
                processed_df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
            )
        # Get current values (now in naive UTC)
        current_timestamp = processed_df["timestamp"].iloc[-1-step_back]
        target_value = processed_df["value"].iloc[-1-step_back+1] if step_back != 0 else None
        target_timestamp = current_timestamp + pd.Timedelta(hours=1)

        # Create historical points using proper pandas datetime handling
        if(step_back != 0):
            values = processed_df["value"].tail(time_range+step_back)[-(time_range+step_back):-step_back]
        else:
            values = processed_df["value"].tail(time_range)
        
        historical_data = pd.DataFrame(
            {
                "timestamp": [
                    current_timestamp - pd.Timedelta(hours=i) for i in range(time_range-1, -1, -1)
                ],
                "value": values,
            }
        )

        # Normalize values
        # historical_data["value"] = historical_data["value"] / 100.0
        # historical_data = diff_transform(historical_data)

        historical_data = normalize(historical_data)
        # historical_data = log_transform(historical_data)

        # historical_data, min_val, max_val = min_max_scale(historical_data)
        # Rename columns to match Prophet requirements
        historical_data = historical_data.rename(
            columns={"timestamp": "ds", "value": "y"}
        )

        # Sort by timestamp
        historical_data = historical_data.sort_values("ds").reset_index(drop=True)
        return historical_data, target_timestamp, target_value

    except Exception as e:
        print(f"Error in process_miner_data: {e}")
        raise


async def get_latest_geomag_data(years=[datetime.now(pytz.UTC).year], include_historical=False):
    """
    Fetch, parse, clean, and return the latest valid geomagnetic data point.

    Args:
        include_historical (bool): Whether to include current month's historical data.

    Returns:
        tuple: (timestamp, Dst value, historical_data) of the latest geomagnetic data point.
               `historical_data` will be a DataFrame if `include_historical=True`, otherwise None.
    """
    try:
        # Fetch raw data
        # raw_data = await fetch_data()
        raw_data = await fetch_yearly_data(years=years)
        # Parse and clean raw data into DataFrame
        parsed_df = parse_data(raw_data)
        cleaned_df = clean_data(parsed_df)

        # Extract the latest data point
        if not cleaned_df.empty:
            latest_data_point = cleaned_df.iloc[-1]
            timestamp = latest_data_point["timestamp"]
            dst_value = int(
                latest_data_point["Dst"]
            )  # Convert to native int for JSON compatibility
        else:
            # If no valid data available
            return "N/A", "N/A", None

        # If historical data is requested, filter the DataFrame for the current month
        historical_data = None
        if include_historical:
            now = datetime.now(timezone.utc)
            start_of_month = now.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            # historical_data = cleaned_df[cleaned_df["timestamp"] >= start_of_month]
            historical_data = cleaned_df
            return timestamp, dst_value, historical_data
        return timestamp, dst_value
    except Exception as e:
        logger.error(f"Error fetching geomagnetic data: {e}")
        logger.error(f"{traceback.format_exc()}")
        return "N/A", "N/A", None
    
def parse_data(data):
    dates = []
    hourly_values = []

    def parse_line(line):
        try:
            # Extract year, month, and day
            year = int("20" + line[3:5])  # Prefix with "20" for full year
            month = int(line[5:7])
            day = int(line[8:10].strip())
        except ValueError:
            print(f"Skipping line due to invalid date format: {line}")
            return

        # Iterate over 24 hourly values
        for hour in range(24):
            start_idx = 20 + (hour * 4)
            end_idx = start_idx + 4
            value_str = line[start_idx:end_idx].strip()

            # Skip placeholder and invalid values
            if value_str != PLACEHOLDER_VALUE and value_str:
                try:
                    value = int(value_str)
                    timestamp = datetime(year, month, day, hour, tzinfo=timezone.utc)

                    # Only include valid timestamps and exclude future timestamps
                    if timestamp < datetime.now(timezone.utc):
                        dates.append(timestamp)
                        hourly_values.append(value)
                except ValueError:
                    print(f"Skipping invalid value: {value_str}")

    # Parse all lines that start with "DST"
    for line in data.splitlines():
        if line.startswith("DST"):
            parse_line(line)

    # Create a DataFrame with parsed data
    return pd.DataFrame({"timestamp": dates, "Dst": hourly_values})


def clean_data(df):
    now = datetime.now(timezone.utc)

    # Drop duplicate timestamps
    df = df.drop_duplicates(subset="timestamp")

    # Filter valid Dst range
    df = df[df["Dst"].between(-500, 500)]

    # Exclude future timestamps (ensure strictly less than current time)
    df = df[df["timestamp"] < now]

    # Reset index
    return df.reset_index(drop=True)

async def fetch_data(url=None, max_retries=3):
    """
    Fetch raw geomagnetic data from the specified or dynamically generated URL.

    Args:
        url (str, optional): The URL to fetch data from. If not provided, a URL will be generated
                             based on the current year and month.
        max_retries (int): Maximum number of retry attempts

    Returns:
        str: The raw data as a text string.
    """
    # Generate the default URL based on the current year and month if not provided
    if url is None:
        current_time = datetime.now(pytz.UTC)
        current_year = current_time.year
        current_month = current_time.month
        # Get last month's time
        if current_time.month == 1:  # If it's January
            last_month = current_time.replace(day=1).replace(year=current_time.year - 1, month=12)
        else:
            last_month = current_time.replace(day=1).replace(month=current_time.month - 1)

        # Format the URL dynamically
        url = f"https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{current_year}{current_month:02d}/dst{str(current_year)[-2:]}{current_month:02d}.for.request"
        last_month_url = f"https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{last_month.year}{last_month.month:02d}/dst{str(last_month.year)[-2:]}{last_month.month:02d}.for.request"
    logger.info(f"Fetching data from URL: {url}")

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                current_month_response, last_month_response = await asyncio.gather(
                client.get(url),
                client.get(last_month_url),
            )

            current_month_response.raise_for_status()
            last_month_response.raise_for_status()
            return f"{last_month_response.text}\n{current_month_response.text}"

        except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f"Failed to fetch data after {max_retries} attempts: {e}")
                raise RuntimeError(
                    f"Error fetching data after {max_retries} retries: {e}"
                )
            else:
                wait_time = (attempt + 1) * 5  # Exponential backoff
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            logger.error(f"{traceback.format_exc()}")
            raise e
        
def calculate_score(predicted_value, actual_value):
    """
    Calculates the score for a miner's prediction based on deviation.

    Args:
        predicted_value (float): The predicted DST value from the miner.
        actual_value (int): The ground truth DST value.

    Returns:
        float: The absolute deviation between the predicted value and the ground truth.
    """
    if not isinstance(predicted_value, (int, float)):
        raise ValueError("Predicted value must be an integer or a float.")
    if not isinstance(actual_value, int):
        raise ValueError("Actual value must be an integer.")

    # Calculate the absolute deviation
    score = abs(predicted_value - actual_value)
    return score

def diff_transform(df: pd.DataFrame):
    df['value'] = df['value'].diff().dropna()
    return df

def robust_scale(df: pd.DataFrame):
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    df['value'] = scaler.fit_transform(df[['value']])
    return df

def log_transform(df: pd.DataFrame):
    df['value'] = df['value'] + 400
    df['value'] = np.log1p(df['value'])
    return df

def reverse_log_transform(df: pd.DataFrame):
    df['value'] = np.expm1(df['value']) - 400
    return df

def reverse_log_transform_value(value):
    value = np.expm1(value) - 400
    return value

def normalize(df: pd.DataFrame):
    df['value'] = df['value']/100
    return df

def min_max_scale(df: pd.DataFrame, column: str='value'):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df, min_val, max_val

def reverse_min_max_scale(df: pd.DataFrame, min_val: float, max_val: float, column: str='value'):
    df[column] = df[column] * (max_val - min_val) + min_val
    return df

def reverse_min_max_scale_value(value, min_val: float, max_val: float):
    value = value * (max_val - min_val) + min_val
    return value

if __name__ == "__main__":
    asyncio.run(main())
