import datetime
from datetime import datetime
import traceback
import httpx
import pytz
import asyncio
from fiber.logging_utils import get_logger


logger = get_logger(__name__)

def get_url(year, month):
    if(year == 2024 and month > 6 or year > 2024):
        return f"https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{year}{month:02d}/dst{str(year)[-2:]}{month:02d}.for.request"
    elif(year>=2021 and year <=2024):
        return f"https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year}{month:02d}/dst{str(year)[-2:]}{month:02d}.for.request"
    else:
        return f"https://wdc.kugi.kyoto-u.ac.jp/dst_final/{year}{month:02d}/dst{str(year)[-2:]}{month:02d}.for.request"
    
async def fetch_yearly_data(years=[], max_retries=3):
    """
    Fetch raw geomagnetic data for the entire year from dynamically generated URLs.

    Args:
        max_retries (int): Maximum number of retry attempts.

    Returns:
        str: The raw data as a text string.
    """
    current_time = datetime.now(pytz.UTC)
    current_year = current_time.year
    if(len(years) < 1):
        years = [current_year]

    years = sorted(years)
    # Generate URLs for all months of the current year
    urls = [
        get_url(year, month)
        for year in years
        for month in range(1, 13)
    ]

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Fetch all URLs concurrently
                responses = await asyncio.gather(
                    *[client.get(url) for url in urls], return_exceptions=True
                )

            # Process responses
            raw_data = []
            for i, response in enumerate(responses):
                if isinstance(response, httpx.Response):
                    try:
                        response.raise_for_status()  # Ensure the response is valid
                        raw_data.append(response.text)
                    except httpx.HTTPStatusError as e:
                        logger.error(
                            f"HTTP error for {urls[i]}: {e.response.status_code}"
                        )
                else:
                    logger.error(f"Failed to fetch {urls[i]}: {response}")

            # Combine all data
            return "\n".join(raw_data)

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