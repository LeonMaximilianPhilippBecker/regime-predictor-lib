import logging
import sys
from pathlib import Path

import pandas as pd

from regime_predictor_lib.data_ingestion.api_clients.fred_client import FredApiClient

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FRED_SERIES_TO_FETCH_EM_TBILL = {
    "BAMLEMCBPIEY": "fred_BAMLEMCBPIEY_raw_vintage.csv",
    "DTB3": "fred_DTB3_raw_vintage.csv",
}

if __name__ == "__main__":
    RAW_DATA_DIR = PROJECT_ROOT_PATH / "data" / "raw"
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        fred_client = FredApiClient()
    except ValueError as e:
        logger.error(f"Failed to initialize FredApiClient: {e}. Ensure FRED_API_KEY is set.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred initializing FredApiClient: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Starting raw FRED EM & T-Bill yield data fetching process...")

    for series_id, filename in FRED_SERIES_TO_FETCH_EM_TBILL.items():
        output_path = RAW_DATA_DIR / filename
        logger.info(f"Fetching all releases for series {series_id}...")
        try:
            data_df = fred_client.fetch_series_all_releases(series_id)
            if data_df is not None and not data_df.empty:
                data_df.to_csv(output_path, index=False)
                logger.info(
                    f"Successfully fetched and saved {len(data_df)} "
                    f"records for {series_id} to {output_path}"
                )
            elif data_df is not None and data_df.empty:
                logger.warning(
                    f"No data returned by FRED for series {series_id}. "
                    f"Empty CSV created at {output_path}"
                )
                pd.DataFrame(
                    columns=["reference_date", "release_date", "value", "series_id"]
                ).to_csv(output_path, index=False)
            else:
                logger.error(f"Failed to fetch data for series {series_id}. CSV not created.")

        except Exception as e:
            logger.error(
                f"Error fetching or saving data for series {series_id}: {e}", exc_info=True
            )

    logger.info("Raw FRED EM & T-Bill yield data fetching script finished.")
