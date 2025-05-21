import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

logger = logging.getLogger(__name__)
load_dotenv()

FRED_API_KEY_ENV = os.getenv("FRED_API_KEY")


class FredApiClient:
    def __init__(self, api_key: str | None = None):
        self.client_name = "FredApiClient"
        self.api_key = api_key or FRED_API_KEY_ENV
        if not self.api_key:
            msg = (
                f"[{self.client_name}] FRED API key not provided or found "
                "in environment variable FRED_API_KEY. Client will not function."
            )
            logger.error(msg)
            raise ValueError(msg)
        try:
            self.fred = Fred(api_key=self.api_key)
        except Exception as e:
            logger.error(f"[{self.client_name}] Failed to initialize Fred connection: {e}")
            raise
        logger.info(f"{self.client_name} initialized.")

    def fetch_series_all_releases(
        self, series_id: str, retry_attempts: int = 3, retry_delay: int = 5
    ) -> pd.DataFrame | None:
        attempt = 0
        while attempt < retry_attempts:
            try:
                logger.info(
                    f"[{self.client_name}] Fetching all releases for series {series_id} "
                    f"(Attempt {attempt + 1}/{retry_attempts})"
                )
                data_series = self.fred.get_series_all_releases(series_id)

                if data_series.empty:
                    logger.warning(
                        f"[{self.client_name}] No data returned by FRED for series {series_id}"
                    )
                    return None

                df = data_series.reset_index()
                df.rename(
                    columns={
                        "realtime_start": "release_date",
                        "date": "reference_date",
                        0: "value",
                    },
                    inplace=True,
                )
                if "value" not in df.columns and 0 in df.columns:
                    df.rename(columns={0: "value"}, inplace=True)

                df["release_date"] = pd.to_datetime(df["release_date"]).dt.date
                df["reference_date"] = pd.to_datetime(df["reference_date"]).dt.date
                df["value"] = pd.to_numeric(df["value"], errors="coerce")

                df.dropna(subset=["value"], inplace=True)
                df["series_id"] = series_id

                logger.info(
                    f"[{self.client_name}] Successfully fetched {len(df)} "
                    f"vintage records for series {series_id}"
                )
                return df[["reference_date", "release_date", "value", "series_id"]]

            except Exception as e:
                logger.error(
                    f"[{self.client_name}] Error fetching data for series {series_id} "
                    f"(Attempt {attempt + 1}): {e}",
                )
                attempt += 1
                if attempt < retry_attempts:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to fetch {series_id} after {retry_attempts} attempts.")
                    return None
        return None
