import logging
import os
import time

import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


class AlphaVantageClient(BaseAPIClient):
    def __init__(self, api_key: str | None = None):
        super().__init__(client_name="AlphaVantageClient")
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        if not self.api_key:
            logger.warning(
                f"[{self.client_name}] API key not provided or found "
                "in environment variable ALPHA_VANTAGE_API_KEY. Client may not function."
            )
        self.ts = TimeSeries(key=self.api_key, output_format="pandas")

    def fetch_ohlcv_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        if not self.api_key:
            logger.error(f"[{self.client_name}] API key is missing. Cannot fetch data.")
            return None
        try:
            logger.info(f"[{self.client_name}] Fetching daily adjusted data for {symbol}")
            data, meta_data = self.ts.get_daily_adjusted(symbol=symbol, outputsize="full")
            if data.empty:
                logger.warning(f"[{self.client_name}]No data returned by AlphaVantage for {symbol}")
                return None

            data.index = pd.to_datetime(data.index)

            data = data.sort_index(ascending=True)
            mask = (data.index >= pd.to_datetime(start_date)) & (
                data.index <= pd.to_datetime(end_date)
            )
            filtered_data = data.loc[mask]

            if filtered_data.empty:
                logger.warning(
                    f"[{self.client_name}] No data for {symbol} "
                    f"within the date range {start_date} to {end_date} "
                    "after filtering."
                )
                return None

            standardized_df = self._standardize_dataframe(filtered_data.copy(), symbol)

            if standardized_df is not None:
                logger.info(
                    f"[{self.client_name}] Successfully fetched "
                    f"and standardized {len(standardized_df)} "
                    f"records for {symbol}"
                )

            time.sleep(13)
            return standardized_df

        except ValueError as ve:
            logger.error(
                f"[{self.client_name}] ValueError fetching data "
                f"for {symbol} (possibly API limit or invalid symbol): {ve}",
            )
            if "call frequency" in str(ve).lower():
                logger.warning(f"[{self.client_name}] Hit API rate limit. Sleeping for 60s.")
                time.sleep(61)
            return None
        except Exception as e:
            logger.error(
                f"[{self.client_name}] Generic error fetching data for {symbol}: {e}",
                exc_info=True,
            )
            return None
