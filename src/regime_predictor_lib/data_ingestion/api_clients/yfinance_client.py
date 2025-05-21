import logging

import pandas as pd
import yfinance as yf

from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class YFinanceClient(BaseAPIClient):
    def __init__(self):
        super().__init__(client_name="YFinanceClient")

    def fetch_ohlcv_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        try:
            logger.info(
                f"[{self.client_name}] Fetching OHLCV data for {symbol} "
                f"from {start_date} to {end_date}"
            )
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if data.empty:
                logger.warning(
                    f"[{self.client_name}] No data returned by yfinance "
                    f"for {symbol} from {start_date} to {end_date}"
                )
                return None

            standardized_df = self._standardize_dataframe(data.copy(), symbol)

            if standardized_df is not None:
                logger.info(
                    f"[{self.client_name}] Successfully fetched and "
                    f"standardized {len(standardized_df)} records for {symbol}"
                )
            return standardized_df

        except Exception as e:
            logger.error(
                f"[{self.client_name}] Error fetching data for {symbol}: {e}",
                exc_info=True,
            )
            return None
