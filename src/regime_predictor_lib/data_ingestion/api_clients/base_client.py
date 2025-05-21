import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class BaseAPIClient(ABC):
    def __init__(self, client_name: str):
        self.client_name = client_name
        logger.info(f"{self.client_name} initialized.")

    @abstractmethod
    def fetch_ohlcv_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        pass

    def _standardize_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
        if df is None or df.empty:
            logger.warning(
                f"[{self.client_name}] Input DataFrame "
                f"for {symbol} is None or empty. Cannot standardize."
            )
            return None

        df_copy = df.copy()

        if isinstance(df_copy.columns, pd.MultiIndex):
            logger.debug(
                f"[{self.client_name}] DataFrame "
                f"for {symbol} has MultiIndex columns. Flattening."
            )
            new_cols = []
            for col_tuple in df_copy.columns:
                if isinstance(col_tuple, tuple):
                    new_cols.append(col_tuple[0])
                else:
                    new_cols.append(col_tuple)
            df_copy.columns = new_cols
            logger.debug(
                f"[{self.client_name}] Columns after attempting "
                f"to flatten MultiIndex: {df_copy.columns.tolist()}"
            )

        if isinstance(df_copy.index, pd.DatetimeIndex):
            index_name = df_copy.index.name
            df_copy = df_copy.reset_index()
            if index_name and index_name not in df_copy.columns:
                pass
            elif "index" in df_copy.columns and df_copy["index"].dtype == "datetime64[ns]":
                if "Date" not in df_copy.columns and "date" not in df_copy.columns:
                    df_copy.rename(columns={"index": "Date"}, inplace=True)
            logger.debug(
                f"[{self.client_name}] DataFrame index reset "
                f"for {symbol}. Columns: {df_copy.columns.tolist()}"
            )

        rename_map = {
            "Date": "date",
            "Timestamp": "date",
            "datetime": "date",
            "Open": "open",
            "1. open": "open",
            "High": "high",
            "2. high": "high",
            "Low": "low",
            "3. low": "low",
            "Close": "close",
            "4. close": "close",
            "Adj Close": "adjusted_close",
            "Adjusted Close": "adjusted_close",
            "5. adjusted close": "adjusted_close",
            "Volume": "volume",
            "6. volume": "volume",
        }

        cols_to_rename = {k: v for k, v in rename_map.items() if k in df_copy.columns}
        if cols_to_rename:
            df_copy.rename(columns=cols_to_rename, inplace=True)
        logger.debug(
            f"[{self.client_name}] Columns after "
            f"renaming for {symbol}: {df_copy.columns.tolist()}"
        )

        if "date" not in df_copy.columns:
            logger.error(
                f"[{self.client_name}] 'date' column missing "
                f"after standardization for {symbol}. "
                f"Columns: {df_copy.columns.tolist()}"
            )
            return None
        try:
            df_copy["date"] = pd.to_datetime(df_copy["date"])
        except Exception as e:
            logger.error(
                f"[{self.client_name}] Could not convert 'date' "
                f"column to datetime for {symbol}: {e}. "
                f"Date column head: \n{df_copy['date'].head().to_string()}"
            )
            return None

        if "adjusted_close" not in df_copy.columns and "close" in df_copy.columns:
            df_copy["adjusted_close"] = df_copy["close"]
            logger.debug(
                f"[{self.client_name}] Used 'close' as fallback for 'adjusted_close' for {symbol}."
            )
        elif "adjusted_close" not in df_copy.columns and "close" not in df_copy.columns:
            logger.error(
                f"[{self.client_name}] Neither 'adjusted_close' nor "
                f"'close' found for {symbol}. Columns: {df_copy.columns.tolist()}"
            )
            return None

        df_copy["ticker"] = symbol

        standard_final_columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "ticker",
        ]

        missing_cols = [col for col in standard_final_columns if col not in df_copy.columns]
        if missing_cols:
            logger.error(
                f"[{self.client_name}] Missing required "
                f"columns {missing_cols} for {symbol} "
                "before final selection. "
                f"Available columns: {df_copy.columns.tolist()}"
            )
            return None

        df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()]

        logger.debug(
            f"[{self.client_name}] Final columns for {symbol} "
            f"before selection: {df_copy.columns.tolist()}"
        )
        return df_copy[standard_final_columns]
