import logging

import numpy as np
import pandas as pd

from regime_predictor_lib.data_ingestion.api_clients import YFinanceClient
from regime_predictor_lib.utils.financial_calculations import calculate_z_score

logger = logging.getLogger(__name__)

ANNUALIZATION_FACTOR = 252
ZSCORE_WINDOW = 63

RETURN_PERIODS = {"1d": 1, "5d": 5, "21d": 21, "63d": 63, "126d": 126}
VOL_PERIODS = {"1d": 1, "5d": 5, "21d": 21, "63d": 63, "126d": 126}


class SP500DerivedIndicatorCalculator:
    def __init__(self, yf_client: YFinanceClient):
        self.yf_client = yf_client
        logger.info("SP500DerivedIndicatorCalculator initialized.")

    def _fetch_sp500_data(
        self, start_date_str: str, end_date_str: str | None, ticker: str = "^GSPC"
    ) -> pd.DataFrame | None:
        if end_date_str is None:
            end_date_str = pd.Timestamp.now().strftime("%Y-%m-%d")

        max_lookback = max(max(VOL_PERIODS.values()), max(RETURN_PERIODS.values()), ZSCORE_WINDOW)
        buffer_start_date = (
            pd.to_datetime(start_date_str) - pd.DateOffset(days=max_lookback * 2)
        ).strftime("%Y-%m-%d")

        logger.info(f"Fetching S&P 500 data ({ticker}) from {buffer_start_date} to {end_date_str}")
        df_raw = self.yf_client.fetch_ohlcv_data(ticker, buffer_start_date, end_date_str)

        if df_raw is None or df_raw.empty:
            logger.warning(f"Could not fetch S&P 500 data for {ticker}.")
            return None

        df = df_raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df

    def calculate_indicators(
        self, start_date: str = "1985-01-01", end_date: str | None = None
    ) -> pd.DataFrame | None:
        ohlcv_df = self._fetch_sp500_data(start_date, end_date)
        if ohlcv_df is None or ohlcv_df.empty:
            return None

        indicators_df = pd.DataFrame(index=ohlcv_df.index)

        for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
            if col in ohlcv_df.columns:
                indicators_df[f"sp500_{col}"] = ohlcv_df[col]
            else:
                logger.warning(f"Column {col} not found in fetched S&P 500 data.")
                indicators_df[f"sp500_{col}"] = np.nan

        adj_close = ohlcv_df["adjusted_close"]

        for period_name, period_days in RETURN_PERIODS.items():
            indicators_df[f"ret_{period_name}"] = adj_close.pct_change(periods=period_days)

        log_returns = np.log(adj_close / adj_close.shift(1))

        for period_name, period_days in VOL_PERIODS.items():
            col_name = f"log_vol_{period_name}"
            if period_name == "1d":
                indicators_df[col_name] = log_returns.abs() * np.sqrt(ANNUALIZATION_FACTOR)
            else:
                min_p = max(2, period_days // 2)
                indicators_df[col_name] = log_returns.rolling(
                    window=period_days, min_periods=min_p
                ).std(ddof=0) * np.sqrt(ANNUALIZATION_FACTOR)

        for period_name in RETURN_PERIODS:
            ret_col = f"ret_{period_name}"
            zscore_col = f"ret_{period_name}_zscore_{ZSCORE_WINDOW}d"
            if ret_col in indicators_df:
                indicators_df[zscore_col] = calculate_z_score(
                    indicators_df[ret_col], window=ZSCORE_WINDOW
                )
            else:
                indicators_df[zscore_col] = np.nan

        for period_name in VOL_PERIODS:
            vol_col = f"log_vol_{period_name}"
            zscore_col = f"log_vol_{period_name}_zscore_{ZSCORE_WINDOW}d"
            if vol_col in indicators_df:
                indicators_df[zscore_col] = calculate_z_score(
                    indicators_df[vol_col], window=ZSCORE_WINDOW
                )
            else:
                indicators_df[zscore_col] = np.nan

        indicators_df = indicators_df[indicators_df.index >= pd.to_datetime(start_date)]

        indicators_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        indicator_columns = [
            col for col in indicators_df.columns if col.startswith(("ret_", "log_vol_"))
        ]
        indicators_df.dropna(subset=indicator_columns, how="all", inplace=True)

        indicators_df.reset_index(inplace=True)
        indicators_df["date"] = indicators_df["date"].dt.strftime("%Y-%m-%d")

        expected_cols = [
            "date",
            "sp500_open",
            "sp500_high",
            "sp500_low",
            "sp500_close",
            "sp500_adjusted_close",
            "sp500_volume",
            "ret_1d",
            "ret_5d",
            "ret_21d",
            "ret_63d",
            "ret_126d",
            "log_vol_1d",
            "log_vol_5d",
            "log_vol_21d",
            "log_vol_63d",
            "log_vol_126d",
            "ret_1d_zscore_63d",
            "ret_5d_zscore_63d",
            "ret_21d_zscore_63d",
            "ret_63d_zscore_63d",
            "ret_126d_zscore_63d",
            "log_vol_1d_zscore_63d",
            "log_vol_5d_zscore_63d",
            "log_vol_21d_zscore_63d",
            "log_vol_63d_zscore_63d",
            "log_vol_126d_zscore_63d",
        ]
        for col in expected_cols:
            if col not in indicators_df.columns:
                indicators_df[col] = np.nan

        final_df = indicators_df[expected_cols]

        logger.info(f"Calculated S&P 500 derived indicators. Shape: {final_df.shape}")
        return final_df
