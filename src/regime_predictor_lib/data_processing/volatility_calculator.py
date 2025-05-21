import logging

import numpy as np
import pandas as pd

from regime_predictor_lib.data_ingestion.api_clients import YFinanceClient

logger = logging.getLogger(__name__)


class VolatilityCalculator:
    def __init__(self, yf_client: YFinanceClient):
        self.yf_client = yf_client
        self.vix_sma_windows = [20, 50]
        logger.info("VolatilityCalculator initialized.")

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> pd.Series:
        if not all(isinstance(s, pd.Series) for s in [high, low, close]):
            raise TypeError("Inputs (high, low, close) must be pandas Series.")
        if not (len(high) == len(low) == len(close)):
            raise ValueError("Input Series must have the same length.")
        if high.empty:
            return pd.Series(dtype=float)

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)

        atr = true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        return atr.rename(f"atr_{window}")

    def _calculate_sma(self, series: pd.Series, window: int) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series.")
        if series.empty:
            return pd.Series(dtype=float)
        return series.rolling(window=window, min_periods=window).mean()

    def fetch_and_calculate_volatility_metrics(
        self,
        start_date: str,
        end_date: str,
        sp500_ticker: str = "^GSPC",
        vix_ticker: str = "^VIX",
        vvix_ticker: str = "^VVIX",
        skew_ticker: str = "^SKEW",
        atr_window: int = 14,
    ) -> pd.DataFrame | None:
        logger.info(
            f"Fetching volatility-related data from {start_date} to {end_date} "
            f"for tickers: {vix_ticker}, {vvix_ticker}, {skew_ticker}, {sp500_ticker}"
        )

        dfs_to_merge = []

        vix_data = self.yf_client.fetch_ohlcv_data(vix_ticker, start_date, end_date)
        if vix_data is None or vix_data.empty:
            logger.warning(f"Could not fetch VIX data ({vix_ticker}).")
            vix_df = pd.DataFrame(columns=["date", "vix"]).set_index("date")
        else:
            vix_df = vix_data[["date", "close"]].rename(columns={"close": "vix"}).set_index("date")
            for window in self.vix_sma_windows:
                sma_col_name = f"vix_sma_{window}"
                vix_df[sma_col_name] = self._calculate_sma(vix_df["vix"], window)
        dfs_to_merge.append(vix_df)

        vvix_data = self.yf_client.fetch_ohlcv_data(vvix_ticker, start_date, end_date)
        if vvix_data is None or vvix_data.empty:
            logger.warning(f"Could not fetch VVIX data ({vvix_ticker}).")
            vvix_df = pd.DataFrame(columns=["date", "vvix"]).set_index("date")
        else:
            vvix_df = (
                vvix_data[["date", "close"]].rename(columns={"close": "vvix"}).set_index("date")
            )
        dfs_to_merge.append(vvix_df)

        skew_data = self.yf_client.fetch_ohlcv_data(skew_ticker, start_date, end_date)
        if skew_data is None or skew_data.empty:
            logger.warning(f"Could not fetch SKEW data ({skew_ticker}).")
            skew_df = pd.DataFrame(columns=["date", "skew_index"]).set_index("date")
        else:
            skew_df = (
                skew_data[["date", "close"]]
                .rename(columns={"close": "skew_index"})
                .set_index("date")
            )
        dfs_to_merge.append(skew_df)

        buffer_days = atr_window + 50
        atr_start_date = (pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)).strftime(
            "%Y-%m-%d"
        )

        sp500_data = self.yf_client.fetch_ohlcv_data(sp500_ticker, atr_start_date, end_date)
        if sp500_data is None or sp500_data.empty:
            logger.warning(f"Could not fetch S&P 500 data ({sp500_ticker}) for ATR calculation.")
            atr_df = pd.DataFrame(columns=["date", f"atr_{atr_window}"]).set_index("date")
        else:
            sp500_data_indexed = sp500_data.set_index("date")
            atr_series = self._calculate_atr(
                sp500_data_indexed["high"],
                sp500_data_indexed["low"],
                sp500_data_indexed["close"],
                window=atr_window,
            )
            atr_df = atr_series.to_frame().rename(columns={f"atr_{atr_window}": "atr"})
            atr_df = atr_df[atr_df.index >= pd.to_datetime(start_date)]
        dfs_to_merge.append(atr_df)

        if not dfs_to_merge:
            logger.error("No data fetched for any volatility metric. Returning None.")
            return None

        merged_df = pd.concat(dfs_to_merge, axis=1, join="outer")

        cols_to_ffill = ["vix", "vvix", "skew_index"]
        for window in self.vix_sma_windows:  # also ffill vix smas
            cols_to_ffill.append(f"vix_sma_{window}")

        merged_df[cols_to_ffill] = merged_df[cols_to_ffill].ffill()

        merged_df.reset_index(inplace=True)
        merged_df["date"] = pd.to_datetime(merged_df["date"]).dt.strftime("%Y-%m-%d")

        merged_df = merged_df[(merged_df["date"] >= start_date) & (merged_df["date"] <= end_date)]

        final_columns = ["date", "vix"]
        for w in self.vix_sma_windows:
            final_columns.append(f"vix_sma_{w}")
        final_columns.extend(["vvix", "atr", "skew_index"])

        for col in final_columns:
            if col not in merged_df.columns:
                merged_df[col] = np.nan

        merged_df = merged_df[final_columns]

        indicator_cols = [col for col in final_columns if col != "date"]
        merged_df.dropna(subset=indicator_cols, how="all", inplace=True)

        if merged_df.empty:
            logger.warning("Final volatility metrics DataFrame is empty after processing.")
            return None

        logger.info(f"Successfully calculated volatility metrics. Shape: {merged_df.shape}")
        return merged_df
