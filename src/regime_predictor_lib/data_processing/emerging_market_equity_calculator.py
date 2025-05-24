import logging

import numpy as np
import pandas as pd

from ..data_ingestion.api_clients import YFinanceClient
from ..utils.database_manager import DatabaseManager  # If storing raw data

logger = logging.getLogger(__name__)


class EmergingMarketEquityCalculator:
    def __init__(
        self,
        yf_client: YFinanceClient,
        db_manager: DatabaseManager,
        em_symbol: str = "EEM",
        market_symbol: str = "SPY",
    ):
        self.yf_client = yf_client
        self.db_manager = db_manager
        self.em_symbol = em_symbol
        self.market_symbol = market_symbol
        logger.info(
            f"EmergingMarketEquityCalculator initialized for EM: {em_symbol}, "
            f"Market: {market_symbol}"
        )

    def _fetch_ohlcv_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        df = self.yf_client.fetch_ohlcv_data(symbol, start_date, end_date)
        if df is None or df.empty:
            logger.warning(f"Could not fetch data for {symbol}")
            return None
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df[["adjusted_close"]].rename(columns={"adjusted_close": "price"})

    def calculate_signals(self, start_date="2000-01-01", end_date=None) -> pd.DataFrame | None:
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        extended_start_date = (pd.to_datetime(start_date) - pd.DateOffset(years=2)).strftime(
            "%Y-%m-%d"
        )

        em_df = self._fetch_ohlcv_data(self.em_symbol, extended_start_date, end_date)
        spy_df = self._fetch_ohlcv_data(self.market_symbol, extended_start_date, end_date)

        if em_df is None or spy_df is None or em_df.empty or spy_df.empty:
            logger.error("Could not fetch data for EM or SPY. Cannot calculate signals.")
            return None

        merged_df = pd.merge(
            em_df.rename(columns={"price": "em_price"}),
            spy_df.rename(columns={"price": "spy_price"}),
            left_index=True,
            right_index=True,
            how="inner",
        )

        if merged_df.empty:
            logger.warning("No overlapping data for EM and SPY.")
            return None

        signals_df = pd.DataFrame(index=merged_df.index)
        signals_df["symbol"] = self.em_symbol
        signals_df["close_price"] = merged_df["em_price"]

        # 21-day % return
        signals_df["pct_return_21d"] = merged_df["em_price"].pct_change(periods=21)

        # Z-score of returns (3M window)
        return_3m = merged_df["em_price"].pct_change(periods=63)
        rolling_mean_3m_ret = return_3m.rolling(window=63, min_periods=21).mean()
        rolling_std_3m_ret = return_3m.rolling(window=63, min_periods=21).std()
        signals_df["z_score_return_3m"] = (return_3m - rolling_mean_3m_ret) / rolling_std_3m_ret

        # Price above/below 200-day MA
        sma_200 = merged_df["em_price"].rolling(window=200, min_periods=100).mean()
        signals_df["above_sma_200"] = (merged_df["em_price"] > sma_200).astype(int)

        # Relative performance to SPY (simple ratio of prices)
        signals_df["relative_performance_spy"] = merged_df["em_price"] / merged_df["spy_price"]

        # 90-day rolling beta to SPY
        em_returns = merged_df["em_price"].pct_change()
        spy_returns = merged_df["spy_price"].pct_change()

        rolling_cov = em_returns.rolling(window=90, min_periods=45).cov(spy_returns)
        rolling_var_spy = spy_returns.rolling(window=90, min_periods=45).var()
        signals_df["beta_to_spy_90d"] = rolling_cov / rolling_var_spy

        signals_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        signals_df = signals_df[signals_df.index >= pd.to_datetime(start_date)]

        signals_df.dropna(
            how="all",
            subset=[
                "pct_return_21d",
                "z_score_return_3m",
                "above_sma_200",
                "relative_performance_spy",
                "beta_to_spy_90d",
            ],
            inplace=True,
        )

        signals_df.reset_index(inplace=True)
        signals_df["date"] = signals_df["date"].dt.strftime("%Y-%m-%d")

        logger.info(f"Calculated EM Equity signals for {self.em_symbol}. Shape: {signals_df.shape}")
        return signals_df
