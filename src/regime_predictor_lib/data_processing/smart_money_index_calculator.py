import logging

import pandas as pd

from regime_predictor_lib.data_ingestion.api_clients import YFinanceClient
from regime_predictor_lib.utils.financial_calculations import (
    calculate_percentile_rank,
    calculate_roc,
    calculate_sma,
    calculate_sma_crossover_signal,
    calculate_value_vs_sma_signal,
)

logger = logging.getLogger(__name__)


class SmartMoneyIndexCalculator:
    def __init__(
        self,
        yf_client: YFinanceClient,
        roc_periods_days: list[int] | None = None,
        percentile_windows_days: list[int] | None = None,
        sma_windows_days: list[int] | None = None,
    ):
        self.yf_client = yf_client
        self.roc_periods = roc_periods_days or [21, 63, 126]
        self.percentile_windows = percentile_windows_days or [252, 504]
        self.sma_windows = sma_windows_days or [20, 50, 200]
        logger.info("SmartMoneyIndexCalculator initialized.")

    def calculate_smi_and_signals(
        self,
        symbol: str = "SPY",
        start_date: str = "1993-01-29",
        end_date: str | None = None,
        initial_smi_value: float = 0.0,
    ) -> pd.DataFrame | None:
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        buffer_start_date = (
            pd.to_datetime(start_date) - pd.DateOffset(days=max(self.sma_windows) + 50)
        ).strftime("%Y-%m-%d")

        spy_df_raw = self.yf_client.fetch_ohlcv_data(symbol, buffer_start_date, end_date)

        if spy_df_raw is None or spy_df_raw.empty:
            logger.error(f"Could not fetch SPY data for SMI calculation (symbol: {symbol}).")
            return None

        spy_df = spy_df_raw.copy()
        spy_df["date"] = pd.to_datetime(spy_df["date"])
        spy_df.set_index("date", inplace=True)
        spy_df.sort_index(inplace=True)

        if not all(col in spy_df.columns for col in ["open", "close"]):
            logger.error("SPY data missing 'open' or 'close' columns.")
            return None

        spy_df["smi_change"] = spy_df["close"] - 2 * spy_df["open"] + spy_df["close"].shift(1)
        spy_df["smi_value"] = spy_df["smi_change"].cumsum() + initial_smi_value
        spy_df["smi_value"].fillna(initial_smi_value, inplace=True)

        signals_df = pd.DataFrame(index=spy_df.index)
        signals_df["smi_value"] = spy_df["smi_value"]
        signals_df["spy_open"] = spy_df["open"]
        signals_df["spy_close"] = spy_df["close"]

        for period in self.roc_periods:
            signals_df[f"smi_roc_{period}d"] = calculate_roc(signals_df["smi_value"], period)

        for window in self.sma_windows:
            signals_df[f"smi_sma_{window}d"] = calculate_sma(signals_df["smi_value"], window)

        if "smi_sma_20d" in signals_df.columns:
            signals_df["smi_vs_sma20_signal"] = calculate_value_vs_sma_signal(
                signals_df["smi_value"], signals_df["smi_sma_20d"]
            )
        if "smi_sma_20d" in signals_df.columns and "smi_sma_50d" in signals_df.columns:
            signals_df["smi_sma20_vs_sma50_signal"] = calculate_sma_crossover_signal(
                signals_df["smi_sma_20d"], signals_df["smi_sma_50d"]
            )

        for window in self.percentile_windows:
            min_p_perc = max(1, window // 2)
            signals_df[f"smi_percentile_{window}d"] = calculate_percentile_rank(
                signals_df["smi_value"], window, min_periods=min_p_perc
            )

        signals_df = signals_df[signals_df.index >= pd.to_datetime(start_date)]

        signals_df.reset_index(inplace=True)
        signals_df["date"] = signals_df["date"].dt.strftime("%Y-%m-%d")

        logger.info(f"Calculated SMI and signals. Shape: {signals_df.shape}")
        return signals_df
