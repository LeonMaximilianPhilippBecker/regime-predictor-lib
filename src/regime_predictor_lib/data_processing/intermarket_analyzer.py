import logging

import numpy as np
import pandas as pd

from regime_predictor_lib.data_ingestion.api_clients import YFinanceClient
from regime_predictor_lib.utils.financial_calculations import (
    calculate_percentile_rank,
    calculate_roc,
    calculate_sma,
    calculate_sma_crossover_signal,
    calculate_value_vs_sma_signal,
    calculate_z_score,
)

logger = logging.getLogger(__name__)


class IntermarketAnalyzer:
    def __init__(self, yf_client: YFinanceClient):
        self.yf_client = yf_client
        self.roc_periods_days = [21, 63, 126, 252]
        self.sma_periods_days = [20, 50, 200]
        self.z_score_windows_days = [63, 252]
        self.percentile_windows_days = [252]
        logger.info("IntermarketAnalyzer initialized.")

    def _fetch_and_prepare_data(
        self, tickers: list[str], start_date: str, end_date: str | None
    ) -> pd.DataFrame | None:
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        all_data = {}
        for ticker in tickers:
            df_raw = self.yf_client.fetch_ohlcv_data(ticker, start_date, end_date)
            if df_raw is None or df_raw.empty:
                logger.warning(f"Could not fetch data for {ticker}.")
                return None
            df = df_raw[["date", "adjusted_close"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.rename(columns={"adjusted_close": f"{ticker}_close"}, inplace=True)
            all_data[ticker] = df

        if not all_data:
            return None

        if len(all_data) == 1:
            merged_df = list(all_data.values())[0]
        else:
            dfs_to_merge = list(all_data.values())
            merged_df = dfs_to_merge[0]
            for i in range(1, len(dfs_to_merge)):
                merged_df = pd.merge(merged_df, dfs_to_merge[i], on="date", how="outer")

        merged_df.sort_index(inplace=True)
        return merged_df

    def calculate_stock_bond_return_difference(
        self,
        stock_ticker: str = "SPY",
        bond_ticker: str = "TLT",
        start_date_signals: str = "2003-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        logger.info(f"Calculating stock-bond return difference for {stock_ticker} vs {bond_ticker}")
        buffer_start_date = (pd.to_datetime(start_date_signals) - pd.DateOffset(days=300)).strftime(
            "%Y-%m-%d"
        )
        data = self._fetch_and_prepare_data(
            [stock_ticker, bond_ticker], buffer_start_date, end_date
        )

        if (
            data is None
            or f"{stock_ticker}_close" not in data
            or f"{bond_ticker}_close" not in data
        ):
            logger.error(f"Could not retrieve necessary data for {stock_ticker} or {bond_ticker}.")
            return None

        stock_prices = data[f"{stock_ticker}_close"].dropna()
        bond_prices = data[f"{bond_ticker}_close"].dropna()

        aligned_data = pd.merge(
            stock_prices.rename("stock"),
            bond_prices.rename("bond"),
            left_index=True,
            right_index=True,
            how="inner",
        )
        if aligned_data.empty:
            logger.warning("No overlapping data for stock and bond tickers.")
            return None

        stock_ret_20d = aligned_data["stock"].pct_change(periods=20)
        bond_ret_20d = aligned_data["bond"].pct_change(periods=20)

        signals_df = pd.DataFrame(index=aligned_data.index)
        signals_df["return_diff_20d"] = stock_ret_20d - bond_ret_20d

        signals_df["return_diff_20d_sma_20"] = calculate_sma(signals_df["return_diff_20d"], 20)
        signals_df["return_diff_20d_sma_50"] = calculate_sma(signals_df["return_diff_20d"], 50)

        for window in self.z_score_windows_days:
            signals_df[f"return_diff_20d_zscore_{window}d"] = calculate_z_score(
                signals_df["return_diff_20d"], window
            )

        signals_df = signals_df[signals_df.index >= pd.to_datetime(start_date_signals)]
        signals_df.dropna(subset=["return_diff_20d"], how="all", inplace=True)
        signals_df.reset_index(inplace=True)
        signals_df["date"] = signals_df["date"].dt.strftime("%Y-%m-%d")

        logger.info(f"Calculated stock-bond return difference. Shape: {signals_df.shape}")
        return signals_df

    def calculate_generic_ratio(
        self,
        numerator_ticker: str,
        denominator_ticker: str,
        ratio_name: str,
        start_date_signals: str = "1990-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        logger.info(f"Calculating ratio '{ratio_name}' for {numerator_ticker}/{denominator_ticker}")
        buffer_start_date = (pd.to_datetime(start_date_signals) - pd.DateOffset(days=300)).strftime(
            "%Y-%m-%d"
        )

        data = self._fetch_and_prepare_data(
            [numerator_ticker, denominator_ticker], buffer_start_date, end_date
        )

        if (
            data is None
            or f"{numerator_ticker}_close" not in data
            or f"{denominator_ticker}_close" not in data
        ):
            logger.error(
                f"Could not retrieve necessary data for {numerator_ticker} or {denominator_ticker}."
            )
            return None

        num_series = data[f"{numerator_ticker}_close"]
        den_series = data[f"{denominator_ticker}_close"]

        ratio_data = pd.merge(
            num_series.rename("num"),
            den_series.rename("den"),
            left_index=True,
            right_index=True,
            how="inner",
        )

        if ratio_data.empty or ratio_data["num"].isnull().all() or ratio_data["den"].isnull().all():
            logger.warning(
                f"Not enough overlapping data for {numerator_ticker}/{denominator_ticker}."
            )
            return None

        signals_df = pd.DataFrame(index=ratio_data.index)
        signals_df["raw_ratio"] = ratio_data["num"] / ratio_data["den"]
        signals_df["raw_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

        signals_df.dropna(subset=["raw_ratio"], inplace=True)
        if signals_df.empty:
            logger.warning(f"Raw ratio for {ratio_name} is all NaN. Cannot proceed.")
            return None

        for p in self.roc_periods_days:
            signals_df[f"roc_{p}d"] = calculate_roc(signals_df["raw_ratio"], p)
        for p in self.sma_periods_days:
            signals_df[f"sma_{p}d"] = calculate_sma(signals_df["raw_ratio"], p)
        for p in self.z_score_windows_days:
            signals_df[f"zscore_value_{p}d"] = calculate_z_score(signals_df["raw_ratio"], p)
        for p in self.percentile_windows_days:
            signals_df[f"percentile_{p // 252}y"] = calculate_percentile_rank(
                signals_df["raw_ratio"], p
            )

        if "sma_20d" in signals_df.columns:
            signals_df["vs_sma20_signal"] = calculate_value_vs_sma_signal(
                signals_df["raw_ratio"], signals_df["sma_20d"]
            )
        if "sma_20d" in signals_df.columns and "sma_50d" in signals_df.columns:
            signals_df["sma20_vs_sma50_signal"] = calculate_sma_crossover_signal(
                signals_df["sma_20d"], signals_df["sma_50d"]
            )

        signals_df["ratio_name"] = ratio_name
        signals_df["numerator_ticker"] = numerator_ticker
        signals_df["denominator_ticker"] = denominator_ticker

        signals_df = signals_df[signals_df.index >= pd.to_datetime(start_date_signals)]

        signal_cols_to_check = [
            col
            for col in signals_df.columns
            if col not in ["ratio_name", "numerator_ticker", "denominator_ticker", "raw_ratio"]
        ]
        signals_df.dropna(subset=signal_cols_to_check, how="all", inplace=True)

        signals_df.reset_index(inplace=True)
        signals_df["date"] = signals_df["date"].dt.strftime("%Y-%m-%d")

        ordered_cols = (
            ["date", "ratio_name", "numerator_ticker", "denominator_ticker", "raw_ratio"]
            + sorted([c for c in signals_df.columns if c.startswith("roc_")])
            + sorted(
                [
                    c
                    for c in signals_df.columns
                    if c.startswith("sma_") and c != "sma20_vs_sma50_signal"
                ]
            )
            + sorted([c for c in signals_df.columns if c.startswith("zscore_")])
            + sorted([c for c in signals_df.columns if c.startswith("percentile_")])
            + [c for c in ["vs_sma20_signal", "sma20_vs_sma50_signal"] if c in signals_df.columns]
        )

        for col in ["vs_sma20_signal", "sma20_vs_sma50_signal"]:
            if col not in signals_df.columns:
                signals_df[col] = pd.NA

        current_cols = signals_df.columns.tolist()
        final_cols = [col for col in ordered_cols if col in current_cols]
        final_cols.extend([col for col in current_cols if col not in final_cols])

        signals_df = signals_df[final_cols]

        logger.info(f"Calculated ratio signals for {ratio_name}. Shape: {signals_df.shape}")
        return signals_df
