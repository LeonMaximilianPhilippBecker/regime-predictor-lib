import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    def __init__(self):
        logger.info("TechnicalIndicatorCalculator initialized.")

    def _calculate_sma(self, series: pd.Series, window: int) -> pd.Series:
        if series.empty or len(series) < window:
            return pd.Series(index=series.index, dtype=float)
        return series.rolling(window=window, min_periods=window).mean()

    def _calculate_ema(self, series: pd.Series, window: int) -> pd.Series:
        if series.empty or len(series) < window:
            return pd.Series(index=series.index, dtype=float)
        return series.ewm(span=window, adjust=False, min_periods=window).mean()

    def _calculate_macd(
        self,
        close_prices: pd.Series,
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        if close_prices.empty:
            return (
                pd.Series(dtype=float, index=close_prices.index, name="macd"),
                pd.Series(dtype=float, index=close_prices.index, name="macd_signal"),
                pd.Series(dtype=float, index=close_prices.index, name="macd_hist"),
            )

        ema_short = self._calculate_ema(close_prices, short_window)
        ema_long = self._calculate_ema(close_prices, long_window)
        macd_line = ema_short - ema_long
        macd_signal_line = self._calculate_ema(macd_line, signal_window)
        macd_histogram = macd_line - macd_signal_line
        return (
            macd_line.rename("macd"),
            macd_signal_line.rename("macd_signal"),
            macd_histogram.rename("macd_histogram"),
        )

    def _calculate_rsi(self, close_prices: pd.Series, window: int = 14) -> pd.Series:
        if close_prices.empty or len(close_prices) < window + 1:
            return pd.Series(index=close_prices.index, dtype=float, name="rsi")

        delta = close_prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=window - 1, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, adjust=False, min_periods=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi.loc[avg_loss == 0] = 100
        rsi.loc[(avg_gain == 0) & (avg_loss == 0)] = 0

        return rsi.rename("rsi")

    def _calculate_roc(self, close_prices: pd.Series, window: int = 12) -> pd.Series:
        if close_prices.empty or len(close_prices) < window + 1:
            return pd.Series(index=close_prices.index, dtype=float)
        roc = ((close_prices - close_prices.shift(window)) / close_prices.shift(window)) * 100
        return roc.rename("roc")

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> pd.Series:
        if high.empty or len(high) < window:
            return pd.Series(index=high.index, dtype=float, name=f"atr_{window}")

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        return atr.rename(f"atr_{window}")

    def _calculate_adx(
        self,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        window: int = 14,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        if high_prices.empty or len(high_prices) < window + 1:
            return (
                pd.Series(dtype=float, index=high_prices.index, name="adx"),
                pd.Series(dtype=float, index=high_prices.index, name="plus_di"),
                pd.Series(dtype=float, index=high_prices.index, name="minus_di"),
            )

        atr_series = self._calculate_atr(high_prices, low_prices, close_prices, window)

        move_up = high_prices.diff()
        move_down = -low_prices.diff()

        plus_dm = pd.Series(
            np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=high_prices.index
        )
        minus_dm = pd.Series(
            np.where((move_down > move_up) & (move_down > 0), move_down, 0.0),
            index=high_prices.index,
        )

        smoothed_plus_dm = plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        smoothed_minus_dm = minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

        plus_di = 100 * (smoothed_plus_dm / atr_series)
        minus_di = 100 * (smoothed_minus_dm / atr_series)

        dx_abs = abs(plus_di - minus_di)
        dx_sum = plus_di + minus_di
        dx = 100 * (dx_abs / dx_sum.replace(0, np.nan))
        dx.fillna(0, inplace=True)

        adx = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

        return adx.rename("adx"), plus_di.rename("plus_di"), minus_di.rename("minus_di")

    def _calculate_stochastic_oscillator(
        self,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        k_window: int = 14,
        d_window: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        if high_prices.empty or len(high_prices) < k_window:
            return (
                pd.Series(dtype=float, index=high_prices.index, name="stoch_k"),
                pd.Series(dtype=float, index=high_prices.index, name="stoch_d"),
            )

        low_n = low_prices.rolling(window=k_window, min_periods=k_window).min()
        high_n = high_prices.rolling(window=k_window, min_periods=k_window).max()

        stoch_k = 100 * ((close_prices - low_n) / (high_n - low_n))
        stoch_k.replace([np.inf, -np.inf], np.nan, inplace=True)
        stoch_k.fillna(method="ffill", inplace=True)

        stoch_d = self._calculate_sma(stoch_k, d_window)
        return stoch_k.rename("stoch_k"), stoch_d.rename("stoch_d")

    def _calculate_williams_r(
        self,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        if high_prices.empty or len(high_prices) < window:
            return pd.Series(index=high_prices.index, dtype=float, name="williams_r")

        low_n = low_prices.rolling(window=window, min_periods=window).min()
        high_n = high_prices.rolling(window=window, min_periods=window).max()

        williams_r = -100 * ((high_n - close_prices) / (high_n - low_n))
        williams_r.replace([np.inf, -np.inf], np.nan, inplace=True)
        williams_r.fillna(method="ffill", inplace=True)

        return williams_r.rename("williams_r")

    def calculate_all_indicators(self, ohlcv_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
            if "date" in ohlcv_df.columns:
                ohlcv_df = ohlcv_df.set_index("date")
            else:
                raise ValueError("DataFrame must have a DatetimeIndex or a 'date' column.")

        ohlcv_df = ohlcv_df.sort_index()

        close = ohlcv_df["close"]
        high = ohlcv_df["high"]
        low = ohlcv_df["low"]

        indicators = pd.DataFrame(index=ohlcv_df.index)
        indicators["symbol"] = symbol

        for window in [20, 50, 100, 200]:
            indicators[f"sma_{window}"] = self._calculate_sma(close, window)

        for window in [20, 50, 100, 200]:
            indicators[f"ema_{window}"] = self._calculate_ema(close, window)

        macd_line, macd_signal, macd_hist = self._calculate_macd(close)
        indicators["macd"] = macd_line
        indicators["macd_signal"] = macd_signal
        indicators["macd_histogram"] = macd_hist

        indicators["rsi_14"] = self._calculate_rsi(close, window=14)

        indicators["roc"] = self._calculate_roc(close, window=12)

        adx_val, _, _ = self._calculate_adx(high, low, close, window=14)
        indicators["adx"] = adx_val

        stoch_k, stoch_d = self._calculate_stochastic_oscillator(
            high, low, close, k_window=14, d_window=3
        )
        indicators["stochastic_k"] = stoch_k
        indicators["stochastic_d"] = stoch_d

        indicators["williams_r"] = self._calculate_williams_r(high, low, close, window=14)

        indicators.reset_index(inplace=True)
        indicators["date"] = pd.to_datetime(indicators["date"]).dt.strftime("%Y-%m-%d")

        schema_cols = [
            "symbol",
            "date",
            "sma_20",
            "sma_50",
            "sma_100",
            "sma_200",
            "ema_20",
            "ema_50",
            "ema_100",
            "ema_200",
            "macd",
            "macd_histogram",
            "rsi_14",
            "roc",
            "adx",
            "stochastic_k",
            "stochastic_d",
            "williams_r",
        ]

        for col in schema_cols:
            if col not in indicators.columns:
                indicators[col] = np.nan

        final_indicators_df = indicators[schema_cols]

        indicator_only_cols = [col for col in schema_cols if col not in ["symbol", "date"]]
        final_indicators_df.dropna(subset=indicator_only_cols, how="all", inplace=True)

        if final_indicators_df.empty:
            logger.warning(
                f"Final technical indicators DataFrame for {symbol} is empty after processing."
            )
            return pd.DataFrame(columns=schema_cols)

        logger.info(
            f"Calculated technical indicators for {symbol}. Shape: {final_indicators_df.shape}"
        )
        return final_indicators_df
