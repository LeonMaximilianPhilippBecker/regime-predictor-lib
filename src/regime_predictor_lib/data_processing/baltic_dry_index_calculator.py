import logging

import numpy as np
import pandas as pd

from ..utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class BalticDryIndexCalculator:
    def __init__(self, db_manager: DatabaseManager, oil_series_for_ratio: str = "DCOILWTICO"):
        self.db_manager = db_manager
        self.oil_series_for_ratio = oil_series_for_ratio
        logger.info(
            "BalticDryIndexCalculator initialized. "
            f"Will use {oil_series_for_ratio} for BDI/Oil ratio."
        )

    def _fetch_raw_bdi_data(self) -> pd.DataFrame:
        query = """
            SELECT date, value
            FROM bdi_raw_csv
            ORDER BY date ASC;
        """
        session = self.db_manager.get_session()
        try:
            df = pd.read_sql_query(query, session.bind, parse_dates=["date"])
            df.rename(columns={"value": "bdi_value"}, inplace=True)
            df.set_index("date", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching BDI raw data: {e}")
            return pd.DataFrame()
        finally:
            session.close()

    def _fetch_oil_prices_for_ratio(self) -> pd.DataFrame:
        query = f"""
            SELECT date, price
            FROM oil_price_signals
            WHERE symbol = '{self.oil_series_for_ratio}'
            ORDER BY date ASC;
        """
        session = self.db_manager.get_session()
        try:
            df = pd.read_sql_query(query, session.bind, parse_dates=["date"])
            df.rename(columns={"price": "oil_price"}, inplace=True)
            df.set_index("date", inplace=True)
            return df
        except Exception as e:
            logger.error(
                f"Error fetching Oil prices for BDI ratio ({self.oil_series_for_ratio}): {e}"
            )
            return pd.DataFrame()
        finally:
            session.close()

    def calculate_signals(self) -> pd.DataFrame | None:
        bdi_df = self._fetch_raw_bdi_data()
        if bdi_df.empty:
            logger.warning("No raw BDI data found. Cannot calculate signals.")
            return None

        signals_df = pd.DataFrame(index=bdi_df.index)
        signals_df["bdi_value"] = bdi_df["bdi_value"]

        # 1. 1M % change (approx 21 trading days, adjust if BDI is not daily)
        signals_df["pct_change_1m"] = bdi_df["bdi_value"].pct_change(periods=21)

        # 2. Z-score of 3M returns (approx 63 trading days)
        return_3m = bdi_df["bdi_value"].pct_change(periods=63)
        rolling_mean_3m_ret = return_3m.rolling(window=63, min_periods=21).mean()
        rolling_std_3m_ret = return_3m.rolling(window=63, min_periods=21).std()
        signals_df["z_score_return_3m"] = (return_3m - rolling_mean_3m_ret) / rolling_std_3m_ret

        # 3. EMA(30) - EMA(90)
        ema_30 = bdi_df["bdi_value"].ewm(span=30, adjust=False, min_periods=30).mean()
        ema_90 = bdi_df["bdi_value"].ewm(span=90, adjust=False, min_periods=90).mean()
        signals_df["ema_30_minus_ema_90"] = ema_30 - ema_90

        # 4. BDI / Oil ratio
        oil_prices_df = self._fetch_oil_prices_for_ratio()
        if not oil_prices_df.empty:
            merged_for_ratio = pd.merge(
                bdi_df[["bdi_value"]],
                oil_prices_df[["oil_price"]],
                left_index=True,
                right_index=True,
                how="left",
            )
            merged_for_ratio["oil_price"].ffill(inplace=True)
            signals_df["bdi_oil_ratio"] = (
                merged_for_ratio["bdi_value"] / merged_for_ratio["oil_price"]
            )
            signals_df["oil_symbol_for_ratio"] = self.oil_series_for_ratio
        else:
            signals_df["bdi_oil_ratio"] = np.nan
            signals_df["oil_symbol_for_ratio"] = None
            logger.warning(
                f"Oil prices for ratio ({self.oil_series_for_ratio}) "
                "not found. BDI/Oil ratio will be NaN."
            )

        signals_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        signals_df.dropna(
            how="all",
            subset=["pct_change_1m", "z_score_return_3m", "ema_30_minus_ema_90", "bdi_oil_ratio"],
            inplace=True,
        )

        signals_df.reset_index(inplace=True)
        signals_df["date"] = signals_df["date"].dt.strftime("%Y-%m-%d")

        logger.info(f"Calculated BDI signals. Shape: {signals_df.shape}")
        return signals_df
