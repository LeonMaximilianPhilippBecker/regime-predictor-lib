import logging
from pathlib import Path
from sqlite3 import OperationalError as SQLiteOperationalError

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError

from regime_predictor_lib.utils.database_manager import DatabaseManager
from regime_predictor_lib.utils.financial_calculations import (
    calculate_percentile_rank,
    calculate_roc,
    calculate_sma,
    calculate_z_score,
)

logger = logging.getLogger(__name__)

SMA_WINDOWS_DAILY = [5, 20, 21, 50, 63, 200]
ROC_WINDOWS_DAILY = [5, 21, 63]
ZSCORE_WINDOWS_DAILY_SHORT = [60, 63]
ZSCORE_WINDOWS_DAILY_LONG = [252]
PERCENTILE_WINDOWS_DAILY = [252]

SMA_WINDOWS_ECO_M = [3, 6, 12]
SMA_WINDOWS_ECO_W = [4, 8, 52]
ZSCORE_WINDOWS_ECO_M = [24, 36]
ZSCORE_WINDOWS_ECO_W = [52, 104]
PERCENTILE_WINDOWS_ECO_M = [36, 60]
PERCENTILE_WINDOWS_ECO_W = [52, 104]


class MasterSignalOrchestrator:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def _apply_sql_file(self, sql_file_path: Path):
        if not sql_file_path.exists():
            logger.error(f"SQL file not found: {sql_file_path}")
            raise FileNotFoundError(f"SQL file not found: {sql_file_path}")

        with open(sql_file_path, "r") as f:
            sql_script = f.read()

        session = self.db_manager.get_session()
        try:
            statements = [stmt.strip() for stmt in sql_script.split(";") if stmt.strip()]
            for stmt in statements:
                if stmt:
                    logger.debug(f"Executing SQL: {stmt[:200]}...")
                    try:
                        session.execute(sqlalchemy.text(stmt))
                    except (
                        SQLAlchemyOperationalError,
                        SQLiteOperationalError,
                    ) as oe:
                        error_msg = str(oe).lower()
                        if "duplicate column name" in error_msg or "already exists" in error_msg:
                            logger.warning(
                                "Skipping DDL (column/table likely already exists):"
                                f" {stmt[:100]}... Error: {oe}"
                            )
                        else:
                            raise
            session.commit()
            logger.info(
                f"Successfully executed SQL script: {sql_file_path.name} "
                "(ignoring duplicates if any)"
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Error executing SQL script {sql_file_path.name}: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def _get_point_in_time_series(
        self,
        raw_vintage_df: pd.DataFrame,
        value_col_names: list[str],
        series_id_col_name: str | None = None,
        extra_groupby_cols: list | None = None,
    ) -> pd.DataFrame:
        if raw_vintage_df.empty:
            return pd.DataFrame()

        raw_vintage_df["reference_date"] = pd.to_datetime(raw_vintage_df["reference_date"])
        raw_vintage_df["release_date"] = pd.to_datetime(raw_vintage_df["release_date"])

        sort_by_cols = ["reference_date", "release_date"]
        groupby_keys = ["reference_date"]

        if series_id_col_name and series_id_col_name in raw_vintage_df.columns:
            if series_id_col_name not in sort_by_cols:
                sort_by_cols.append(series_id_col_name)
            if series_id_col_name not in groupby_keys:
                groupby_keys.append(series_id_col_name)
        if extra_groupby_cols:
            for col in extra_groupby_cols:
                if col in raw_vintage_df.columns:
                    if col not in sort_by_cols:
                        sort_by_cols.append(col)
                    if col not in groupby_keys:
                        groupby_keys.append(col)

        raw_vintage_df_sorted = raw_vintage_df.sort_values(by=sort_by_cols)
        pit_df = raw_vintage_df_sorted.groupby(groupby_keys, as_index=False).first()

        final_cols = ["reference_date", "release_date"]
        renamed_value_cols = {}
        for value_col in value_col_names:
            pit_col_name = f"{value_col}_pit"
            if value_col in pit_df.columns:
                renamed_value_cols[value_col] = pit_col_name
                if pit_col_name not in final_cols:
                    final_cols.append(pit_col_name)
            elif pit_col_name in pit_df.columns:
                if pit_col_name not in final_cols:
                    final_cols.append(pit_col_name)

        pit_df.rename(columns=renamed_value_cols, inplace=True)

        if series_id_col_name and series_id_col_name in pit_df.columns:
            if series_id_col_name not in final_cols:
                final_cols.append(series_id_col_name)
        if extra_groupby_cols:
            final_cols.extend(
                [
                    col
                    for col in extra_groupby_cols
                    if col in pit_df.columns and col not in final_cols
                ]
            )

        final_cols = list(dict.fromkeys(final_cols))
        return pit_df[final_cols]

    def _get_sp500_close_data(self, start_date_str: str, end_date_str: str) -> pd.Series:
        session = self.db_manager.get_session()
        try:
            query = sqlalchemy.text(
                """
                SELECT date, sp500_adjusted_close
                FROM sp500_derived_indicators
                WHERE date >= :start_date AND date <= :end_date
                ORDER BY date;
            """
            )
            df = pd.read_sql_query(
                query,
                session.bind,
                params={"start_date": start_date_str, "end_date": end_date_str},
                parse_dates=["date"],
            )
            if df.empty:
                logger.warning(
                    "No S&P 500 close data found in sp500_derived_indicators "
                    "for the given date range."
                )
                return pd.Series(dtype=float, name="close_price")
            return df.set_index("date")["sp500_adjusted_close"].rename("close_price")
        except Exception as e:
            logger.error(f"Error fetching SP500 close from DB: {e}")
            return pd.Series(dtype=float, name="close_price")
        finally:
            session.close()

    def _process_technical_indicators_signals(self):
        logger.info("Processing signals for technical_indicators table...")
        table_name = "technical_indicators"
        session = self.db_manager.get_session()
        try:
            symbol_filter = "^GSPC"
            df_ti_raw = pd.read_sql_query(
                f"SELECT * FROM {table_name} WHERE symbol = '{symbol_filter}' ORDER BY date ASC;",
                session.bind,
                parse_dates=["date"],
            )
            if df_ti_raw.empty:
                logger.warning(f"No data for {symbol_filter} in {table_name}.")
                return

            min_date_ti, max_date_ti = df_ti_raw["date"].min(), df_ti_raw["date"].max()
            sp500_close = self._get_sp500_close_data(
                min_date_ti.strftime("%Y-%m-%d"), max_date_ti.strftime("%Y-%m-%d")
            )

            df = pd.merge(df_ti_raw, sp500_close, on="date", how="left")
            df.set_index("date", inplace=True)

            if "close_price" not in df.columns or df["close_price"].isnull().all():
                logger.error(
                    "Close price not available for technical indicator "
                    "signal calculation. Skipping."
                )
                return

            df["price_vs_sma50_pct_diff"] = (df["close_price"] - df["sma_50"]) / df["sma_50"]
            df["price_vs_sma200_pct_diff"] = (df["close_price"] - df["sma_200"]) / df["sma_200"]
            df["sma20_vs_sma50_signal"] = np.sign(df["sma_20"] - df["sma_50"]).astype("Int64")
            df["sma50_vs_sma200_signal"] = np.sign(df["sma_50"] - df["sma_200"]).astype("Int64")
            df["rsi_14_zscore_60d"] = calculate_z_score(
                df["rsi_14"], window=ZSCORE_WINDOWS_DAILY_SHORT[0]
            )
            df["rsi_14_percentile_252d"] = calculate_percentile_rank(
                df["rsi_14"], window=PERCENTILE_WINDOWS_DAILY[0]
            )
            df["macd_hist_roc_5d"] = calculate_roc(df["macd_histogram"], ROC_WINDOWS_DAILY[0])
            df["adx_level_signal"] = pd.cut(
                df["adx"], bins=[-np.inf, 20, 40, np.inf], labels=[0, 1, 2], right=False
            ).astype("Int64")
            df["stoch_k_oversold_signal"] = (df["stochastic_k"] < 20).astype(int)
            df["stoch_k_overbought_signal"] = (df["stochastic_k"] > 80).astype(int)

            df.reset_index(inplace=True)
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

            if "close_price" in df.columns:
                df_to_upsert = df.drop(columns=["close_price"])
            else:
                df_to_upsert = df.copy()

            self.db_manager.upsert_dataframe(
                df=df_to_upsert, table_name=table_name, conflict_columns=["symbol", "date"]
            )
            logger.info(f"Updated {table_name} with signals for {symbol_filter}.")
        except Exception as e:
            logger.error(f"Error processing {table_name}: {e}", exc_info=True)
        finally:
            session.close()

    def _process_volatility_indicators_signals(self):
        logger.info("Processing signals for volatility_indicators table...")
        table_name = "volatility_indicators"
        session = self.db_manager.get_session()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {table_name} ORDER BY date ASC;", session.bind, parse_dates=["date"]
            )
            if df.empty:
                logger.warning(f"No data in {table_name}.")
                return
            df.set_index("date", inplace=True)

            min_date = df.index.min().strftime("%Y-%m-%d") if not df.empty else "1900-01-01"
            max_date = df.index.max().strftime("%Y-%m-%d") if not df.empty else "2100-01-01"
            sp500_close = self._get_sp500_close_data(min_date, max_date)

            if "vix" in df.columns:
                df["vix_roc_21d"] = calculate_roc(df["vix"], ROC_WINDOWS_DAILY[1])
                df["vix_roc_63d"] = calculate_roc(df["vix"], ROC_WINDOWS_DAILY[2])
                df["vix_zscore_63d"] = calculate_z_score(df["vix"], ZSCORE_WINDOWS_DAILY_SHORT[1])
                df["vix_percentile_252d"] = calculate_percentile_rank(
                    df["vix"], PERCENTILE_WINDOWS_DAILY[0]
                )
                if "vix_sma_20" in df.columns:
                    df["vix_vs_sma20_signal"] = np.sign(df["vix"] - df["vix_sma_20"]).astype(
                        "Int64"
                    )
            if "vvix" in df.columns:
                df["vvix_roc_21d"] = calculate_roc(df["vvix"], ROC_WINDOWS_DAILY[1])
                df["vvix_zscore_63d"] = calculate_z_score(df["vvix"], ZSCORE_WINDOWS_DAILY_SHORT[1])
                df["vvix_percentile_252d"] = calculate_percentile_rank(
                    df["vvix"], PERCENTILE_WINDOWS_DAILY[0]
                )
            if "vix" in df.columns and "vvix" in df.columns:
                df["vix_vvix_ratio"] = df["vix"] / df["vvix"]
            if "skew_index" in df.columns:
                df["skew_index_zscore_63d"] = calculate_z_score(
                    df["skew_index"], ZSCORE_WINDOWS_DAILY_SHORT[1]
                )
                df["skew_index_percentile_252d"] = calculate_percentile_rank(
                    df["skew_index"], PERCENTILE_WINDOWS_DAILY[0]
                )
            if "atr" in df.columns and not sp500_close.empty:
                sp500_close_aligned = sp500_close.reindex(df.index, method="ffill")
                df["atr_pct_of_price"] = (df["atr"] / sp500_close_aligned) * 100

            df.replace([np.inf, -np.inf], pd.NA, inplace=True)
            df.reset_index(inplace=True)
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

            self.db_manager.upsert_dataframe(
                df=df, table_name=table_name, conflict_columns=["date"]
            )
            logger.info(f"Updated {table_name} with signals.")
        except Exception as e:
            logger.error(f"Error processing {table_name}: {e}", exc_info=True)
        finally:
            session.close()

    def _process_put_call_ratios_signals(self):
        logger.info("Processing signals for put_call_ratios table...")
        table_name = "put_call_ratios"
        session = self.db_manager.get_session()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {table_name} ORDER BY date ASC;", session.bind, parse_dates=["date"]
            )
            if df.empty:
                logger.warning(f"No data in {table_name}.")
                return
            df.set_index("date", inplace=True)

            pc_col = "equity_pc_ratio"
            if pc_col in df.columns:
                df["equity_pc_ratio_sma_5d"] = calculate_sma(df[pc_col], SMA_WINDOWS_DAILY[0])
                df["equity_pc_ratio_sma_21d"] = calculate_sma(df[pc_col], SMA_WINDOWS_DAILY[2])
                df["equity_pc_ratio_vs_sma21d_diff"] = df[pc_col] - df["equity_pc_ratio_sma_21d"]
                df["equity_pc_ratio_roc_5d"] = calculate_roc(df[pc_col], ROC_WINDOWS_DAILY[0])
                df["equity_pc_ratio_roc_21d"] = calculate_roc(df[pc_col], ROC_WINDOWS_DAILY[1])
                df["equity_pc_ratio_zscore_63d"] = calculate_z_score(
                    df[pc_col], ZSCORE_WINDOWS_DAILY_SHORT[1]
                )
                df["equity_pc_ratio_percentile_252d"] = calculate_percentile_rank(
                    df[pc_col], PERCENTILE_WINDOWS_DAILY[0]
                )

            df.replace([np.inf, -np.inf], pd.NA, inplace=True)
            df.reset_index(inplace=True)
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            self.db_manager.upsert_dataframe(
                df=df, table_name=table_name, conflict_columns=["date"]
            )
            logger.info(f"Updated {table_name} with signals.")
        except Exception as e:
            logger.error(f"Error processing {table_name}: {e}", exc_info=True)
        finally:
            session.close()

    def _process_index_breadth_signals(self):
        logger.info("Processing signals for index_breadth_indicators table...")
        table_name = "index_breadth_indicators"
        session = self.db_manager.get_session()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {table_name} ORDER BY date ASC;", session.bind, parse_dates=["date"]
            )
            if df.empty:
                logger.warning(f"No data in {table_name}.")
                return
            df.set_index("date", inplace=True)

            if "pct_above_sma50" in df.columns:
                df["pct_above_sma50_roc_21d"] = calculate_roc(
                    df["pct_above_sma50"], ROC_WINDOWS_DAILY[1]
                )
            if "pct_above_sma200" in df.columns:
                df["pct_above_sma200_roc_21d"] = calculate_roc(
                    df["pct_above_sma200"], ROC_WINDOWS_DAILY[1]
                )
            if "ad_line" in df.columns:
                df["ad_line_roc_21d"] = calculate_roc(df["ad_line"], ROC_WINDOWS_DAILY[1])
                ad_line_sma21 = calculate_sma(df["ad_line"], SMA_WINDOWS_DAILY[2])
                df["ad_line_sma_21d_diff"] = df["ad_line"] - ad_line_sma21
            if "ad_ratio" in df.columns:
                df["ad_ratio_sma_5d"] = calculate_sma(df["ad_ratio"], SMA_WINDOWS_DAILY[0])

            df.replace([np.inf, -np.inf], pd.NA, inplace=True)
            df.reset_index(inplace=True)
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                df["created_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

            try:
                self.db_manager.upsert_dataframe(
                    df=df, table_name=table_name, conflict_columns=["date"]
                )
                logger.info(f"Updated {table_name} with signals using upsert.")
            except (SQLAlchemyOperationalError, SQLiteOperationalError) as oe_upsert:
                if (
                    "on conflict clause does not match any primary key or unique constraint"
                    in str(oe_upsert).lower()
                ):
                    logger.warning(
                        f"Upsert failed for {table_name} due to missing PK/UNIQUE on 'date'. "
                        "Falling back to if_exists='replace'. PRAGMA info was "
                        "logged on previous attempt."
                    )
                    df.to_sql(
                        table_name,
                        self.db_manager.engine,
                        if_exists="replace",
                        index=False,
                        dtype={
                            "date": sqlalchemy.types.Date,
                            "pct_above_sma50": sqlalchemy.types.Float,
                            "ad_line": sqlalchemy.types.Float,
                        },
                    )
                    logger.info(f"Replaced {table_name} with signals due to PK/UNIQUE issue.")
                else:
                    raise

        except Exception as e:
            logger.error(f"Error processing {table_name}: {e}", exc_info=True)
        finally:
            if session:
                session.close()

    def _process_cnn_fear_greed_signals(self):
        logger.info("Processing signals for cnn_fear_greed_index table...")
        table_name = "cnn_fear_greed_index"
        session = self.db_manager.get_session()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {table_name} ORDER BY reference_date ASC;",
                session.bind,
                parse_dates=["reference_date", "release_date"],
            )
            if df.empty:
                logger.warning(f"No data in {table_name}.")
                return
            df.set_index("reference_date", inplace=True)

            value_col = "value"
            if value_col in df.columns:
                df["fg_value_sma_5d"] = calculate_sma(df[value_col], SMA_WINDOWS_DAILY[0])
                df["fg_value_sma_21d"] = calculate_sma(df[value_col], SMA_WINDOWS_DAILY[2])
                df["fg_value_vs_sma21d_diff"] = df[value_col] - df["fg_value_sma_21d"]
                df["fg_value_roc_5d"] = calculate_roc(df[value_col], ROC_WINDOWS_DAILY[0])
                df["is_extreme_fear_signal"] = (df[value_col] < 25).astype(int)
                df["is_extreme_greed_signal"] = (df[value_col] > 75).astype(int)
                df["fg_value_percentile_252d"] = calculate_percentile_rank(
                    df[value_col], PERCENTILE_WINDOWS_DAILY[0]
                )

            df.replace([np.inf, -np.inf], pd.NA, inplace=True)
            df.reset_index(inplace=True)
            df["reference_date"] = df["reference_date"].dt.strftime("%Y-%m-%d")
            if "release_date" in df.columns:
                df["release_date"] = df["release_date"].dt.strftime("%Y-%m-%d")
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            self.db_manager.upsert_dataframe(
                df=df, table_name=table_name, conflict_columns=["reference_date"]
            )
            logger.info(f"Updated {table_name} with signals.")
        except Exception as e:
            logger.error(f"Error processing {table_name}: {e}", exc_info=True)
        finally:
            session.close()

    def _process_generic_vintage_signals(
        self,
        raw_table_name: str,
        signals_table_name: str,
        series_id_filter: str,
        value_col_name: str,
        series_id_col_name_in_raw: str,
        series_id_col_name_in_signal: str,
        data_frequency: str,
        expected_signal_cols: list[str],
    ):
        logger.info(
            f"Processing generic vintage signals for {raw_table_name} "
            f"(Series: {series_id_filter}) -> {signals_table_name}"
        )
        session = self.db_manager.get_session()
        try:
            query = (
                f"SELECT reference_date, release_date,"
                f" {value_col_name},"
                f" {series_id_col_name_in_raw}"
                f" FROM {raw_table_name}"
                f" WHERE {series_id_col_name_in_raw} = '{series_id_filter}'"
                " ORDER BY reference_date ASC, release_date ASC;"
            )
            raw_df = pd.read_sql_query(
                query, session.bind, parse_dates=["reference_date", "release_date"]
            )

            if raw_df.empty:
                logger.warning(f"No raw data for {series_id_filter} in {raw_table_name}")
                return

            pit_df = self._get_point_in_time_series(
                raw_df,
                value_col_names=[value_col_name],
                series_id_col_name=series_id_col_name_in_raw,
            )
            if pit_df.empty:
                logger.warning(f"No PIT data for {series_id_filter} from {raw_table_name}")
                return

            pit_df_indexed = pit_df.set_index("reference_date").sort_index()

            value_pit_col = f"{value_col_name}_pit"

            signals_df = pd.DataFrame(index=pit_df_indexed.index)
            signals_df[series_id_col_name_in_signal] = series_id_filter
            signals_df["value_pit"] = pit_df_indexed[value_pit_col]
            signals_df["release_date"] = pit_df_indexed["release_date"].dt.strftime("%Y-%m-%d")

            periods_in_year = {"D": 252, "W": 52, "M": 12}[data_frequency.upper()]
            periods_1m = {"D": 21, "W": 4, "M": 1}[data_frequency.upper()]
            periods_3m = {"D": 63, "W": 13, "M": 3}[data_frequency.upper()]
            periods_6m = {"D": 126, "W": 26, "M": 6}[data_frequency.upper()]

            z_win_suffix = (
                f"{ZSCORE_WINDOWS_ECO_M[0]}m"
                if data_frequency == "M"
                else f"{ZSCORE_WINDOWS_ECO_W[0]}w"
            )
            p_win_suffix = (
                f"{PERCENTILE_WINDOWS_ECO_M[0]}m"
                if data_frequency == "M"
                else f"{PERCENTILE_WINDOWS_ECO_W[0]}w"
            )
            sma_yoy_win_suffix = (
                f"{SMA_WINDOWS_ECO_M[2]}m" if data_frequency == "M" else f"{SMA_WINDOWS_ECO_W[2]}w"
            )

            z_win_val = (
                ZSCORE_WINDOWS_ECO_M[0] if data_frequency == "M" else ZSCORE_WINDOWS_ECO_W[0]
            )
            p_win_val = (
                PERCENTILE_WINDOWS_ECO_M[0]
                if data_frequency == "M"
                else PERCENTILE_WINDOWS_ECO_W[0]
            )
            sma_yoy_win_val = (
                SMA_WINDOWS_ECO_M[2] if data_frequency == "M" else SMA_WINDOWS_ECO_W[2]
            )

            if "yoy_change" in expected_signal_cols and len(pit_df_indexed) > periods_in_year:
                signals_df["yoy_change"] = pit_df_indexed[value_pit_col].pct_change(
                    periods=periods_in_year
                )

            if (
                "mom_change" in expected_signal_cols
                and data_frequency.upper() == "M"
                and len(pit_df_indexed) > periods_1m
            ):
                signals_df["mom_change"] = pit_df_indexed[value_pit_col].pct_change(
                    periods=periods_1m
                )
            elif (
                "change_1wk" in expected_signal_cols
                and data_frequency.upper() == "W"
                and len(pit_df_indexed) > periods_1m
            ):
                signals_df["change_1wk"] = pit_df_indexed[value_pit_col].pct_change(
                    periods=periods_1m
                )

            if "change_3m_annualized" in expected_signal_cols:
                val_3periods_ago = pit_df_indexed[value_pit_col].shift(periods_3m)
                if not val_3periods_ago.isnull().all().all():
                    signals_df["change_3m_annualized"] = (
                        (pit_df_indexed[value_pit_col] / val_3periods_ago)
                        ** (periods_in_year / periods_3m)
                    ) - 1

            if "change_6m_annualized" in expected_signal_cols and data_frequency.upper() == "M":
                val_6periods_ago = pit_df_indexed[value_pit_col].shift(periods_6m)
                if not val_6periods_ago.isnull().all().all():
                    signals_df["change_6m_annualized"] = (
                        (pit_df_indexed[value_pit_col] / val_6periods_ago)
                        ** (periods_in_year / periods_6m)
                    ) - 1

            if (
                f"yoy_change_zscore_{z_win_suffix}" in expected_signal_cols
                and "yoy_change" in signals_df
            ):
                signals_df[f"yoy_change_zscore_{z_win_suffix}"] = calculate_z_score(
                    signals_df["yoy_change"], window=z_win_val
                )

            if (
                f"yoy_change_percentile_{p_win_suffix}" in expected_signal_cols
                and "yoy_change" in signals_df
            ):
                signals_df[f"yoy_change_percentile_{p_win_suffix}"] = calculate_percentile_rank(
                    signals_df["yoy_change"], window=p_win_val
                )

            if (
                signals_table_name == "non_farm_payrolls_signals"
                and f"yoy_change_vs_sma{sma_yoy_win_suffix}_diff" in expected_signal_cols
                and "yoy_change" in signals_df
            ):
                yoy_sma = calculate_sma(signals_df["yoy_change"], window=sma_yoy_win_val)
                signals_df[f"yoy_change_vs_sma{sma_yoy_win_suffix}_diff"] = (
                    signals_df["yoy_change"] - yoy_sma
                )

            if signals_table_name == "initial_jobless_claims_signals":
                if "value_pit_4wk_ma" in expected_signal_cols:
                    signals_df["value_pit_4wk_ma"] = calculate_sma(
                        pit_df_indexed[value_pit_col], window=SMA_WINDOWS_ECO_W[0]
                    )
                if (
                    "change_4wk_ma_yoy" in expected_signal_cols
                    and len(pit_df_indexed) > periods_in_year
                    and "value_pit_4wk_ma" in signals_df
                ):
                    signals_df["change_4wk_ma_yoy"] = signals_df["value_pit_4wk_ma"].pct_change(
                        periods=periods_in_year
                    )
                    z_wk_win_z = ZSCORE_WINDOWS_ECO_W[0]
                    p_wk_win_p = PERCENTILE_WINDOWS_ECO_W[1]
                    if f"change_4wk_ma_zscore_{z_wk_win_z}wk" in expected_signal_cols:
                        signals_df[f"change_4wk_ma_zscore_{z_wk_win_z}wk"] = calculate_z_score(
                            signals_df["change_4wk_ma_yoy"], window=z_wk_win_z
                        )
                    if f"change_4wk_ma_percentile_{p_wk_win_p}wk" in expected_signal_cols:
                        signals_df[
                            f"change_4wk_ma_percentile_{p_wk_win_p}wk"
                        ] = calculate_percentile_rank(
                            signals_df["change_4wk_ma_yoy"], window=p_wk_win_p
                        )

            if (
                signals_table_name == "housing_starts_signals"
                and "change_3m_sma" in expected_signal_cols
            ):
                signals_df["change_3m_sma"] = calculate_sma(
                    pit_df_indexed[value_pit_col], window=SMA_WINDOWS_ECO_M[0]
                )

            if signals_table_name == "consumer_confidence_signals":
                if "value_mom_diff" in expected_signal_cols:
                    signals_df["value_mom_diff"] = pit_df_indexed[value_pit_col].diff(
                        periods=periods_1m
                    )
                if "value_yoy_diff" in expected_signal_cols:
                    signals_df["value_yoy_diff"] = pit_df_indexed[value_pit_col].diff(
                        periods=periods_in_year
                    )
                if "value_sma_3m" in expected_signal_cols:
                    signals_df["value_sma_3m"] = calculate_sma(
                        pit_df_indexed[value_pit_col], window=SMA_WINDOWS_ECO_M[0]
                    )
                if "value_sma_6m" in expected_signal_cols:
                    signals_df["value_sma_6m"] = calculate_sma(
                        pit_df_indexed[value_pit_col], window=SMA_WINDOWS_ECO_M[1]
                    )

                z_conf_suffix = f"{ZSCORE_WINDOWS_ECO_M[0]}m"
                p_conf_suffix = f"{PERCENTILE_WINDOWS_ECO_M[0]}m"
                if f"value_zscore_{z_conf_suffix}" in expected_signal_cols:
                    signals_df[f"value_zscore_{z_conf_suffix}"] = calculate_z_score(
                        pit_df_indexed[value_pit_col], window=ZSCORE_WINDOWS_ECO_M[0]
                    )
                if f"value_percentile_{p_conf_suffix}" in expected_signal_cols:
                    signals_df[f"value_percentile_{p_conf_suffix}"] = calculate_percentile_rank(
                        pit_df_indexed[value_pit_col], window=PERCENTILE_WINDOWS_ECO_M[0]
                    )

            signals_df.replace([np.inf, -np.inf], pd.NA, inplace=True)

            signals_df_final = signals_df.reset_index()
            if "reference_date" not in signals_df_final.columns:
                logger.error(
                    f"CRITICAL: 'reference_date' missing after reset_index for {signals_table_name}"
                )
                return

            cols_to_select = [
                col for col in expected_signal_cols if col in signals_df_final.columns
            ]
            if "reference_date" not in cols_to_select:
                cols_to_select.insert(0, "reference_date")

            signals_df_to_upsert = signals_df_final[list(dict.fromkeys(cols_to_select))].copy()

            signals_df_to_upsert["reference_date"] = pd.to_datetime(
                signals_df_to_upsert["reference_date"]
            ).dt.strftime("%Y-%m-%d")
            if "release_date" in signals_df_to_upsert.columns:
                signals_df_to_upsert["release_date"] = pd.to_datetime(
                    signals_df_to_upsert["release_date"]
                ).dt.strftime("%Y-%m-%d")

            calculated_signal_cols = [
                col
                for col in cols_to_select
                if col
                not in ["reference_date", series_id_col_name_in_signal, "value_pit", "release_date"]
            ]
            if calculated_signal_cols:
                signals_df_to_upsert.dropna(subset=calculated_signal_cols, how="all", inplace=True)

            if signals_df_to_upsert.empty:
                logger.info(
                    f"No signals to upsert for {signals_table_name} "
                    f"for {series_id_filter} after NaN drop."
                )
                return

            self.db_manager.upsert_dataframe(
                df=signals_df_to_upsert,
                table_name=signals_table_name,
                conflict_columns=["reference_date", series_id_col_name_in_signal],
            )
            logger.info(f"Upserted signals to {signals_table_name} for {series_id_filter}.")

        except Exception as e:
            logger.error(
                f"Error processing {raw_table_name} for {series_id_filter}: {e}", exc_info=True
            )
        finally:
            if session:
                session.close()

    def _process_aaii_sentiment_signals(self):
        logger.info("Processing AAII sentiment signals...")
        raw_table_name = "aaii_sentiment"
        signals_table_name = "aaii_sentiment_signals"
        session = self.db_manager.get_session()
        try:
            query = f"SELECT * FROM {raw_table_name} ORDER BY reference_date ASC, release_date ASC;"
            raw_df = pd.read_sql_query(
                query, session.bind, parse_dates=["reference_date", "release_date"]
            )
            if raw_df.empty:
                logger.warning(f"No data in {raw_table_name}.")
                return

            value_cols_to_pit = [
                "bullish_pct",
                "neutral_pct",
                "bearish_pct",
                "bull_bear_spread_pct",
                "bullish_8wk_ma_pct",
            ]
            pit_df = self._get_point_in_time_series(raw_df, value_col_names=value_cols_to_pit)
            if pit_df.empty:
                logger.warning(f"No PIT data from {raw_table_name}.")
                return

            pit_df.set_index("reference_date", inplace=True)
            pit_df.sort_index(inplace=True)

            signals_df = pd.DataFrame(index=pit_df.index)
            signals_df["release_date"] = pit_df["release_date"].dt.strftime("%Y-%m-%d")
            for col in value_cols_to_pit:
                signals_df[f"{col}_pit"] = pit_df[f"{col}_pit"]

            bb_spread_col = "bull_bear_spread_pct_pit"
            bullish_col = "bullish_pct_pit"
            bearish_col = "bearish_pct_pit"

            if bb_spread_col in signals_df.columns:
                signals_df["bull_bear_spread_pct_sma_4wk"] = calculate_sma(
                    signals_df[bb_spread_col], window=SMA_WINDOWS_ECO_W[0]
                )
                signals_df["bull_bear_spread_pct_zscore_52wk"] = calculate_z_score(
                    signals_df[bb_spread_col], window=ZSCORE_WINDOWS_ECO_W[0]
                )
            if bullish_col in signals_df.columns:
                signals_df["bullish_pct_percentile_52wk"] = calculate_percentile_rank(
                    signals_df[bullish_col], window=PERCENTILE_WINDOWS_ECO_W[0]
                )
            if bearish_col in signals_df.columns:
                signals_df["bearish_pct_percentile_52wk"] = calculate_percentile_rank(
                    signals_df[bearish_col], window=PERCENTILE_WINDOWS_ECO_W[0]
                )

            signals_df.replace([np.inf, -np.inf], pd.NA, inplace=True)

            calculated_cols = [
                "bull_bear_spread_pct_sma_4wk",
                "bull_bear_spread_pct_zscore_52wk",
                "bullish_pct_percentile_52wk",
                "bearish_pct_percentile_52wk",
            ]
            actual_calculated_cols = [col for col in calculated_cols if col in signals_df.columns]
            if actual_calculated_cols:
                signals_df.dropna(subset=actual_calculated_cols, how="all", inplace=True)

            signals_df.reset_index(inplace=True)
            signals_df["reference_date"] = signals_df["reference_date"].dt.strftime("%Y-%m-%d")

            expected_cols = [
                "reference_date",
                "release_date",
                "bullish_pct_pit",
                "neutral_pct_pit",
                "bearish_pct_pit",
                "bull_bear_spread_pct_pit",
                "bullish_8wk_ma_pct_pit",
                "bull_bear_spread_pct_sma_4wk",
                "bull_bear_spread_pct_zscore_52wk",
                "bullish_pct_percentile_52wk",
                "bearish_pct_percentile_52wk",
            ]
            final_df_cols = [col for col in expected_cols if col in signals_df.columns]
            signals_df_to_upsert = signals_df[final_df_cols]

            if signals_df_to_upsert.empty:
                logger.info(f"No signals to upsert for {signals_table_name} after NaN drop.")
                return

            self.db_manager.upsert_dataframe(
                df=signals_df_to_upsert,
                table_name=signals_table_name,
                conflict_columns=["reference_date"],
            )
            logger.info(f"Upserted signals to {signals_table_name}.")
        except Exception as e:
            logger.error(f"Error processing {raw_table_name}: {e}", exc_info=True)
        finally:
            session.close()

    def _process_finra_margin_signals(self):
        logger.info("Processing FINRA margin statistics signals...")
        raw_table_name = "finra_margin_statistics"
        signals_table_name = "finra_margin_statistics_signals"
        session = self.db_manager.get_session()
        try:
            query = f"SELECT * FROM {raw_table_name} ORDER BY reference_date ASC, release_date ASC;"
            raw_df = pd.read_sql_query(
                query, session.bind, parse_dates=["reference_date", "release_date"]
            )
            if raw_df.empty:
                logger.warning(f"No data in {raw_table_name}.")
                return

            value_cols = [
                "debit_balances_margin_accounts",
                "free_credit_cash_accounts",
                "free_credit_margin_accounts",
            ]
            pit_df = self._get_point_in_time_series(raw_df, value_col_names=value_cols)
            if pit_df.empty:
                logger.warning(f"No PIT data from {raw_table_name}.")
                return

            pit_df.set_index("reference_date", inplace=True)
            pit_df.sort_index(inplace=True)

            signals_df = pd.DataFrame(index=pit_df.index)
            signals_df["release_date"] = pit_df["release_date"].dt.strftime("%Y-%m-%d")
            signals_df["debit_balances_pit"] = pit_df[f"{value_cols[0]}_pit"]
            signals_df["free_credit_cash_pit"] = pit_df[f"{value_cols[1]}_pit"]
            signals_df["free_credit_margin_pit"] = pit_df[f"{value_cols[2]}_pit"]

            debit_col_pit = signals_df["debit_balances_pit"]
            signals_df["debit_balances_yoy_change"] = debit_col_pit.pct_change(periods=12)
            signals_df["debit_balances_roc_3m"] = calculate_roc(debit_col_pit, 3)
            signals_df["debit_balances_roc_6m"] = calculate_roc(debit_col_pit, 6)

            min_date = pit_df.index.min().strftime("%Y-%m-%d")
            max_date = pit_df.index.max().strftime("%Y-%m-%d")
            sp500_close = self._get_sp500_close_data(min_date, max_date)
            if not sp500_close.empty:
                sp500_monthly_close = (
                    sp500_close.resample("ME").last().reindex(pit_df.index, method="ffill")
                )
                signals_df["normalized_margin_debt"] = debit_col_pit / sp500_monthly_close
            else:
                signals_df["normalized_margin_debt"] = np.nan

            signals_df["margin_debt_vs_free_credit_ratio"] = debit_col_pit / (
                signals_df["free_credit_cash_pit"].fillna(0)
                + signals_df["free_credit_margin_pit"].fillna(0)
            )

            signals_df["debit_balances_yoy_change_zscore_24m"] = calculate_z_score(
                signals_df["debit_balances_yoy_change"], window=ZSCORE_WINDOWS_ECO_M[0]
            )

            signals_df.replace([np.inf, -np.inf], pd.NA, inplace=True)

            calculated_cols = [
                "debit_balances_yoy_change",
                "debit_balances_roc_3m",
                "debit_balances_roc_6m",
                "normalized_margin_debt",
                "margin_debt_vs_free_credit_ratio",
                "debit_balances_yoy_change_zscore_24m",
            ]
            actual_calculated_cols = [col for col in calculated_cols if col in signals_df.columns]
            if actual_calculated_cols:
                signals_df.dropna(subset=actual_calculated_cols, how="all", inplace=True)

            signals_df.reset_index(inplace=True)
            signals_df["reference_date"] = signals_df["reference_date"].dt.strftime("%Y-%m-%d")

            expected_cols = [
                "reference_date",
                "release_date",
                "debit_balances_pit",
                "free_credit_cash_pit",
                "free_credit_margin_pit",
                "debit_balances_yoy_change",
                "debit_balances_roc_3m",
                "debit_balances_roc_6m",
                "normalized_margin_debt",
                "margin_debt_vs_free_credit_ratio",
                "debit_balances_yoy_change_zscore_24m",
            ]
            final_df_cols = [col for col in expected_cols if col in signals_df.columns]
            signals_df_to_upsert = signals_df[final_df_cols]

            if signals_df_to_upsert.empty:
                logger.info(f"No signals to upsert for {signals_table_name} after NaN drop.")
                return

            self.db_manager.upsert_dataframe(
                df=signals_df_to_upsert,
                table_name=signals_table_name,
                conflict_columns=["reference_date"],
            )
            logger.info(f"Upserted signals to {signals_table_name}.")
        except Exception as e:
            logger.error(f"Error processing {raw_table_name}: {e}", exc_info=True)
        finally:
            session.close()

    def _process_bond_spread_signals(self):
        logger.info("Processing signals for investment_grade_junk_bond_yield_spread...")
        raw_table_name = "investment_grade_junk_bond_yield_spread"
        signals_table_name = "investment_grade_junk_bond_yield_spread_signals"
        session = self.db_manager.get_session()
        try:
            query = f"""SELECT reference_date,
             release_date,
              spread_value,
               high_yield_series_id,
                investment_grade_series_id
                FROM {raw_table_name} ORDER BY high_yield_series_id,
                 investment_grade_series_id,
                  reference_date ASC,
                   release_date ASC;"""
            raw_df = pd.read_sql_query(
                query, session.bind, parse_dates=["reference_date", "release_date"]
            )
            if raw_df.empty:
                logger.warning(f"No data in {raw_table_name}.")
                return

            all_signal_dfs = []
            for group_keys_tuple, group_df_for_series in raw_df.groupby(
                ["high_yield_series_id", "investment_grade_series_id"]
            ):
                current_hy_id, current_ig_id = group_keys_tuple

                pit_df = self._get_point_in_time_series(
                    group_df_for_series,
                    value_col_names=["spread_value"],
                    extra_groupby_cols=None,
                )
                if pit_df.empty:
                    logger.warning(f"No PIT data for spread {current_hy_id}/{current_ig_id}.")
                    continue

                pit_df.set_index("reference_date", inplace=True)
                pit_df.sort_index(inplace=True)

                signals_df = pd.DataFrame(index=pit_df.index)
                signals_df["high_yield_series_id"] = current_hy_id
                signals_df["investment_grade_series_id"] = current_ig_id
                signals_df["release_date"] = pit_df["release_date"].dt.strftime("%Y-%m-%d")
                signals_df["spread_value_pit"] = pit_df["spread_value_pit"]

                sv_pit = signals_df["spread_value_pit"]
                signals_df["spread_value_sma_21d"] = calculate_sma(sv_pit, SMA_WINDOWS_DAILY[2])
                signals_df["spread_value_sma_63d"] = calculate_sma(sv_pit, SMA_WINDOWS_DAILY[4])
                signals_df["spread_value_vs_sma63d_diff"] = (
                    sv_pit - signals_df["spread_value_sma_63d"]
                )
                signals_df["spread_value_roc_21d"] = calculate_roc(sv_pit, ROC_WINDOWS_DAILY[1])
                signals_df["spread_value_roc_63d"] = calculate_roc(sv_pit, ROC_WINDOWS_DAILY[2])
                signals_df["spread_value_zscore_252d"] = calculate_z_score(
                    sv_pit, ZSCORE_WINDOWS_DAILY_LONG[0]
                )
                signals_df["spread_value_percentile_252d"] = calculate_percentile_rank(
                    sv_pit, PERCENTILE_WINDOWS_DAILY[0]
                )

                roc5d = calculate_roc(sv_pit, ROC_WINDOWS_DAILY[0])
                sma5d = calculate_sma(sv_pit, SMA_WINDOWS_DAILY[0])
                signals_df["spread_widening_signal_5d"] = ((sv_pit > sma5d) & (roc5d > 0)).astype(
                    "Int64"
                )

                signals_df.replace([np.inf, -np.inf], pd.NA, inplace=True)

                calculated_cols = [
                    "spread_value_sma_21d",
                    "spread_value_sma_63d",
                    "spread_value_vs_sma63d_diff",
                    "spread_value_roc_21d",
                    "spread_value_roc_63d",
                    "spread_value_zscore_252d",
                    "spread_value_percentile_252d",
                    "spread_widening_signal_5d",
                ]
                actual_calculated_cols = [
                    col for col in calculated_cols if col in signals_df.columns
                ]
                if actual_calculated_cols:
                    signals_df.dropna(subset=actual_calculated_cols, how="all", inplace=True)

                if not signals_df.empty:
                    all_signal_dfs.append(signals_df.reset_index())

            if not all_signal_dfs:
                logger.warning(f"No signals calculated for {raw_table_name}.")
                return

            final_signals_df = pd.concat(all_signal_dfs)
            if final_signals_df.empty:
                logger.warning(f"Final signals df is empty for {raw_table_name}.")
                return

            final_signals_df["reference_date"] = pd.to_datetime(
                final_signals_df["reference_date"]
            ).dt.strftime("%Y-%m-%d")

            self.db_manager.upsert_dataframe(
                df=final_signals_df,
                table_name=signals_table_name,
                conflict_columns=[
                    "reference_date",
                    "high_yield_series_id",
                    "investment_grade_series_id",
                ],
            )
            logger.info(f"Upserted signals to {signals_table_name}.")
        except Exception as e:
            logger.error(f"Error processing {raw_table_name}: {e}", exc_info=True)
        finally:
            session.close()

    def run_all_processing(self, schema_updates_sql_file: Path | None = None):
        if schema_updates_sql_file and schema_updates_sql_file.exists():
            logger.info("Applying schema updates for new signals...")
            self._apply_sql_file(schema_updates_sql_file)
        else:
            logger.warning(
                "Schema update file not provided or does not "
                f"exist: {schema_updates_sql_file}. Skipping DDL execution."
            )

        logger.info("--- Starting Category 1 Table Processing (Modifying Existing Tables) ---")
        self._process_technical_indicators_signals()
        self._process_volatility_indicators_signals()
        self._process_put_call_ratios_signals()
        self._process_index_breadth_signals()
        self._process_cnn_fear_greed_signals()

        logger.info("--- Starting Category 2 Table Processing (Creating New _signals Tables) ---")
        economic_indicators_configs = [
            {
                "raw": "non_farm_payrolls",
                "signals": "non_farm_payrolls_signals",
                "id": "PAYEMS",
                "val_col": "value",
                "id_col_raw": "series_id",
                "id_col_sig": "series_id",
                "freq": "M",
                "expected_cols": [
                    "reference_date",
                    "series_id",
                    "value_pit",
                    "release_date",
                    "yoy_change",
                    "mom_change",
                    "change_3m_annualized",
                    "change_6m_annualized",
                    "yoy_change_zscore_24m",
                    "yoy_change_percentile_36m",
                    "yoy_change_vs_sma12m_diff",
                ],
            },
            {
                "raw": "initial_jobless_claims",
                "signals": "initial_jobless_claims_signals",
                "id": "ICSA",
                "val_col": "value",
                "id_col_raw": "series_id",
                "id_col_sig": "series_id",
                "freq": "W",
                "expected_cols": [
                    "reference_date",
                    "series_id",
                    "value_pit",
                    "release_date",
                    "value_pit_4wk_ma",
                    "change_1wk",
                    "change_4wk_ma_yoy",
                    "change_4wk_ma_zscore_52wk",
                    "change_4wk_ma_percentile_104wk",
                ],
            },
            {
                "raw": "cpi",
                "signals": "cpi_signals",
                "id": "CPIAUCSL",
                "val_col": "value",
                "id_col_raw": "series_id",
                "id_col_sig": "series_id",
                "freq": "M",
                "expected_cols": [
                    "reference_date",
                    "series_id",
                    "value_pit",
                    "release_date",
                    "yoy_change",
                    "mom_change",
                    "change_3m_annualized",
                    "change_6m_annualized",
                    "yoy_change_zscore_24m",
                    "yoy_change_percentile_36m",
                ],
            },
            {
                "raw": "retail_sales",
                "signals": "retail_sales_signals",
                "id": "RSAFS",
                "val_col": "value",
                "id_col_raw": "series_id",
                "id_col_sig": "series_id",
                "freq": "M",
                "expected_cols": [
                    "reference_date",
                    "series_id",
                    "value_pit",
                    "release_date",
                    "yoy_change",
                    "mom_change",
                    "change_3m_annualized",
                    "yoy_change_zscore_24m",
                    "yoy_change_percentile_36m",
                ],
            },
            {
                "raw": "m2_money_supply",
                "signals": "m2_money_supply_signals",
                "id": "M2SL",
                "val_col": "value",
                "id_col_raw": "series_id",
                "id_col_sig": "series_id",
                "freq": "M",
                "expected_cols": [
                    "reference_date",
                    "series_id",
                    "value_pit",
                    "release_date",
                    "yoy_change",
                    "mom_change",
                    "change_3m_annualized",
                    "yoy_change_zscore_24m",
                    "yoy_change_percentile_60m",
                ],
            },
            {
                "raw": "housing_starts",
                "signals": "housing_starts_signals",
                "id": "HOUST",
                "val_col": "value",
                "id_col_raw": "series_id",
                "id_col_sig": "series_id",
                "freq": "M",
                "expected_cols": [
                    "reference_date",
                    "series_id",
                    "value_pit",
                    "release_date",
                    "yoy_change",
                    "mom_change",
                    "change_3m_sma",
                    "yoy_change_zscore_24m",
                ],
            },
            {
                "raw": "housing_prices",
                "signals": "housing_prices_signals",
                "id": "CSUSHPINSA",
                "val_col": "value",
                "id_col_raw": "series_id",
                "id_col_sig": "series_id",
                "freq": "M",
                "expected_cols": [
                    "reference_date",
                    "series_id",
                    "value_pit",
                    "release_date",
                    "yoy_change",
                    "mom_change",
                    "yoy_change_zscore_36m",
                    "yoy_change_percentile_60m",
                ],
            },
            {
                "raw": "consumer_confidence",
                "signals": "consumer_confidence_signals",
                "id": "UMCSENT",
                "val_col": "value",
                "id_col_raw": "series_id",
                "id_col_sig": "series_id",
                "freq": "M",
                "expected_cols": [
                    "reference_date",
                    "series_id",
                    "value_pit",
                    "release_date",
                    "value_mom_diff",
                    "value_yoy_diff",
                    "value_sma_3m",
                    "value_sma_6m",
                    "value_zscore_24m",
                    "value_percentile_36m",
                ],
            },
        ]
        for config in economic_indicators_configs:
            self._process_generic_vintage_signals(
                raw_table_name=config["raw"],
                signals_table_name=config["signals"],
                series_id_filter=config["id"],
                value_col_name=config["val_col"],
                series_id_col_name_in_raw=config["id_col_raw"],
                series_id_col_name_in_signal=config["id_col_sig"],
                data_frequency=config["freq"],
                expected_signal_cols=config["expected_cols"],
            )

        self._process_aaii_sentiment_signals()
        self._process_finra_margin_signals()
        self._process_bond_spread_signals()

        logger.info("Master signal orchestration complete.")
