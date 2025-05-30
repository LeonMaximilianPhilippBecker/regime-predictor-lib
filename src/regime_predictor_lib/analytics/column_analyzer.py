import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy
import yaml

from regime_predictor_lib.utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

PREFIX_THEME_TABLE_MAP = {
    "ti_gspc_": {
        "table": "technical_indicators",
        "theme": "1. Simple Technical Trend and Momentum Signals",
    },
    "vol_": {
        "table": "volatility_indicators",
        "theme": "2. Volatility and Market Stress Indicators",
    },
    "pcr_": {
        "table": "put_call_ratios",
        "theme": "3. Market Internals (Under the Hood Checkup)",
    },
    "breadth_": {
        "table": "index_breadth_indicators",
        "theme": "3. Market Internals (Under the Hood Checkup)",
    },
    "stk_bond_diff_": {
        "table": "intermarket_stock_bond_returns_diff",
        "theme": "4. Intermarket Relationships",
    },
    "spy_tlt_ratio_": {
        "table": "intermarket_ratios",
        "theme": "4. Intermarket Relationships",
    },
    "gold_silver_ratio_": {
        "table": "intermarket_ratios",
        "theme": "4. Intermarket Relationships",
    },
    "copper_gold_ratio_": {
        "table": "intermarket_ratios",
        "theme": "4. Intermarket Relationships",
    },
    "junk_spread_": {
        "table": "investment_grade_junk_bond_yield_spread_signals",
        "theme": "5. Credit and Bond Market Tells",
    },
    "corp_oas_": {
        "table": "corporate_bond_oas",
        "theme": "5. Credit and Bond Market Tells",
    },
    "tsy_spread_": {
        "table": "treasury_yield_spreads",
        "theme": "5. Credit and Bond Market Tells",
    },
    "em_tbill_spread_": {
        "table": "em_corporate_vs_tbill_spread",
        "theme": "5. Credit and Bond Market Tells",
    },
    "fg_": {
        "table": "cnn_fear_greed_index",
        "theme": "6. Sentiment and Behavior Gauges",
    },
    "conf_": {
        "table": "consumer_confidence_signals",
        "theme": "6. Sentiment and Behavior Gauges",
    },
    "aaii_": {
        "table": "aaii_sentiment_signals",
        "theme": "6. Sentiment and Behavior Gauges",
    },
    "finra_": {
        "table": "finra_margin_statistics_signals",
        "theme": "6. Sentiment and Behavior Gauges",
    },
    "sentconf_": {
        "table": "sentiment_confidence_indices",
        "theme": "6. Sentiment and Behavior Gauges",
    },
    "nfp_": {
        "table": "non_farm_payrolls_signals",
        "theme": "7. Macro Economic Data and Forecasts",
    },
    "icj_": {
        "table": "initial_jobless_claims_signals",
        "theme": "7. Macro Economic Data and Forecasts",
    },
    "cpi_": {"table": "cpi_signals", "theme": "7. Macro Economic Data and Forecasts"},
    "retail_": {
        "table": "retail_sales_signals",
        "theme": "7. Macro Economic Data and Forecasts",
    },
    "m2_": {
        "table": "m2_money_supply_signals",
        "theme": "7. Macro Economic Data and Forecasts",
    },
    "houst_": {
        "table": "housing_starts_signals",
        "theme": "7. Macro Economic Data and Forecasts",
    },
    "hpi_": {
        "table": "housing_prices_signals",
        "theme": "7. Macro Economic Data and Forecasts",
    },
    "smi_": {
        "table": "smart_money_index",
        "theme": "8. Market Structure & Fund Flow Indicators",
    },
    "djt_vs_gspc_": {
        "table": "relative_strength_metrics",
        "theme": "9. Sector and Micro-Market Tells",
    },
    "rut_vs_gspc_": {
        "table": "relative_strength_metrics",
        "theme": "9. Sector and Micro-Market Tells",
    },
    "qqq_vs_dju_": {
        "table": "relative_strength_metrics",
        "theme": "9. Sector and Micro-Market Tells",
    },
    "xlv_vs_gspc_": {
        "table": "relative_strength_metrics",
        "theme": "9. Sector and Micro-Market Tells",
    },
    "dxy_": {
        "table": "dxy_signals",
        "theme": "10. Global Markets and Currency Signals",
    },
    "em_": {
        "table": "em_equity_signals",
        "theme": "10. Global Markets and Currency Signals",
    },
    "oil_": {
        "table": "oil_price_signals",
        "theme": "10. Global Markets and Currency Signals",
    },
    "bdi_": {
        "table": "bdi_signals",
        "theme": "10. Global Markets and Currency Signals",
    },
    "gex_": {
        "table": "gex_signals",
        "theme": "11. Advanced Derivative-Based Metrics",
    },
    "sp500_": {
        "table": "sp500_derived_indicators",
        "theme": "0. S&P 500 Base Market Data & Derived Features",
    },
    "regime_": {
        "table": "hmm_regime_output",
        "theme": "Market Regime Information",
    },
}


class ColumnAnalyzer:
    def __init__(self, db_manager: DatabaseManager, output_yaml_path: Path | str):
        self.db_manager = db_manager
        self.output_yaml_path = Path(output_yaml_path)
        self.master_df = None

    def _load_master_features_table(self) -> bool:
        logger.info("Loading master_features table...")
        try:
            query = "SELECT * FROM master_features ORDER BY date ASC"
            self.master_df = pd.read_sql_query(
                sql=sqlalchemy.text(query),
                con=self.db_manager.engine,
                parse_dates=["date"],
            )
            self.master_df.set_index("date", inplace=True)
            logger.info(f"Loaded master_features table with shape: {self.master_df.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading master_features table: {e}", exc_info=True)
            return False

    def _infer_source_and_theme(self, column_name: str) -> tuple[str, str]:
        for prefix, info in PREFIX_THEME_TABLE_MAP.items():
            if column_name.startswith(prefix):
                return info["table"], info["theme"]
        if column_name in ["regime_t", "regime_t_plus_6m"]:
            return PREFIX_THEME_TABLE_MAP["regime_"]["table"], PREFIX_THEME_TABLE_MAP["regime_"]["theme"]
        return "Unknown", "Unknown"

    def _determine_cadence(self, series: pd.Series) -> tuple[str, int | None]:
        valid_series = series.dropna()
        if len(valid_series) < 2:
            return "Insufficient data", None

        if not isinstance(valid_series.index, pd.DatetimeIndex):
            logger.warning(f"Series index is not DatetimeIndex for cadence calculation. Column: {series.name}")
            return "Invalid Index Type", None
        if not valid_series.index.is_monotonic_increasing:
            valid_series = valid_series.sort_index()

        diffs_days_series = valid_series.index.to_series().diff().dropna()
        if diffs_days_series.empty:
            return "Single data point after diff", None

        diffs_days = diffs_days_series.dt.days

        if diffs_days.empty:
            return "Single data point or all same date", None

        mode_days_series = diffs_days.mode()
        if mode_days_series.empty:
            median_days = diffs_days.median()
            if pd.notna(median_days):
                common_delta_days = int(round(median_days))
                logger.debug(f"Cadence mode failed for {series.name}, using median: {common_delta_days} days")
            else:
                logger.warning(f"Cadence mode and median failed for {series.name}")
                return "Irregular (no clear mode or median)", None
        else:
            common_delta_days = int(mode_days_series.iloc[0])

        if 0 <= common_delta_days <= 3:
            business_days_in_span = len(pd.bdate_range(valid_series.index.min(), valid_series.index.max()))
            actual_valid_points_on_bdays = len(valid_series[valid_series.index.dayofweek < 5])

            if (
                business_days_in_span > 0
                and (actual_valid_points_on_bdays / len(valid_series)) > 0.8
                and (len(valid_series) / business_days_in_span) > 0.6
            ):
                return "Daily (Business)", 1
            return f"Daily-like (Mode: {common_delta_days} days)", common_delta_days
        elif 5 <= common_delta_days <= 9:
            return "Weekly", common_delta_days
        elif 18 <= common_delta_days <= 24:
            return "Monthly (Business Days approx)", common_delta_days
        elif 25 <= common_delta_days <= 35:
            return "Monthly (Calendar approx)", common_delta_days
        elif 58 <= common_delta_days <= 70:
            return "Quarterly (Business Days approx)", common_delta_days
        elif 80 <= common_delta_days <= 100:
            return "Quarterly (Calendar approx)", common_delta_days
        else:
            return f"Irregular (Mode: {common_delta_days} days)", common_delta_days

    def _find_extraordinary_nans(
        self,
        series: pd.Series,
        first_valid_date: pd.Timestamp,
        estimated_cadence_str: str,
    ) -> dict:
        results = {
            "has_extraordinary_nans": False,
            "extraordinary_nan_dates": [],
            "extraordinary_nan_timedeltas_days": [],
            "num_extraordinary_nans": 0,
        }

        if pd.isna(first_valid_date):
            return results

        last_valid_date = series.last_valid_index()
        if pd.isna(last_valid_date) or first_valid_date >= last_valid_date:
            return results

        series_in_span = series.loc[first_valid_date:last_valid_date]

        internal_nans_indices = series_in_span[series_in_span.isnull()].index

        extraordinary_nan_dates_ts = []

        if "Daily (Business)" in estimated_cadence_str:
            full_bday_index_for_series = pd.bdate_range(start=series.index.min(), end=series.index.max())
            relevant_bdays = full_bday_index_for_series[
                (full_bday_index_for_series >= first_valid_date)
                & (full_bday_index_for_series <= last_valid_date)
            ]

            for bday in relevant_bdays:
                if bday in series.index:
                    if pd.isna(series.loc[bday]):
                        extraordinary_nan_dates_ts.append(bday)
        elif not internal_nans_indices.empty:
            extraordinary_nan_dates_ts = internal_nans_indices.tolist()

        if extraordinary_nan_dates_ts:
            extraordinary_nan_dates_ts = sorted(list(set(extraordinary_nan_dates_ts)))
            results["has_extraordinary_nans"] = True
            results["extraordinary_nan_dates"] = [d.strftime("%Y-%m-%d") for d in extraordinary_nan_dates_ts]
            results["num_extraordinary_nans"] = len(extraordinary_nan_dates_ts)

            if len(extraordinary_nan_dates_ts) > 1:
                dt_index_for_diff = pd.DatetimeIndex(extraordinary_nan_dates_ts)
                timedeltas_np = np.diff(dt_index_for_diff)
                results["extraordinary_nan_timedeltas_days"] = [
                    int(td / np.timedelta64(1, "D")) for td in timedeltas_np
                ]

        return results

    def analyze_columns(self) -> list[dict] | None:
        if self.master_df is None:
            if not self._load_master_features_table():
                return None

        all_column_data = []

        for col_name in self.master_df.columns:
            logger.info(f"Analyzing column: {col_name}")
            series = self.master_df[col_name].copy()

            nan_count = int(series.isnull().sum())
            is_all_nan = bool(series.isnull().all())

            source_table, theme = self._infer_source_and_theme(col_name)

            first_valid_date_val_ts = series.first_valid_index()
            first_valid_date_str = (
                first_valid_date_val_ts.strftime("%Y-%m-%d") if pd.notna(first_valid_date_val_ts) else None
            )

            cadence_str, common_delta_days = self._determine_cadence(series)

            extra_nans_info = self._find_extraordinary_nans(
                series, first_valid_date_val_ts, cadence_str, common_delta_days
            )

            col_data = {
                "column_name": col_name,
                "nan_count": nan_count,
                "is_all_nan": is_all_nan,
                "source_table": source_table,
                "theme": theme,
                "first_valid_data_date": first_valid_date_str,
                "general_cadence": cadence_str,
                "typical_cadence_days": common_delta_days if pd.notna(common_delta_days) else "N/A",
                "has_extraordinary_nans": extra_nans_info["has_extraordinary_nans"],
                "extraordinary_nan_dates": extra_nans_info["extraordinary_nan_dates"],
                "extraordinary_nan_timedeltas_days": extra_nans_info["extraordinary_nan_timedeltas_days"],
                "num_extraordinary_nans": extra_nans_info["num_extraordinary_nans"],
            }
            all_column_data.append(col_data)

        return all_column_data

    def run_analysis_and_save(self):
        analysis_results = self.analyze_columns()
        if analysis_results:
            self.output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.output_yaml_path, "w") as f:
                    yaml.dump(analysis_results, f, sort_keys=False, allow_unicode=True, width=120, indent=2)
                logger.info(f"Column analysis saved to: {self.output_yaml_path}")
            except Exception as e:
                logger.error(f"Error saving analysis to YAML: {e}", exc_info=True)
        else:
            logger.error("Column analysis failed or produced no results.")
