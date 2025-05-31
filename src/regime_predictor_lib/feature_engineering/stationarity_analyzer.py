import logging
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
import sqlalchemy
from statsmodels.tsa.stattools import adfuller, kpss

from regime_predictor_lib.utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class StationarityAnalyzer:
    def __init__(self, db_manager: DatabaseManager, output_csv_path: Path | str):
        self.db_manager = db_manager
        self.output_csv_path = Path(output_csv_path)
        self.master_df = None
        logger.info(f"StationarityAnalyzer initialized. Output will be saved to: {self.output_csv_path}")

    def _load_master_features_table(self) -> bool:
        logger.info("Loading master_features table for stationarity analysis...")
        try:
            query = "SELECT * FROM master_features ORDER BY date ASC"
            self.master_df = pd.read_sql_query(
                sql=sqlalchemy.text(query),
                con=self.db_manager.engine,
                parse_dates=["date"],
            )
            if "date" in self.master_df.columns:
                self.master_df.set_index("date", inplace=True)
            else:
                logger.error("Master features table does not have a 'date' column after loading.")
                return False

            logger.info(f"Loaded master_features table with shape: {self.master_df.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading master_features table: {e}", exc_info=True)
            return False

    def _perform_adf_test(self, series: pd.Series, col_name: str) -> dict:
        results = {
            "adf_statistic": np.nan,
            "adf_p_value": np.nan,
            "adf_n_lags": np.nan,
            "adf_critical_1%": np.nan,
            "adf_critical_5%": np.nan,
            "adf_critical_10%": np.nan,
            "adf_is_stationary_0.05": None,
            "adf_error": None,
        }
        try:
            adf_result = adfuller(series, autolag="AIC")
            results["adf_statistic"] = adf_result[0]
            results["adf_p_value"] = adf_result[1]
            results["adf_n_lags"] = adf_result[2]
            crit_values = adf_result[4]
            results["adf_critical_1%"] = crit_values.get("1%")
            results["adf_critical_5%"] = crit_values.get("5%")
            results["adf_critical_10%"] = crit_values.get("10%")
            if pd.notna(results["adf_p_value"]):
                results["adf_is_stationary_0.05"] = bool(results["adf_p_value"] < 0.05)
        except Exception as e:
            logger.warning(f"ADF test failed for column {col_name}: {e}")
            results["adf_error"] = str(e)
        return results

    def _perform_kpss_test(self, series: pd.Series, col_name: str, regression_type="c") -> dict:
        results = {
            f"kpss_statistic_{regression_type}": np.nan,
            f"kpss_p_value_{regression_type}": np.nan,
            f"kpss_n_lags_{regression_type}": np.nan,
            f"kpss_critical_1%_{regression_type}": np.nan,
            f"kpss_critical_5%_{regression_type}": np.nan,
            f"kpss_critical_10%_{regression_type}": np.nan,
            f"kpss_is_stationary_0.05_{regression_type}": None,
            f"kpss_p_value_upper_bound_{regression_type}": False,
            f"kpss_p_value_lower_bound_{regression_type}": False,
            f"kpss_error_{regression_type}": None,
        }
        try:
            with catch_warnings(record=True) as ws:
                filterwarnings("always", category=UserWarning)
                kpss_result = kpss(series, regression=regression_type, nlags="auto")

            results[f"kpss_statistic_{regression_type}"] = kpss_result[0]
            results[f"kpss_p_value_{regression_type}"] = kpss_result[1]
            results[f"kpss_n_lags_{regression_type}"] = kpss_result[2]
            crit_values = kpss_result[3]
            results[f"kpss_critical_1%_{regression_type}"] = crit_values.get("1%")
            results[f"kpss_critical_5%_{regression_type}"] = crit_values.get("5%")
            results[f"kpss_critical_10%_{regression_type}"] = crit_values.get("10%")

            p_val = results[f"kpss_p_value_{regression_type}"]
            if pd.notna(p_val):
                results[f"kpss_is_stationary_0.05_{regression_type}"] = bool(p_val >= 0.05)

            for w in ws:
                if issubclass(w.category, UserWarning):
                    msg = str(w.message).lower()
                    if "the actual p-value is greater than the p-value returned" in msg:
                        results[f"kpss_p_value_upper_bound_{regression_type}"] = True
                    elif "the actual p-value is smaller than the p-value returned" in msg:
                        results[f"kpss_p_value_lower_bound_{regression_type}"] = True
        except Exception as e:
            logger.warning(f"KPSS test ({regression_type}) failed for column {col_name}: {e}")
            results[f"kpss_error_{regression_type}"] = str(e)
        return results

    def _interpret_stationarity(
        self, adf_p: float, kpss_p: float, kpss_p_upper: bool, kpss_p_lower: bool
    ) -> str:
        adf_stationary = pd.notna(adf_p) and adf_p < 0.05

        actual_kpss_p = kpss_p
        if pd.notna(kpss_p):
            if kpss_p_lower:
                actual_kpss_p = kpss_p - 0.001
            elif kpss_p_upper:
                actual_kpss_p = kpss_p + 0.001

        kpss_stationary = pd.notna(actual_kpss_p) and actual_kpss_p >= 0.05

        if pd.isna(adf_p) or pd.isna(kpss_p):
            return "Error in tests"
        if adf_stationary and kpss_stationary:
            return "Stationary"
        if not adf_stationary and not kpss_stationary:
            return "Non-Stationary"
        if adf_stationary and not kpss_stationary:
            return "Difference Stationary (ADF Stationary, KPSS Non-Stationary)"
        if not adf_stationary and kpss_stationary:
            return "Trend Stationary (ADF Non-Stationary, KPSS Stationary)"
        return "Inconclusive"

    def analyze_stationarity(self) -> pd.DataFrame | None:
        if self.master_df is None:
            if not self._load_master_features_table():
                logger.error("Failed to load master_features table. Aborting stationarity analysis.")
                return None

        all_results = []

        numeric_cols = self.master_df.select_dtypes(include=np.number).columns.tolist()
        cols_to_exclude = ["regime_t", "regime_t_plus_6m"]
        cols_to_test = [col for col in numeric_cols if col not in cols_to_exclude]

        logger.info(f"Will attempt stationarity tests on {len(cols_to_test)} numeric columns.")

        for col_name in cols_to_test:
            if col_name.endswith("_ref_date"):
                logger.debug(f"Skipping ref_date column: {col_name}")
                continue

            logger.info(f"Analyzing stationarity for column: {col_name}")
            series = self.master_df[col_name].copy().dropna()

            if len(series) < 20:
                logger.warning(
                    f"Skipping column {col_name} due to insufficient data points after dropna: {len(series)}"
                )
                results_row = {"column_name": col_name, "error": "Insufficient data points"}
                all_results.append(results_row)
                continue

            if series.nunique() <= 1:
                logger.warning(f"Skipping column {col_name} as it is constant or near-constant.")
                results_row = {"column_name": col_name, "error": "Constant series"}
                all_results.append(results_row)
                continue

            adf_results = self._perform_adf_test(series, col_name)
            kpss_results_c = self._perform_kpss_test(series, col_name, regression_type="c")

            overall_interpretation = self._interpret_stationarity(
                adf_results["adf_p_value"],
                kpss_results_c["kpss_p_value_c"],
                kpss_results_c["kpss_p_value_upper_bound_c"],
                kpss_results_c["kpss_p_value_lower_bound_c"],
            )

            results_row = {
                "column_name": col_name,
                **adf_results,
                **kpss_results_c,
                "overall_stationarity_0.05": overall_interpretation,
            }
            all_results.append(results_row)

        if not all_results:
            logger.warning("No stationarity results generated.")
            return None

        results_df = pd.DataFrame(all_results)
        return results_df

    def run_analysis_and_save(self):
        results_df = self.analyze_stationarity()

        if results_df is not None and not results_df.empty:
            try:
                self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(self.output_csv_path, index=False)
                logger.info(f"Stationarity analysis results saved to: {self.output_csv_path}")
            except Exception as e:
                logger.error(f"Error saving stationarity results to CSV: {e}", exc_info=True)
        else:
            logger.warning("No stationarity results to save.")
