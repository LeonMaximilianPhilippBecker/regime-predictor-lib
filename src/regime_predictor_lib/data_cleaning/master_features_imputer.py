import logging
from pathlib import Path

import pandas as pd
import sqlalchemy
import yaml

from regime_predictor_lib.utils.database_manager import DatabaseManager

imputer_logger = logging.getLogger("MasterFeaturesImputer")

CONCRETE_BFILL_PATTERNS = ["_signal", "_is_extreme_fear_signal", "_is_extreme_greed_signal"]
CONCRETE_NO_INTERP_PATTERNS = ["_ref_date"]
LOW_CARDINALITY_THRESHOLD = 15


class MasterFeaturesImputer:
    def __init__(
        self,
        db_manager: DatabaseManager,
        analysis_yaml_path: Path | str,
        imputation_log_file_path: Path | str | None = None,
    ):
        self.db_manager = db_manager
        self.analysis_yaml_path = Path(analysis_yaml_path)
        self.master_df = None
        self.column_analysis_data = None
        self.imputation_log_file_path = Path(imputation_log_file_path) if imputation_log_file_path else None

        if self.imputation_log_file_path:
            self.imputation_log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.imputation_log_file_path, mode="w")
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            imputer_logger.addHandler(file_handler)
            imputer_logger.setLevel(logging.INFO)

        imputer_logger.info(f"MasterFeaturesImputer initialized. Analysis YAML: {self.analysis_yaml_path}")
        if self.imputation_log_file_path:
            imputer_logger.info(f"Detailed imputation log will be saved to: {self.imputation_log_file_path}")

    def _load_dependencies(self) -> bool:
        imputer_logger.info(f"Loading column analysis YAML from: {self.analysis_yaml_path}")
        if not self.analysis_yaml_path.exists():
            imputer_logger.error(f"Analysis YAML file not found: {self.analysis_yaml_path}")
            return False
        try:
            with open(self.analysis_yaml_path, "r") as f:
                self.column_analysis_data = yaml.safe_load(f)
            if not isinstance(self.column_analysis_data, list):
                imputer_logger.error("Analysis YAML content is not a list.")
                return False
        except Exception as e:
            imputer_logger.error(f"Error loading or parsing analysis YAML: {e}", exc_info=True)
            return False
        imputer_logger.info("Column analysis YAML loaded successfully.")

        imputer_logger.info("Loading master_features table...")
        try:
            query = "SELECT * FROM master_features ORDER BY date ASC"
            self.master_df = pd.read_sql_query(
                sql=sqlalchemy.text(query),
                con=self.db_manager.engine,
                parse_dates=["date"],
            )
            if "date" in self.master_df.columns:
                self.master_df["date"] = pd.to_datetime(self.master_df["date"])
                self.master_df.set_index("date", inplace=True)

            if not isinstance(self.master_df.index, pd.DatetimeIndex):
                imputer_logger.error("Master features table must have a DatetimeIndex named 'date'.")
                return False

            imputer_logger.info(f"Loaded master_features table with shape: {self.master_df.shape}")
            return True
        except Exception as e:
            imputer_logger.error(f"Error loading master_features table: {e}", exc_info=True)
            return False

    def _get_column_imputation_type(self, col_name: str, series: pd.Series) -> str:
        for pattern in CONCRETE_NO_INTERP_PATTERNS:
            if col_name.endswith(pattern):
                return "concrete_no_interp"
        for pattern in CONCRETE_BFILL_PATTERNS:
            if col_name.endswith(pattern):
                return "concrete_bfill_like"

        if series.dropna().nunique() < LOW_CARDINALITY_THRESHOLD:
            imputer_logger.debug(
                f"Column '{col_name}' classified as 'concrete_bfill_like' "
                f"due to low cardinality ({series.dropna().nunique()} unique values)."
            )
            return "concrete_bfill_like"

        if pd.api.types.is_float_dtype(series.dtype):
            return "flexible_interpolate"
        if pd.api.types.is_integer_dtype(series.dtype):
            return "flexible_interpolate"
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            return "concrete_no_interp"

        imputer_logger.warning(
            f"Column {col_name} has unknown dtype {series.dtype} for imputation. Will not impute."
        )
        return "unknown"

    def drop_all_nan_columns(self):
        if self.master_df is None or self.column_analysis_data is None:
            imputer_logger.error("Data or analysis YAML not loaded. Run _load_dependencies first.")
            return

        cols_to_drop = [item["column_name"] for item in self.column_analysis_data if item["is_all_nan"]]

        if cols_to_drop:
            imputer_logger.info(f"Dropping all-NaN columns: {cols_to_drop}")
            self.master_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
            imputer_logger.info(f"Shape after dropping all-NaN columns: {self.master_df.shape}")
        else:
            imputer_logger.info("No all-NaN columns found to drop.")

    def impute_extraordinary_nans(self):
        if self.master_df is None or self.column_analysis_data is None:
            imputer_logger.error("Data or analysis YAML not loaded for imputing extraordinary NaNs.")
            return

        imputer_logger.info("Starting targeted imputation of 'extraordinary' NaNs...")
        columns_with_extra_nans_info = [
            item
            for item in self.column_analysis_data
            if item.get("has_extraordinary_nans") and item.get("num_extraordinary_nans", 0) > 0
        ]

        if not columns_with_extra_nans_info:
            imputer_logger.info("No columns identified with extraordinary NaNs to impute specifically.")
            return

        for col_info in columns_with_extra_nans_info:
            col_name = col_info["column_name"]
            if col_name not in self.master_df.columns:
                imputer_logger.debug(
                    f"Column {col_name} not present in DataFrame (possibly dropped). "
                    "Skipping extraordinary NaN imputation."
                )
                continue

            imputer_logger.info(f"Processing extraordinary NaNs for column: {col_name}")
            series = self.master_df[col_name]
            imputation_type = self._get_column_imputation_type(col_name, series)

            if imputation_type in ["concrete_no_interp", "unknown"]:
                imputer_logger.info(
                    f"Skipping extraordinary NaN imputation for {col_name} due to its type: {imputation_type}"
                )
                continue

            nan_dates_str = col_info.get("extraordinary_nan_dates", [])
            if not nan_dates_str:
                imputer_logger.info(f"No extraordinary NaN dates listed for {col_name} in analysis. Skipping.")
                continue

            nan_indices_dt = pd.to_datetime(nan_dates_str)
            imputed_count_for_col = 0

            for nan_idx_dt in nan_indices_dt:
                if nan_idx_dt not in series.index or not pd.isna(series.loc[nan_idx_dt]):
                    if nan_idx_dt in series.index:
                        imputer_logger.debug(
                            f"Skipping extraordinary NaN for {col_name} "
                            f"at {nan_idx_dt.strftime('%Y-%m-%d')}: "
                            f"already has value {series.loc[nan_idx_dt]}."
                        )
                    continue

                prev_valid_series = series.loc[: nan_idx_dt - pd.Timedelta(days=1)].dropna()
                prev_idx = prev_valid_series.last_valid_index() if not prev_valid_series.empty else None

                next_valid_series = series.loc[nan_idx_dt + pd.Timedelta(days=1) :].dropna()
                next_idx = next_valid_series.first_valid_index() if not next_valid_series.empty else None

                if pd.notna(prev_idx) and pd.notna(next_idx):
                    left_val = series.loc[prev_idx]
                    right_val = series.loc[next_idx]
                    imputed_value = None

                    if imputation_type == "concrete_bfill_like":
                        imputed_value = right_val
                        self.master_df.loc[nan_idx_dt, col_name] = imputed_value
                        imputer_logger.info(
                            f"IMPUTED (Extraordinary): Column '{col_name}', "
                            f"Date '{nan_idx_dt.strftime('%Y-%m-%d')}'. "
                            f"Strategy: '{imputation_type}'. Filled NaN with {imputed_value} "
                            f"(from next valid date: {next_idx.strftime('%Y-%m-%d')}, value: {right_val}). "
                            f"Previous valid was: {left_val} (on {prev_idx.strftime('%Y-%m-%d')})."
                        )
                        imputed_count_for_col += 1
                    elif imputation_type == "flexible_interpolate":
                        if pd.api.types.is_integer_dtype(series.dtype) and not series.isnull().any():
                            interpolated_val = (float(left_val) + float(right_val)) / 2.0
                        else:
                            interpolated_val = (left_val + right_val) / 2.0
                        imputed_value = interpolated_val
                        self.master_df.loc[nan_idx_dt, col_name] = imputed_value
                        imputer_logger.info(
                            f"IMPUTED (Extraordinary): Column '{col_name}', "
                            f"Date '{nan_idx_dt.strftime('%Y-%m-%d')}'. "
                            f"Strategy: '{imputation_type}'. Filled NaN with {imputed_value:.4f}. "
                            f"Interpolated using prev: {left_val:.4f} (on {prev_idx.strftime('%Y-%m-%d')}) and "
                            f"next: {right_val:.4f} (on {next_idx.strftime('%Y-%m-%d')})."
                        )
                        imputed_count_for_col += 1
                else:
                    imputer_logger.debug(
                        f"SKIPPED IMPUTATION (Extraordinary) for {col_name} "
                        f"at {nan_idx_dt.strftime('%Y-%m-%d')}: "
                        f"Not sandwiched. Prev: {prev_idx.strftime('%Y-%m-%d') if prev_idx else 'None'}, "
                        f"Next: {next_idx.strftime('%Y-%m-%d') if next_idx else 'None'}."
                    )
            if imputed_count_for_col > 0:
                imputer_logger.info(
                    f"Finished column '{col_name}'. Total extraordinary NaNs "
                    f"imputed: {imputed_count_for_col} using '{imputation_type}'."
                )
            else:
                imputer_logger.info(
                    f"Finished column '{col_name}'. No extraordinary "
                    "NaNs imputed this run (check criteria/sandwiching)."
                )
        imputer_logger.info("Completed targeted imputation of 'extraordinary' NaNs.")

    def forward_fill_all_eligible_columns(self):
        if self.master_df is None:
            imputer_logger.error("DataFrame not loaded. Cannot perform forward fill.")
            return

        imputer_logger.info("Starting aggressive forward-fill for eligible columns...")
        for col_name in self.master_df.columns:
            series = self.master_df[col_name]
            imputation_type = self._get_column_imputation_type(col_name, series)

            if imputation_type in ["concrete_no_interp", "unknown"]:
                imputer_logger.debug(
                    f"Skipping forward-fill for column '{col_name}' (type: {imputation_type})."
                )
                continue

            nan_before_ffill = series.isnull().sum()
            if nan_before_ffill == 0:
                imputer_logger.debug(f"Column '{col_name}' has no NaNs to forward-fill.")
                continue

            self.master_df[col_name] = series.ffill()
            nan_after_ffill = self.master_df[col_name].isnull().sum()
            filled_count = nan_before_ffill - nan_after_ffill

            if filled_count > 0:
                imputer_logger.info(
                    f"FORWARD-FILLED: Column '{col_name}'. Filled {filled_count} NaN(s). "
                    f"Remaining NaNs: {nan_after_ffill}."
                )
            elif nan_before_ffill > 0:
                imputer_logger.info(
                    f"FORWARD-FILL ATTEMPTED: Column '{col_name}'. No NaNs filled (likely all leading NaNs). "
                    f"Original NaNs: {nan_before_ffill}."
                )

        imputer_logger.info("Completed aggressive forward-fill.")

    def update_database(self):
        if self.master_df is None:
            imputer_logger.error("No data to update in the database.")
            return

        imputer_logger.info("Preparing to update 'master_features' table in the database.")
        df_to_save = self.master_df.reset_index()

        if "date" in df_to_save.columns and pd.api.types.is_datetime64_any_dtype(df_to_save["date"]):
            df_to_save["date"] = df_to_save["date"].dt.strftime("%Y-%m-%d")

        try:
            imputer_logger.info(f"Replacing data in 'master_features' table. Shape: {df_to_save.shape}")
            df_to_save.to_sql(
                "master_features",
                self.db_manager.engine,
                if_exists="replace",
                index=False,
                chunksize=1000,
            )
            imputer_logger.info("'master_features' table updated successfully.")
        except Exception as e:
            imputer_logger.error(f"Error updating 'master_features' table in database: {e}", exc_info=True)

    def run_imputation_workflow(self):
        imputer_logger.info("Starting Master Features Imputation Workflow...")
        if not self._load_dependencies():
            imputer_logger.error("Failed to load dependencies. Aborting imputation workflow.")
            return

        self.drop_all_nan_columns()
        self.impute_extraordinary_nans()
        self.forward_fill_all_eligible_columns()

        self.update_database()
        imputer_logger.info("Master features imputation workflow completed.")
