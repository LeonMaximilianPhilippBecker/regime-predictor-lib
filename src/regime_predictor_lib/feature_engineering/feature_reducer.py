import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from regime_predictor_lib.utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class FeatureReducer:
    def __init__(
        self,
        db_manager: DatabaseManager,
        output_dir: Path | str,
        mutual_info_target_col: str = "regime_t_plus_6m",
        correlation_threshold: float = 0.9,
        vif_threshold: float = 7.5,
        random_state_mi: int = 42,
    ):
        self.db_manager = db_manager
        self.output_dir = Path(output_dir)
        self.mutual_info_target_col = mutual_info_target_col
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.random_state_mi = random_state_mi

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"ThematicFeatureReducer initialized. Output dir: {self.output_dir}, "
            f"MI Target: {self.mutual_info_target_col}, "
            f"Corr threshold: {self.correlation_threshold}, VIF threshold: {self.vif_threshold}"
        )

    def _load_thematic_table(self, table_name: str) -> pd.DataFrame | None:
        logger.debug(f"Loading thematic table: {table_name}")
        try:
            df = pd.read_sql_table(table_name, self.db_manager.engine, parse_dates=["date"])
            if "date" in df.columns:
                df = df.set_index("date")
            df = df.sort_index()
            logger.info(f"Loaded table '{table_name}' with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading table {table_name}: {e}", exc_info=True)
            return None

    def _get_stats_slice(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> tuple[pd.DataFrame | None, pd.Timestamp | None]:
        if df.empty or not feature_cols:
            return None, None

        first_valid_indices = []
        for col in feature_cols:
            if col in df.columns:
                fvi = df[col].first_valid_index()
                if fvi is not None:
                    first_valid_indices.append(fvi)

        if not first_valid_indices:
            logger.warning("No valid data found in any feature columns. Cannot determine stats slice.")
            return None, None

        prefix_end_date = max(first_valid_indices)
        logger.info(f"Determined prefix end date for stats slice: {prefix_end_date.strftime('%Y-%m-%d')}")

        stats_df_slice = df[df.index > prefix_end_date].copy()

        if stats_df_slice.empty or len(stats_df_slice) < max(20, len(feature_cols) + 1):
            logger.warning(
                f"Stats slice is too small ({len(stats_df_slice)} rows) "
                "after prefix removal. May lead to unreliable stats."
            )
            if stats_df_slice.empty:
                return None, prefix_end_date

        for col in stats_df_slice.columns:
            if stats_df_slice[col].isnull().any():
                stats_df_slice[col] = stats_df_slice[col].interpolate(method="linear", limit_direction="both")
                if stats_df_slice[col].isnull().any():
                    stats_df_slice[col] = stats_df_slice[col].ffill().bfill()

        all_nan_cols = stats_df_slice.columns[stats_df_slice.isnull().all()].tolist()
        if all_nan_cols:
            logger.warning(
                f"Columns {all_nan_cols} are all NaN in the stats "
                "slice even after imputation attempt. They will be excluded from stats."
            )
            stats_df_slice = stats_df_slice.drop(columns=all_nan_cols)

        return stats_df_slice, prefix_end_date

    def _calculate_mutual_information(
        self, X_stats_slice: pd.DataFrame, y_series_stats_slice: pd.Series
    ) -> pd.Series:
        features_list = X_stats_slice.columns.tolist()
        logger.debug(f"Calculating mutual information for {len(features_list)} features using stats slice.")

        if X_stats_slice.empty or y_series_stats_slice.empty:
            logger.warning("Empty X or y for mutual information calculation from stats slice.")
            return pd.Series(dtype=float, index=features_list)

        X = X_stats_slice.copy()
        y = y_series_stats_slice.copy()

        if y.isnull().any():
            nan_count_y = y.isnull().sum()
            logger.warning(
                f"Target column '{y.name}' for MI (in stats slice) "
                f"has {nan_count_y} NaNs. Imputing with ffill then bfill."
            )
            y = y.ffill().bfill()
            if y.isnull().any():
                logger.error(
                    f"Target column '{y.name}' is all NaN even after ffill/bfill. Cannot calculate MI."
                )
                return pd.Series(dtype=float, index=features_list)

        if not pd.api.types.is_integer_dtype(y) and not pd.api.types.is_bool_dtype(y):
            try:
                y = y.astype(int)
            except ValueError as e:
                logger.error(
                    f"Cannot cast target {y.name} to int after NaN fill for MI: {e}. Returning empty scores."
                )
                return pd.Series(dtype=float, index=features_list)

        if X.isnull().values.any():
            logger.error(
                "Features X for MI still contain NaNs unexpectedly. "
                f"Problematic columns: {X.columns[X.isnull().any()].tolist()}. Returning empty scores."
            )
            return pd.Series(dtype=float, index=features_list)

        try:
            mi_values = mutual_info_classif(
                X, y, random_state=self.random_state_mi, n_neighbors=min(3, len(X) - 1) if len(X) > 1 else 1
            )
            mi_scores = pd.Series(mi_values, index=features_list)
            return mi_scores
        except Exception as e:
            logger.error(f"Error calculating mutual information: {e}", exc_info=True)
            return pd.Series(dtype=float, index=features_list)

    def _reduce_by_correlation(
        self,
        df_features_stats_slice: pd.DataFrame,
        features: list[str],
        mi_scores: pd.Series,
    ) -> list[str]:
        logger.info(
            f"Starting correlation-based reduction using stats slice. Initial features: {len(features)}, "
            f"Threshold: {self.correlation_threshold}"
        )
        if not features:
            return []

        corr_matrix = df_features_stats_slice.corr(
            method=lambda x, y: spearmanr(x, y, nan_policy="propagate").correlation
            if pd.notna(x).all() and pd.notna(y).all()
            else np.nan
        )

        features_to_keep = []
        processed_features = set()

        sorted_features_for_iteration = sorted(features, key=lambda f: mi_scores.get(f, -1), reverse=True)

        for feature1 in sorted_features_for_iteration:
            if feature1 in processed_features:
                continue

            correlated_group = {feature1}
            if feature1 in corr_matrix.columns:
                for feature2 in sorted_features_for_iteration:
                    if feature1 == feature2 or feature2 in processed_features:
                        continue
                    if feature2 in corr_matrix.columns and feature1 in corr_matrix.index:
                        correlation_value = corr_matrix.loc[feature1, feature2]
                        if pd.notna(correlation_value) and abs(correlation_value) > self.correlation_threshold:
                            correlated_group.add(feature2)

            best_feature_in_group = feature1
            max_mi = mi_scores.get(feature1, -1)

            for f_in_group in correlated_group:
                if mi_scores.get(f_in_group, -1) > max_mi:
                    max_mi = mi_scores.get(f_in_group)
                    best_feature_in_group = f_in_group

            features_to_keep.append(best_feature_in_group)
            features_dropped_from_group = correlated_group - {best_feature_in_group}
            if features_dropped_from_group:
                logger.debug(
                    f"Correlated group: {correlated_group}. "
                    f"Kept: {best_feature_in_group} (MI: {max_mi:.4f}). Dropped: {features_dropped_from_group}"
                )
            else:
                logger.debug(
                    f"Feature {best_feature_in_group} "
                    f"(MI: {max_mi:.4f}) had no highly correlated partners to drop or was best in group."
                )

            processed_features.update(correlated_group)

        logger.info(f"Features after correlation reduction: {len(features_to_keep)}")
        return features_to_keep

    def _reduce_by_vif(
        self,
        df_features_stats_slice: pd.DataFrame,
        features: list[str],
    ) -> list[str]:
        logger.info(
            f"Starting VIF-based reduction using stats slice. Initial features: {len(features)}, "
            f"Threshold: {self.vif_threshold}"
        )
        if not features or len(features) < 2:
            logger.info("Not enough features to perform VIF reduction.")
            return list(features)

        current_features = list(features)

        X_vif_base = df_features_stats_slice.copy()

        iteration = 0
        max_iterations = len(current_features) * 2

        while True:
            iteration += 1
            if iteration > max_iterations:
                logger.warning(f"VIF reduction exceeded max iterations ({max_iterations}). Stopping.")
                break

            if len(current_features) < 2:
                break

            X_vif_subset = X_vif_base[current_features].copy()

            if X_vif_subset.isnull().values.any():
                problematic_cols = X_vif_subset.columns[X_vif_subset.isnull().any()].tolist()
                logger.error(
                    f"FATAL: NaNs found in VIF subset unexpectedly for columns: {problematic_cols}. "
                    f"This should not happen after _get_stats_slice. Current features: {current_features}"
                )
                current_features = [cf for cf in current_features if cf not in problematic_cols]
                if len(current_features) < 2:
                    break
                continue

            if X_vif_subset.shape[0] < X_vif_subset.shape[1] + 1:
                logger.warning(
                    f"Not enough samples ({X_vif_subset.shape[0]}) relative to features"
                    f" ({X_vif_subset.shape[1]}) for VIF. Stopping at {len(current_features)} features."
                )
                break

            try:
                X_vif_with_const = add_constant(X_vif_subset, prepend=False, has_constant="raise")
            except ValueError as e:
                logger.error(
                    f"Error adding constant for VIF (iteration {iteration}, {len(current_features)} "
                    f"features): {e}. Features: {current_features}"
                )
                variances = X_vif_subset.var()
                if (variances == 0).any():
                    feature_to_drop_heuristic = variances[variances == 0].index[0]
                else:
                    feature_to_drop_heuristic = variances.idxmin()
                if feature_to_drop_heuristic in current_features:
                    logger.warning(
                        f"Heuristically dropping feature '{feature_to_drop_heuristic}' "
                        "due to add_constant error."
                    )
                    current_features.remove(feature_to_drop_heuristic)
                    continue
                else:  # Should not happen
                    logger.error("Could not identify a feature to drop heuristically. Breaking VIF loop.")
                    break

            vif_data = pd.DataFrame()
            vif_data["feature"] = [col for col in X_vif_with_const.columns if col != "const"]

            try:
                vif_values = [
                    variance_inflation_factor(X_vif_with_const.values, i)
                    for i, col_name in enumerate(X_vif_with_const.columns)
                    if col_name != "const"
                ]
                vif_data["VIF"] = vif_values
            except Exception as e:
                logger.error(
                    f"Error calculating VIF values (iteration {iteration}, {len(current_features)} "
                    f"features): {e}. Features: {vif_data['feature'].tolist()}"
                )
                if X_vif_subset.nunique().min() == 1 and len(current_features) > 1:
                    feature_to_drop_heuristic = X_vif_subset.nunique().idxmin()
                else:
                    feature_to_drop_heuristic = X_vif_subset.var(skipna=False).idxmin()

                if feature_to_drop_heuristic in current_features:
                    logger.warning(
                        f"Heuristically dropping feature '{feature_to_drop_heuristic}' "
                        f"due to VIF calculation error."
                    )
                    current_features.remove(feature_to_drop_heuristic)
                    continue
                else:
                    logger.error(
                        "Could not identify a feature to drop heuristically after VIF error. Breaking VIF loop."
                    )
                    break

            max_vif = vif_data["VIF"].max()
            if max_vif > self.vif_threshold:
                feature_to_drop = vif_data.sort_values("VIF", ascending=False)["feature"].iloc[0]
                if feature_to_drop in current_features:
                    current_features.remove(feature_to_drop)
                    logger.debug(f"Dropped feature '{feature_to_drop}' with VIF: {max_vif:.2f}")
                else:
                    logger.warning(
                        f"Attempted to drop {feature_to_drop} but it was not in current_features. "
                        f"Max VIF {max_vif:.2f}"
                    )
                    break
            else:
                break

        logger.info(f"Features after VIF reduction: {len(current_features)}")
        return current_features

    def _update_db_table(self, table_name: str, cols_to_keep: list[str]):
        logger.info(f"Updating table {table_name} to keep {len(cols_to_keep)} columns.")

        final_cols_ordered = []
        final_cols_ordered.append("date")

        essential_metadata_cols = [self.mutual_info_target_col, "regime_t_plus_6m"]
        if self.mutual_info_target_col == "regime_t":
            essential_metadata_cols = ["regime_t", "regime_t_plus_6m"]

        for col in essential_metadata_cols:
            if col not in final_cols_ordered and col != "date":
                final_cols_ordered.append(col)

        predictor_features_to_keep = [f for f in cols_to_keep if f not in final_cols_ordered]
        final_cols_ordered.extend(sorted(predictor_features_to_keep))

        final_cols_ordered = list(dict.fromkeys(final_cols_ordered))

        try:
            with self.db_manager.engine.connect() as connection:
                result = connection.execute(sqlalchemy.text(f"PRAGMA table_info({table_name});"))
                actual_table_cols = [row[1] for row in result.fetchall()]

                df_original = pd.read_sql_table(table_name, connection, parse_dates=["date"])

            if "date" in df_original.columns:
                df_original["date"] = pd.to_datetime(df_original["date"]).dt.strftime("%Y-%m-%d")

            columns_present_in_df = [
                col
                for col in final_cols_ordered
                if col in df_original.columns or (col == "date" and df_original.index.name == "date")
            ]

            if df_original.index.name == "date" and "date" not in columns_present_in_df:
                df_reduced = df_original.reset_index()[columns_present_in_df + ["date"]].copy()
                # Ensure date is first
                cols = df_reduced.columns.tolist()
                cols.insert(0, cols.pop(cols.index("date")))
                df_reduced = df_reduced[cols]
            else:
                df_reduced = df_original[columns_present_in_df].copy()

            if set(df_reduced.columns) == set(actual_table_cols):
                logger.info(f"No change in columns for table {table_name}. Skipping database write.")
                return

            df_reduced.to_sql(
                table_name,
                self.db_manager.engine,
                if_exists="replace",
                index=False,
                chunksize=1000,
            )
            logger.info(f"Successfully updated table '{table_name}' with {len(df_reduced.columns)} columns.")
        except Exception as e:
            logger.error(f"Error updating table {table_name}: {e}", exc_info=True)

    def process_thematic_table(self, table_name: str):
        logger.info(f"--- Processing multicollinearity for table: {table_name} ---")
        df_orig_indexed = self._load_thematic_table(table_name)

        if df_orig_indexed is None or df_orig_indexed.empty:
            logger.warning(f"Table '{table_name}' is empty or could not be loaded. Skipping.")
            return

        if self.mutual_info_target_col not in df_orig_indexed.columns:
            logger.error(f"Target column '{self.mutual_info_target_col}' not found in {table_name}. Skipping.")
            return

        df = df_orig_indexed.copy()

        cols_to_exclude_from_features = ["regime_t", self.mutual_info_target_col, "regime_t_plus_6m"]
        if "date" in df.columns:
            cols_to_exclude_from_features.append("date")

        original_feature_candidates = [col for col in df.columns if col not in cols_to_exclude_from_features]

        df_numeric_features_full = df[original_feature_candidates].select_dtypes(include=np.number)
        numeric_features_list = df_numeric_features_full.columns.tolist()

        if not numeric_features_list:
            logger.warning(f"No numeric features found in {table_name} to process. Skipping.")
            essential_cols_for_empty_features = ["date"]
            if self.mutual_info_target_col in df.columns:
                essential_cols_for_empty_features.append(self.mutual_info_target_col)
            if "regime_t_plus_6m" in df.columns and "regime_t_plus_6m" != self.mutual_info_target_col:
                essential_cols_for_empty_features.append("regime_t_plus_6m")
            if "regime_t" in df.columns and "regime_t" != self.mutual_info_target_col:
                essential_cols_for_empty_features.append("regime_t")

            self._update_db_table(table_name, list(set(essential_cols_for_empty_features)))
            selected_features_path = self.output_dir / f"{table_name}_selected_features.txt"
            with open(selected_features_path, "w") as f:
                pass
            logger.info(
                f"Saved empty selected predictor feature list for {table_name} to {selected_features_path}"
            )
            return

        logger.info(f"Identified {len(numeric_features_list)} numeric features in {table_name} for analysis.")

        all_cols_for_slice = numeric_features_list + [self.mutual_info_target_col]
        if "regime_t" not in all_cols_for_slice and "regime_t" in df.columns:
            all_cols_for_slice.append("regime_t")
        all_cols_for_slice = list(set(all_cols_for_slice))

        df_stats_slice, _ = self._get_stats_slice(df.copy(), all_cols_for_slice)

        if df_stats_slice is None or df_stats_slice.empty or len(df_stats_slice) < 2:
            logger.warning(
                f"Stats slice for {table_name} is empty or too small after NaN handling. "
                "MI and VIF cannot be reliably calculated. Keeping all numeric features."
            )
            mi_scores = pd.Series(dtype=float, index=numeric_features_list)
            features_after_corr = list(numeric_features_list)
            final_selected_features = list(numeric_features_list)
        else:
            X_mi_df_stats_slice = df_stats_slice[numeric_features_list].copy()
            y_mi_series_stats_slice = df_stats_slice[self.mutual_info_target_col].copy()

            if y_mi_series_stats_slice.isnull().all():
                logger.warning(
                    f"Target column {self.mutual_info_target_col} is all NaN in "
                    f"stats slice for {table_name}. MI scores will be zero."
                )
                mi_scores = pd.Series(0.0, index=numeric_features_list)
            elif y_mi_series_stats_slice.isnull().any():
                logger.warning(
                    f"Target column {self.mutual_info_target_col} for MI "
                    f"still has {y_mi_series_stats_slice.isnull().sum()} NaNs after "
                    "slice prep. Attempting ffill/bfill."
                )
                y_mi_series_stats_slice = y_mi_series_stats_slice.ffill().bfill()
                if y_mi_series_stats_slice.isnull().any():
                    logger.error(
                        f"Target column {self.mutual_info_target_col} is "
                        "still all NaN even after ffill/bfill for MI. Cannot calculate MI."
                    )
                    mi_scores = pd.Series(dtype=float, index=numeric_features_list)
                else:
                    mi_scores = self._calculate_mutual_information(X_mi_df_stats_slice, y_mi_series_stats_slice)
            else:
                mi_scores = self._calculate_mutual_information(X_mi_df_stats_slice, y_mi_series_stats_slice)

            df_numeric_features_stats_slice = df_stats_slice[numeric_features_list].copy()
            features_after_corr = self._reduce_by_correlation(
                df_numeric_features_stats_slice, numeric_features_list, mi_scores
            )

            final_selected_features = self._reduce_by_vif(df_numeric_features_stats_slice, features_after_corr)

        cols_to_keep_in_db_table = []
        if df_orig_indexed.index.name == "date":
            cols_to_keep_in_db_table.append("date")

        essential_metadata_cols = list({"regime_t", self.mutual_info_target_col, "regime_t_plus_6m"})
        for col in essential_metadata_cols:
            if col in df_orig_indexed.columns and col not in cols_to_keep_in_db_table:
                cols_to_keep_in_db_table.append(col)

        cols_to_keep_in_db_table.extend(final_selected_features)
        cols_to_keep_in_db_table = sorted(
            list(set(cols_to_keep_in_db_table)),
            key=lambda x: (x != "date", x not in essential_metadata_cols, x),
        )

        self._update_db_table(table_name, cols_to_keep_in_db_table)

        selected_features_path = self.output_dir / f"{table_name}_selected_features.txt"
        with open(selected_features_path, "w") as f:
            for feature in final_selected_features:
                f.write(f"{feature}\n")
        logger.info(
            f"Saved list of {len(final_selected_features)} selected "
            "predictor features for {table_name} to {selected_features_path}"
        )

    def process_all_thematic_tables(self):
        inspector = sqlalchemy.inspect(self.db_manager.engine)
        all_table_names_in_db = inspector.get_table_names()

        thematic_table_names = [name for name in all_table_names_in_db if name.startswith("theme_")]

        if not thematic_table_names:
            logger.warning(
                "No thematic tables (starting with 'theme_') found in the database. Nothing to process."
            )
            return

        logger.info(f"Found {len(thematic_table_names)} thematic tables to process: {thematic_table_names}")

        for table_name in thematic_table_names:
            self.process_thematic_table(table_name)

        logger.info("Finished processing all thematic tables.")
