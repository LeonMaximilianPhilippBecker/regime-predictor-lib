import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import sqlalchemy
import yaml

from regime_predictor_lib.utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

ESSENTIAL_COLUMNS = ["date", "regime_t", "regime_t_plus_6m"]
THEMES_TO_EXCLUDE_AS_FEATURE_TABLES = [
    "Market Regime Information",
    "Unknown",
    "0. S&P 500 Base Market Data & Derived Features",
]


def sanitize_table_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"^\d+\.?\s*_?\s*", "", name)
    name = re.sub(r"s&p 500", "sandp500", name)
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return f"theme_{name}"


class ThematicFeatureTableCreator:
    def __init__(
        self,
        db_manager: DatabaseManager,
        master_features_table_name: str,
        column_analysis_yaml_path: Path,
    ):
        self.db_manager = db_manager
        self.master_features_table_name = master_features_table_name
        self.column_analysis_yaml_path = column_analysis_yaml_path
        self.master_df = None
        self.column_analysis_data = None
        logger.info("ThematicFeatureTableCreator initialized.")

    def _load_dependencies(self) -> bool:
        logger.info(f"Loading '{self.master_features_table_name}' table...")
        try:
            query = f"SELECT * FROM {self.master_features_table_name} ORDER BY date ASC"
            self.master_df = pd.read_sql_query(
                sql=sqlalchemy.text(query),
                con=self.db_manager.engine,
                parse_dates=["date"],
            )
            if "date" not in self.master_df.columns:
                logger.error("Master features table must have a 'date' column.")
                return False
            logger.info(f"Loaded '{self.master_features_table_name}' table with shape: {self.master_df.shape}")
        except Exception as e:
            logger.error(f"Error loading '{self.master_features_table_name}' table: {e}", exc_info=True)
            return False

        logger.info(f"Loading column analysis YAML from: {self.column_analysis_yaml_path}")
        if not self.column_analysis_yaml_path.exists():
            logger.error(f"Column analysis YAML file not found: {self.column_analysis_yaml_path}")
            return False
        try:
            with open(self.column_analysis_yaml_path, "r") as f:
                self.column_analysis_data = yaml.safe_load(f)
            if not isinstance(self.column_analysis_data, list):
                logger.error("Column analysis YAML content is not a list.")
                return False
        except Exception as e:
            logger.error(f"Error loading or parsing column analysis YAML: {e}", exc_info=True)
            return False
        logger.info("Column analysis YAML loaded successfully.")
        return True

    def _get_theme_column_mapping(self) -> defaultdict[str, list[str]]:
        theme_to_columns = defaultdict(list)
        if not self.column_analysis_data or self.master_df is None or self.master_df.empty:
            logger.warning("Column analysis data or master_df not available for theme mapping.")
            return theme_to_columns

        master_df_cols_set = set(self.master_df.columns)

        for item in self.column_analysis_data:
            column_name = item.get("column_name")
            theme = item.get("theme")
            if column_name and theme and theme not in THEMES_TO_EXCLUDE_AS_FEATURE_TABLES:
                if column_name in master_df_cols_set:
                    theme_to_columns[theme].append(column_name)
                else:
                    logger.debug(
                        f"Column '{column_name}' from YAML not found in current master_features table. "
                        f"Skipping for thematic tables."
                    )
        return theme_to_columns

    def generate_tables(self):
        if not self._load_dependencies():
            logger.error("Failed to load dependencies. Aborting thematic table creation.")
            return

        theme_to_columns_map = self._get_theme_column_mapping()
        if not theme_to_columns_map:
            logger.warning("No themes to process after mapping. Exiting.")
            return

        logger.info(f"Found {len(theme_to_columns_map)} themes to process for table creation.")

        for theme, feature_columns in theme_to_columns_map.items():
            sanitized_table_name = sanitize_table_name(theme)
            logger.info(f"Processing theme: '{theme}' -> table_name: '{sanitized_table_name}'")

            if not feature_columns:
                logger.warning(f"No features found for theme '{theme}'. Skipping table creation.")
                continue

            missing_essential = [col for col in ESSENTIAL_COLUMNS if col not in self.master_df.columns]
            if missing_essential:
                logger.error(
                    f"Essential columns {missing_essential} missing from master_df. "
                    f"Cannot proceed for theme '{theme}'."
                )
                continue

            valid_feature_columns = [col for col in feature_columns if col in self.master_df.columns]
            if not valid_feature_columns:
                logger.warning(
                    f"No valid feature columns left for theme '{theme}' after checking against master_df. "
                    f"Skipping table '{sanitized_table_name}'."
                )
                continue

            cols_for_this_theme_table_ordered = ESSENTIAL_COLUMNS + sorted(list(set(valid_feature_columns)))

            theme_df = self.master_df[cols_for_this_theme_table_ordered].copy()

            if "date" in theme_df.columns and pd.api.types.is_datetime64_any_dtype(theme_df["date"]):
                theme_df["date"] = theme_df["date"].dt.strftime("%Y-%m-%d")

            for col in theme_df.columns:
                if pd.api.types.is_integer_dtype(theme_df[col]) and theme_df[col].isnull().any():
                    theme_df[col] = theme_df[col].astype(float)
                elif theme_df[col].dtype == object:
                    try:
                        if any(x is pd.NA for x in theme_df[col].unique()):
                            theme_df[col] = theme_df[col].replace({pd.NA: None})
                    except TypeError:
                        pass

            logger.info(
                f"Creating/replacing table '{sanitized_table_name}' with {len(theme_df.columns)} columns "
                f"and {len(theme_df)} rows."
            )
            try:
                theme_df.to_sql(
                    sanitized_table_name,
                    self.db_manager.engine,
                    if_exists="replace",
                    index=False,
                    chunksize=1000,
                )
                logger.info(f"Successfully created/replaced table '{sanitized_table_name}'.")
            except Exception as e:
                logger.error(
                    f"Error writing DataFrame for theme '{theme}' to table '{sanitized_table_name}': {e}",
                    exc_info=True,
                )
        logger.info("Thematic table creation process finished.")
