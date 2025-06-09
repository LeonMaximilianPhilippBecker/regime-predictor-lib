import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from regime_predictor_lib.supervised_learning.models import CatBoostModel, LightGBMModel, XGBoostModel
from regime_predictor_lib.supervised_learning.training.hpo_trainer import HyperparameterTuner
from regime_predictor_lib.supervised_learning.training.pipeline_config_manager import PipelineConfigManager
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INCLUDE_CURRENT_REGIME_FEATURE = False
N_TRIALS_PER_PIPELINE = 50

DB_PATH = PROJECT_ROOT_PATH / "data" / "db" / "volume" / "quant.db"
DEFAULT_MODEL_PARAMS_PATH = PROJECT_ROOT_PATH / "config" / "supervised_learning" / "default_model_params.yaml"
THEMATIC_PIPELINES_PATH = PROJECT_ROOT_PATH / "config" / "supervised_learning" / "thematic_pipelines.yaml"
HPO_SEARCH_SPACE_PATH = PROJECT_ROOT_PATH / "config" / "supervised_learning" / "hpo_search_space.yaml"
THEMATIC_FEATURE_LISTS_DIR = (
    PROJECT_ROOT_PATH / "data" / "processed" / "feature_selection" / "thematic_feature_lists"
)

HPO_RESULTS_DIR = PROJECT_ROOT_PATH / "data" / "hpo_results"
HPO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CLASS_MAP = {
    "xgboost": XGBoostModel,
    "lightgbm_dart": LightGBMModel,
    "catboost": CatBoostModel,
}


def main():
    experiment_name = "with_regime_t" if INCLUDE_CURRENT_REGIME_FEATURE else "no_regime_t"
    logger.info(f"--- Starting Thematic Model HPO: {experiment_name.upper()} ---")

    db_manager = DatabaseManager(db_path=DB_PATH)
    config_manager = PipelineConfigManager(
        default_model_params_path=DEFAULT_MODEL_PARAMS_PATH,
        thematic_pipelines_path=THEMATIC_PIPELINES_PATH,
        thematic_feature_lists_dir=THEMATIC_FEATURE_LISTS_DIR,
    )
    with open(HPO_SEARCH_SPACE_PATH, "r") as f:
        hpo_search_spaces = yaml.safe_load(f)

    all_pipeline_configs = config_manager.get_pipeline_configs()
    if not all_pipeline_configs:
        logger.warning("No valid pipeline configurations found. Exiting.")
        return

    shared_label_encoder = LabelEncoder()
    is_encoder_fitted = False

    for i, config in enumerate(all_pipeline_configs):
        theme_name = config["theme_name"]
        model_type = config["model_type"]
        logger.info(f"\n{'=' * 80}")
        logger.info(
            f"TUNING PIPELINE {i + 1}/{len(all_pipeline_configs)}: Theme='{theme_name}', Model='{model_type}'"
        )
        logger.info(f"{'=' * 80}")

        try:
            theme_table_name = f"theme_{theme_name.split('theme_')[-1]}"
            df_theme = pd.read_sql_table(theme_table_name, db_manager.engine, parse_dates=["date"])
            df_theme.set_index("date", inplace=True)
            df_theme.sort_index(inplace=True)
        except Exception as e:
            logger.error(f"Failed to load table '{theme_table_name}': {e}. Skipping.")
            continue

        with open(config["feature_list_path"], "r") as f:
            features_to_use = [line.strip() for line in f if line.strip() and line.strip() in df_theme.columns]

        target_col = config["target_column"]
        if target_col not in df_theme.columns:
            logger.error(f"Target column '{target_col}' not found. Skipping.")
            continue

        if not features_to_use:
            logger.warning(f"No features to use for theme '{theme_name}'. Skipping.")
            continue

        df_model_data = df_theme[features_to_use + [target_col]].dropna(subset=[target_col])
        X = df_model_data[features_to_use].copy()
        y = df_model_data[target_col]

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.ffill(inplace=True)

        initial_rows = len(X)
        X.dropna(inplace=True)
        y = y.loc[X.index]
        if len(X) < initial_rows:
            logger.info(f"Dropped {initial_rows - len(X)} rows with leading NaNs from the start of the series.")

        if not is_encoder_fitted:
            y_unique_values = y.dropna().unique()
            if len(y_unique_values) > 0:
                shared_label_encoder.fit(y_unique_values)
                is_encoder_fitted = True
                logger.info(f"Fitted shared LabelEncoder. Classes: {shared_label_encoder.classes_}")
            else:
                logger.error("Target column has no valid values. Skipping pipeline.")
                continue

        y_encoded = pd.Series(shared_label_encoder.transform(y), index=y.index, name=y.name)

        ModelClass = MODEL_CLASS_MAP.get(model_type)
        if not ModelClass:
            logger.warning(f"Model type '{model_type}' not recognized. Skipping.")
            continue

        search_space = hpo_search_spaces.get(model_type)
        if not search_space:
            logger.warning(f"No HPO search space defined for '{model_type}'. Skipping.")
            continue

        n_splits = config.get("cv_params", {}).get("n_splits", 5)
        cv_splitter = TimeSeriesSplit(n_splits=n_splits)

        tuner = HyperparameterTuner(
            model_class=ModelClass,
            X=X,
            y=y_encoded,
            cv_splitter=cv_splitter,
            search_space=search_space,
            base_model_params=config["model_params"],
        )

        try:
            best_params = tuner.tune(n_trials=N_TRIALS_PER_PIPELINE)

            output_filename = f"{theme_table_name}_{model_type}_best_params.json"
            output_path = HPO_RESULTS_DIR / output_filename
            with open(output_path, "w") as f:
                json.dump(best_params, f, indent=4)
            logger.info(f"Saved best parameters to {output_path}")

            if tuner.study:
                study_filename = f"{theme_table_name}_{model_type}_study.pkl"
                study_output_path = HPO_RESULTS_DIR / study_filename
                try:
                    joblib.dump(tuner.study, study_output_path)
                    logger.info(f"Saved full Optuna study object to {study_output_path}")
                except Exception as e:
                    logger.error(f"Failed to save study object to {study_output_path}: {e}")

        except Exception as e:
            logger.error(f"HPO failed for {theme_name} - {model_type}: {e}", exc_info=True)
            continue

    logger.info("--- Thematic Model HPO Finished ---")


if __name__ == "__main__":
    main()
