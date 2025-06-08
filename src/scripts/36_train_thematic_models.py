import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from regime_predictor_lib.supervised_learning.models import CatBoostModel, LightGBMModel, XGBoostModel
from regime_predictor_lib.supervised_learning.results.result_saver import ResultSaver
from regime_predictor_lib.supervised_learning.training.pipeline_config_manager import PipelineConfigManager
from regime_predictor_lib.supervised_learning.training.trainer import ModelTrainer
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INCLUDE_CURRENT_REGIME_FEATURE = False

DB_PATH = PROJECT_ROOT_PATH / "data" / "db" / "volume" / "quant.db"
DEFAULT_MODEL_PARAMS_PATH = PROJECT_ROOT_PATH / "config" / "supervised_learning" / "default_model_params.yaml"
THEMATIC_PIPELINES_PATH = PROJECT_ROOT_PATH / "config" / "supervised_learning" / "thematic_pipelines.yaml"
THEMATIC_FEATURE_LISTS_DIR = (
    PROJECT_ROOT_PATH / "data" / "processed" / "feature_selection" / "thematic_feature_lists"
)

if INCLUDE_CURRENT_REGIME_FEATURE:
    experiment_name = "thematic_models_with_regime_t"
    logger.info("RUNNING EXPERIMENT: Including 'regime_t' as a feature.")
else:
    experiment_name = "thematic_models"
    logger.info("RUNNING EXPERIMENT: EXCLUDING 'regime_t' as a feature.")

BASE_REPORT_DIR = PROJECT_ROOT_PATH / "data" / "reports" / "supervised_learning"
BASE_MODEL_DIR = PROJECT_ROOT_PATH / "data" / "models" / "supervised"

MODEL_CLASS_MAP = {
    "xgboost": XGBoostModel,
    "lightgbm_dart": LightGBMModel,
    "catboost": CatBoostModel,
}


def main():
    logger.info(f"--- Starting Thematic Model Training: {experiment_name.upper()} ---")

    db_manager = DatabaseManager(db_path=DB_PATH)
    result_saver = ResultSaver(base_report_dir=BASE_REPORT_DIR, base_model_dir=BASE_MODEL_DIR)

    config_manager = PipelineConfigManager(
        default_model_params_path=DEFAULT_MODEL_PARAMS_PATH,
        thematic_pipelines_path=THEMATIC_PIPELINES_PATH,
        thematic_feature_lists_dir=THEMATIC_FEATURE_LISTS_DIR,
    )

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
            f"RUNNING PIPELINE {i + 1}/{len(all_pipeline_configs)}: Theme='{theme_name}', Model='{model_type}'"
        )
        logger.info(f"{'=' * 80}")

        try:
            theme_table_name = f"theme_{theme_name.split('theme_')[-1]}"
            df_theme = pd.read_sql_table(theme_table_name, db_manager.engine, parse_dates=["date"])
            df_theme.set_index("date", inplace=True)
            df_theme.sort_index(inplace=True)
        except Exception as e:
            logger.error(f"Failed to load table '{theme_table_name}' for theme '{theme_name}': {e}. Skipping.")
            continue

        with open(config["feature_list_path"], "r") as f:
            candidate_features = [line.strip() for line in f if line.strip()]

        target_col = config["target_column"]
        if target_col not in df_theme.columns:
            logger.error(f"Target column '{target_col}' not found in table '{theme_table_name}'. Skipping.")
            continue

        features_to_use = [f for f in candidate_features if f in df_theme.columns]

        if INCLUDE_CURRENT_REGIME_FEATURE:
            if "regime_t" in df_theme.columns and "regime_t" not in features_to_use:
                features_to_use.append("regime_t")
                logger.info("Added 'regime_t' to the feature set for this run.")

        if not features_to_use:
            logger.warning(f"No candidate features for theme '{theme_name}' found in its table. Skipping.")
            continue

        logger.info(f"Feature set for '{theme_name}': {len(features_to_use)} features.")

        df_theme.dropna(subset=[target_col], inplace=True)
        df_model_data = df_theme[features_to_use + [target_col]].copy()

        X = df_model_data[features_to_use]
        y = df_model_data[target_col]

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.ffill(inplace=True)
        nan_rows_after = X.isnull().any(axis=1).sum()

        if nan_rows_after > 0:
            logger.info(f"{nan_rows_after} leading NaNs remain after ffill. Passing to model.")

        if not is_encoder_fitted:
            y_unique = y.dropna().unique()
            shared_label_encoder.fit(y_unique)
            is_encoder_fitted = True
            logger.info(f"Fitted shared LabelEncoder. Classes: {shared_label_encoder.classes_}")

        y_encoded = pd.Series(shared_label_encoder.transform(y), index=y.index, name=y.name)

        ModelClass = MODEL_CLASS_MAP.get(model_type)
        if not ModelClass:
            logger.warning(f"Model type '{model_type}' not recognized. Skipping.")
            continue

        model_wrapper = ModelClass(model_params=config["model_params"])
        cv_params = config["cv_params"]
        n_splits = cv_params.get("n_splits", 5)

        cv_splitter = TimeSeriesSplit(n_splits=n_splits)
        logger.info(f"Using TimeSeriesSplit with {n_splits} splits for cross-validation.")

        trainer = ModelTrainer(
            model_wrapper=model_wrapper,
            cv_splitter=cv_splitter,
            result_saver=result_saver,
            theme_name=theme_table_name,
            label_encoder=shared_label_encoder,
        )

        fit_params = {"early_stopping_rounds": 50, "verbose": False}

        try:
            trainer.run_cross_validation(X, y_encoded, fit_params=fit_params, use_class_weights=True)

            final_sample_weights = pd.Series(
                compute_sample_weight(class_weight="balanced", y=y_encoded), index=y_encoded.index
            )
            trainer.train_final_model(X, y_encoded, sample_weight=final_sample_weights, fit_params=fit_params)

        except Exception as e:
            logger.error(
                f"An error occurred during training for {theme_name} - {model_type}: {e}", exc_info=True
            )
            continue

    logger.info("--- Thematic Model Training Orchestration Finished ---")


if __name__ == "__main__":
    main()
