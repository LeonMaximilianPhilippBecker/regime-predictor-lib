import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INCLUDE_CURRENT_REGIME_FEATURE = False

DB_PATH = PROJECT_ROOT_PATH / "data" / "db" / "volume" / "quant.db"
DEFAULT_MODEL_PARAMS_PATH = PROJECT_ROOT_PATH / "config" / "supervised_learning" / "default_model_params.yaml"
THEMATIC_PIPELINES_PATH = PROJECT_ROOT_PATH / "config" / "supervised_learning" / "thematic_pipelines.yaml"
THEMATIC_FEATURE_LISTS_DIR = (
    PROJECT_ROOT_PATH / "data" / "processed" / "feature_selection" / "thematic_feature_lists"
)
HPO_RESULTS_DIR = PROJECT_ROOT_PATH / "data" / "hpo_results"

if INCLUDE_CURRENT_REGIME_FEATURE:
    experiment_name = "with_regime_t_feature"
    logger.info("TRAINING FINAL MODELS: Including 'regime_t' as a feature.")
else:
    experiment_name = "thematic_models"
    logger.info("TRAINING FINAL MODELS: EXCLUDING 'regime_t' as a feature.")

BASE_MODEL_DIR = PROJECT_ROOT_PATH / "data" / "models" / "supervised"
BASE_REPORT_DIR = PROJECT_ROOT_PATH / "data" / "reports" / "supervised_learning"

MODEL_CLASS_MAP = {
    "xgboost": XGBoostModel,
    "lightgbm_dart": LightGBMModel,
    "catboost": CatBoostModel,
}


def main():
    logger.info(f"--- Starting Final Thematic Model Training: {experiment_name.upper()} ---")

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
        theme_table_name = f"theme_{theme_name.split('theme_')[-1]}"

        logger.info(f"\n{'=' * 80}")
        logger.info(
            f"TRAINING FINAL MODEL {i + 1}/{len(all_pipeline_configs)}: "
            f"Theme='{theme_name}', Model='{model_type}'"
        )
        logger.info(f"{'=' * 80}")

        hpo_result_filename = f"{theme_table_name}_{model_type}_best_params.json"
        hpo_result_path = HPO_RESULTS_DIR / hpo_result_filename

        if not hpo_result_path.exists():
            logger.warning(f"HPO result file not found: {hpo_result_path}. Skipping this pipeline.")
            continue

        with open(hpo_result_path, "r") as f:
            best_hpo_params = json.load(f)
        logger.info(f"Loaded best hyperparameters from {hpo_result_path.name}")

        final_model_params = config["model_params"].copy()
        final_model_params.update(best_hpo_params)
        logger.info(f"Final model parameters after HPO update: {final_model_params}")

        try:
            df_theme = pd.read_sql_table(theme_table_name, db_manager.engine, parse_dates=["date"])
            df_theme.set_index("date", inplace=True)
            df_theme.sort_index(inplace=True)
        except Exception as e:
            logger.error(f"Failed to load table '{theme_table_name}': {e}. Skipping.")
            continue

        with open(config["feature_list_path"], "r") as f:
            features_to_use = [line.strip() for line in f if line.strip() and line.strip() in df_theme.columns]

        if (
            INCLUDE_CURRENT_REGIME_FEATURE
            and "regime_t" in df_theme.columns
            and "regime_t" not in features_to_use
        ):
            features_to_use.append("regime_t")
            logger.info("Added 'regime_t' to the feature set for this run.")

        target_col = config["target_column"]
        if target_col not in df_theme.columns:
            logger.error(f"Target column '{target_col}' not found. Skipping.")
            continue

        if not features_to_use:
            logger.warning(f"No valid features for theme '{theme_name}'. Skipping.")
            continue

        df_model_data = df_theme[features_to_use + [target_col]].dropna(subset=[target_col])
        X = df_model_data[features_to_use].copy()
        y = df_model_data[target_col]

        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        X.ffill(inplace=True)

        initial_rows = len(X)
        if X.isnull().values.any():
            X.dropna(inplace=True)
            y = y.loc[X.index]  # Re-align target variable
            logger.info(f"Dropped {initial_rows - len(X)} leading rows with NaN values after ffill.")
        # --- END OF CORRECTION ---

        if X.empty:
            logger.warning(f"Feature set for {theme_name} is empty after cleaning. Skipping.")
            continue

        ModelClass = MODEL_CLASS_MAP.get(model_type)
        model_wrapper = ModelClass(model_params=final_model_params)

        trainer = ModelTrainer(
            model_wrapper=model_wrapper,
            cv_splitter=None,
            result_saver=result_saver,
            theme_name=theme_table_name,
            label_encoder=shared_label_encoder,
        )

        if not is_encoder_fitted:
            shared_label_encoder.fit(y.unique())
            is_encoder_fitted = True
        y_encoded = pd.Series(shared_label_encoder.transform(y), index=y.index, name=y.name)

        try:
            final_sample_weights = pd.Series(
                compute_sample_weight("balanced", y=y_encoded), index=y_encoded.index
            )
            trainer.train_final_model(X, y_encoded, sample_weight=final_sample_weights)
            logger.info(f"Successfully trained and saved final model for {theme_name} - {model_type}")
        except Exception as e:
            logger.error(
                f"An error occurred during final training for {theme_name} - {model_type}: {e}",
                exc_info=True,
            )
            continue

    logger.info("--- Final Thematic Model Training Orchestration Finished ---")


if __name__ == "__main__":
    main()
