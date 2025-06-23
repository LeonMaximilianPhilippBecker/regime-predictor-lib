import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from regime_predictor_lib.supervised_learning.models import LogisticRegressionModel
from regime_predictor_lib.supervised_learning.results.result_saver import ResultSaver

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

USE_ENSEMBLE_WITH_REGIME_T = True

PROCESSED_DATA_DIR = PROJECT_ROOT_PATH / "data" / "processed"
ENSEMBLE_FEATURES_FILENAME = (
    f"ensemble_features_{'with_regime_t' if USE_ENSEMBLE_WITH_REGIME_T else 'no_regime_t'}.csv"
)
ENSEMBLE_FEATURES_PATH = PROCESSED_DATA_DIR / ENSEMBLE_FEATURES_FILENAME
HPO_RESULTS_DIR = PROJECT_ROOT_PATH / "data" / "hpo_results"

EXPERIMENT_NAME = f"logistic_ensemble_{'with_regime_t' if USE_ENSEMBLE_WITH_REGIME_T else 'no_regime_t'}"
MODEL_NAME = "LogisticRegression"
BASE_REPORT_DIR = PROJECT_ROOT_PATH / "data" / "reports" / "supervised_learning"
BASE_MODEL_DIR = PROJECT_ROOT_PATH / "data" / "models" / "supervised"

TARGET_COLUMN = "regime_t_plus_6m"

DEFAULT_LOGISTIC_MODEL_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "liblinear",
    "random_state": 42,
    "max_iter": 2000,
}


def main():
    logger.info(f"--- Starting Final Ensemble Model Training: {EXPERIMENT_NAME} ---")

    if not ENSEMBLE_FEATURES_PATH.exists():
        logger.error(f"Ensemble feature file not found: {ENSEMBLE_FEATURES_PATH}")
        return

    result_saver = ResultSaver(base_report_dir=BASE_REPORT_DIR, base_model_dir=BASE_MODEL_DIR)

    hpo_params_path = HPO_RESULTS_DIR / f"{EXPERIMENT_NAME}_{MODEL_NAME}_best_params.json"
    final_model_params = DEFAULT_LOGISTIC_MODEL_PARAMS.copy()

    if hpo_params_path.exists():
        logger.info(f"Found HPO results. Loading best parameters from: {hpo_params_path}")
        with open(hpo_params_path, "r") as f:
            best_hpo_params = json.load(f)
        final_model_params.update(best_hpo_params)
    else:
        logger.warning(f"HPO result file not found at {hpo_params_path}. Using default parameters.")

    logger.info(f"Final parameters for training: {final_model_params}")

    df = pd.read_csv(ENSEMBLE_FEATURES_PATH, index_col="date", parse_dates=True)
    df.sort_index(inplace=True)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    initial_rows = len(X)
    X.dropna(inplace=True)
    y = y.loc[X.index]
    if len(X) < initial_rows:
        logger.info(f"Dropped {initial_rows - len(X)} rows with NaN values from features.")

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)
    logger.info("Target encoded. Classes: "
                f"{dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    model_wrapper = LogisticRegressionModel(model_params=final_model_params)

    logger.info(f"Training final {MODEL_NAME} model on {len(X)} data points...")
    model_wrapper.fit(X, y_encoded)

    logger.info("Saving final model and label encoder...")
    result_saver.save_model_artifact(
        theme_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        model_object=model_wrapper,
        filename=f"{EXPERIMENT_NAME}_{MODEL_NAME}.pkl",
    )

    result_saver.save_model_artifact(
        theme_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        model_object=label_encoder,
        filename=f"{EXPERIMENT_NAME}_{MODEL_NAME}_label_encoder.pkl",
    )

    logger.info("--- Final Ensemble Model Training Script Finished ---")


if __name__ == "__main__":
    main()
