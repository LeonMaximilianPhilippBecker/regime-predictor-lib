import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from regime_predictor_lib.supervised_learning.evaluation import PurgedKFold
from regime_predictor_lib.supervised_learning.models import LogisticRegressionModel
from regime_predictor_lib.supervised_learning.results.result_saver import ResultSaver
from regime_predictor_lib.supervised_learning.training.trainer import ModelTrainer

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

USE_ENSEMBLE_WITH_REGIME_T = True

PROCESSED_DATA_DIR = PROJECT_ROOT_PATH / "data" / "processed"
ENSEMBLE_FEATURES_FILENAME = (
    f"ensemble_features_{'with_regime_t' if USE_ENSEMBLE_WITH_REGIME_T else 'no_regime_t'}.csv"
)
ENSEMBLE_FEATURES_PATH = PROCESSED_DATA_DIR / ENSEMBLE_FEATURES_FILENAME

EXPERIMENT_NAME = f"logistic_ensemble_{'with_regime_t' if USE_ENSEMBLE_WITH_REGIME_T else 'no_regime_t'}"
BASE_REPORT_DIR = PROJECT_ROOT_PATH / "data" / "reports" / "supervised_learning"
BASE_MODEL_DIR = PROJECT_ROOT_PATH / "data" / "models" / "supervised"

TARGET_COLUMN = "regime_t_plus_6m"

CV_PARAMS = {
    "n_splits": 10,
    "purge_length": 30,
    "embargo_length": 126
}

ENSEMBLE_MODELS_TO_RUN = {
    "LogisticRegression": {
        "class": LogisticRegressionModel,
        "params": {
            "penalty": "l2",
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "liblinear",
            "random_state": 42,
            "max_iter": 1000,
        },
    }
}


def main():
    logger.info(f"--- Starting Ensemble Model Cross-Validation: {EXPERIMENT_NAME} ---")

    if not ENSEMBLE_FEATURES_PATH.exists():
        logger.error(f"Ensemble feature file not found: {ENSEMBLE_FEATURES_PATH}")
        return

    result_saver = ResultSaver(base_report_dir=BASE_REPORT_DIR, base_model_dir=BASE_MODEL_DIR)

    logger.info(f"Loading ensemble features from {ENSEMBLE_FEATURES_PATH}")
    df = pd.read_csv(ENSEMBLE_FEATURES_PATH, index_col="date", parse_dates=True)
    df.sort_index(inplace=True)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    initial_rows = len(X)
    X.dropna(inplace=True)
    y = y.loc[X.index]
    if len(X) < initial_rows:
        logger.info(f"Dropped {initial_rows - len(X)} rows with NaN values from features.")

    logger.info(f"Final feature set shape for CV: {X.shape}")

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)
    logger.info("Target encoded. Classes: "
                f"{dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    for model_name, model_config in ENSEMBLE_MODELS_TO_RUN.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"RUNNING CV FOR ENSEMBLE MODEL: '{model_name}'")
        logger.info(f"{'=' * 80}")

        ModelClass = model_config["class"]
        model_params = model_config["params"]

        model_wrapper = ModelClass(model_params=model_params)

        cv_splitter = PurgedKFold(
            n_splits=CV_PARAMS["n_splits"],
            purge_length=CV_PARAMS["purge_length"],
            embargo_length=CV_PARAMS["embargo_length"]
        )
        logger.info(
            f"Using PurgedKFold with n_splits={CV_PARAMS['n_splits']}, "
            f"purge={CV_PARAMS['purge_length']}, embargo={CV_PARAMS['embargo_length']}")

        trainer = ModelTrainer(
            model_wrapper=model_wrapper,
            cv_splitter=cv_splitter,
            result_saver=result_saver,
            theme_name=EXPERIMENT_NAME,
            label_encoder=label_encoder,
        )

        try:
            trainer.run_cross_validation(X, y_encoded,
                                         use_class_weights=False)
        except Exception as e:
            logger.error(f"An error occurred during CV for {model_name}: {e}", exc_info=True)
            continue

    logger.info("--- Ensemble Model Cross-Validation Finished ---")


if __name__ == "__main__":
    main()
