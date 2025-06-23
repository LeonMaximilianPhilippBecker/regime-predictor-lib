import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from regime_predictor_lib.supervised_learning.evaluation import calculate_classification_metrics
from regime_predictor_lib.supervised_learning.models.gaussian_process import OneVsRestGPModel
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

EXPERIMENT_NAME = f"gp_ensemble_{'with_regime_t' if USE_ENSEMBLE_WITH_REGIME_T else 'no_regime_t'}"
BASE_REPORT_DIR = PROJECT_ROOT_PATH / "data" / "reports" / "supervised_learning"
BASE_MODEL_DIR = PROJECT_ROOT_PATH / "data" / "models" / "supervised"

TEST_SET_SIZE = 0.2
SHUFFLE_SPLIT = False
TARGET_COLUMN = "regime_t_plus_6m"

GP_MODEL_PARAMS = {
    "num_inducing": 10,
    "kernel": "RBF",
    "kernel_params": {"ARD": True},
    "optimizer": "scg",
    "max_iters": 1000,
    "messages": True,
}


def main():
    logger.info(f"--- Starting GP Ensemble Training Script: {EXPERIMENT_NAME} ---")

    if not ENSEMBLE_FEATURES_PATH.exists():
        logger.error(f"Ensemble feature file not found: {ENSEMBLE_FEATURES_PATH}")
        logger.error("Please run script '42_generate_ensemble_features.py' first.")
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

    logger.info(f"Feature set shape: {X.shape}")
    logger.info(f"Target series shape: {y.shape}")

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)
    class_labels_encoded = list(range(len(label_encoder.classes_)))
    logger.info(f"Target encoded. Classes: {dict(zip(label_encoder.classes_, class_labels_encoded))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SET_SIZE, shuffle=SHUFFLE_SPLIT
    )
    logger.info(f"Data split into training ({len(X_train)} rows) and test ({len(X_test)} rows) sets.")

    gp_model = OneVsRestGPModel(model_params=GP_MODEL_PARAMS, model_name="GPClassifier")

    logger.info("Fitting the One-vs-Rest Gaussian Process model...")
    gp_model.fit(X_train, y_train)

    logger.info("Evaluating model on the test set...")
    y_pred = gp_model.predict(X_test)
    y_proba = gp_model.predict_proba(X_test)

    test_metrics = calculate_classification_metrics(y_test, y_pred, y_proba, labels=class_labels_encoded)

    logger.info("--- Test Set Performance ---")
    for key, value in test_metrics.items():
        logger.info(f"{key}: {value:.4f}")

    logger.info("Saving results and model artifacts...")

    metrics_path = result_saver._get_output_path(EXPERIMENT_NAME, "GPClassifier") / "test_set_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=4)
    logger.info(f"Test metrics saved to {metrics_path}")

    result_saver.save_model_artifact(
        theme_name=EXPERIMENT_NAME,
        model_name="GPClassifier",
        model_object=gp_model,
        filename=f"{EXPERIMENT_NAME}_GPClassifier.pkl",
    )

    result_saver.save_model_artifact(
        theme_name=EXPERIMENT_NAME,
        model_name="GPClassifier",
        model_object=label_encoder,
        filename=f"{EXPERIMENT_NAME}_GPClassifier_label_encoder.pkl",
    )

    if hasattr(gp_model, "get_ard_lengthscales"):
        feature_lengthscales_df = gp_model.get_ard_lengthscales()

        logger.info("\n--- Learned ARD Lengthscales (Lower is More Important) ---")
        logger.info(f"\n{feature_lengthscales_df.to_string()}")

        lengthscales_path = (
            result_saver._get_output_path(EXPERIMENT_NAME, "GPClassifier") / "ard_lengthscales.csv"
        )
        lengthscales_path.parent.mkdir(parents=True, exist_ok=True)
        feature_lengthscales_df.to_csv(lengthscales_path)
        logger.info(f"ARD lengthscales saved to {lengthscales_path}")

    logger.info("--- GP Ensemble Training Script Finished ---")


if __name__ == "__main__":
    main()
