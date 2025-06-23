import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from regime_predictor_lib.supervised_learning.evaluation import PurgedKFold
from regime_predictor_lib.supervised_learning.models import LogisticRegressionModel

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

HPO_RESULTS_DIR = PROJECT_ROOT_PATH / "data" / "hpo_results"
HPO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "regime_t_plus_6m"
N_TRIALS = 50
MODEL_NAME = "LogisticRegression"

CV_PARAMS = {
    "n_splits": 5,
    "purge_length": 30,
    "embargo_length": 126
}

HPO_SEARCH_SPACE = {
    "LogisticRegression": {
        "C": {"type": "float", "params": {"low": 1e-4, "high": 1e2, "log": True}},
        "penalty": {"type": "categorical", "params": {"choices": ["l1", "l2"]}},
    }
}

LOGISTIC_BASE_PARAMS = {
    "class_weight": "balanced",
    "solver": "saga",
    "random_state": 42,
    "max_iter": 3000,
    "n_jobs": -1,
}


def main():
    experiment_name = f"logistic_ensemble_{'with_regime_t' if USE_ENSEMBLE_WITH_REGIME_T else 'no_regime_t'}"
    logger.info(f"--- Starting Ensemble Model HPO: {experiment_name} ---")

    df = pd.read_csv(ENSEMBLE_FEATURES_PATH, index_col="date", parse_dates=True)
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    X.dropna(inplace=True)
    y = y.loc[X.index]

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)
    all_known_classes = label_encoder.classes_

    cv_splitter = PurgedKFold(n_splits=CV_PARAMS["n_splits"], purge_length=CV_PARAMS["purge_length"],
                              embargo_length=CV_PARAMS["embargo_length"])

    search_space = HPO_SEARCH_SPACE[MODEL_NAME]

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, config in search_space.items():
            param_type = config.get("type", "float")
            param_config = config.get("params", {})
            if param_type == "categorical":
                params[name] = trial.suggest_categorical(name, **param_config)
            elif param_type == "float":
                params[name] = trial.suggest_float(name, **param_config)
            elif param_type == "int":
                params[name] = trial.suggest_int(name, **param_config)

        current_model_params = {**LOGISTIC_BASE_PARAMS, **params}

        fold_scores = []
        for fold_idx, (train_indices, val_indices) in enumerate(cv_splitter.split(X, y_encoded)):
            if len(train_indices) == 0:
                continue

            X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
            y_train, y_val = y_encoded.iloc[train_indices], y_encoded.iloc[val_indices]

            model = LogisticRegressionModel(model_params=current_model_params)

            try:
                model.fit(X_train, y_train)
                y_proba_fold = model.predict_proba(X_val)

                fold_trained_classes = model.model.classes_

                y_proba_full = np.zeros((len(y_val), len(all_known_classes)))

                for i, class_label in enumerate(fold_trained_classes):
                    col_idx = np.where(all_known_classes == class_label)[0][0]
                    y_proba_full[:, col_idx] = y_proba_fold[:, i]

                score = log_loss(y_val, y_proba_full, labels=all_known_classes)
                fold_scores.append(score)
            except Exception as e:
                logger.warning(f"Trial {trial.number}, fold {fold_idx + 1} failed: {e}")
                return float("inf")

        if not fold_scores:
            return float("inf")

        return np.mean(fold_scores)

    logger.info(f"Tuning {MODEL_NAME} for {N_TRIALS} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    logger.info(f"Optimization finished. Number of trials: {len(study.trials)}")

    if study.best_trial and study.best_value != float("inf"):
        logger.info(f"Best trial value (avg log_loss): {study.best_value:.4f}")
        logger.info("Best parameters found:")
        best_params = study.best_params
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")

        output_filename = f"{experiment_name}_{MODEL_NAME}_best_params.json"
        output_path = HPO_RESULTS_DIR / output_filename
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Saved best parameters to {output_path}")

    else:
        logger.error("No trials completed successfully. HPO failed. No parameters will be saved.")

    if study:
        study_filename = f"{experiment_name}_{MODEL_NAME}_study.pkl"
        study_output_path = HPO_RESULTS_DIR / study_filename
        joblib.dump(study, study_output_path)
        logger.info(f"Saved full Optuna study object to {study_output_path}")

    logger.info("--- Ensemble Model HPO Finished ---")


if __name__ == "__main__":
    main()
