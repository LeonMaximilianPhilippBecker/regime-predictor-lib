import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit

from regime_predictor_lib.supervised_learning.models import CatBoostModel, LightGBMModel, XGBoostModel
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INCLUDE_CURRENT_REGIME_FEATURE = True

DB_PATH = PROJECT_ROOT_PATH / "data" / "db" / "volume" / "quant.db"
MODELS_DIR = PROJECT_ROOT_PATH / "data" / "models" / "supervised" / "thematic_models"

PROCESSED_DATA_DIR = PROJECT_ROOT_PATH / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILENAME = (
    f"ensemble_features_{'with_regime_t' if INCLUDE_CURRENT_REGIME_FEATURE else 'no_regime_t'}.csv"
)
OUTPUT_PATH = PROCESSED_DATA_DIR / OUTPUT_FILENAME

PILLAR_FEATURES = [
    "sp500_ret_252d",
    "cpi_yoy_change",
    "vol_vix",
]

MODEL_CLASS_MAP = {
    "XGBoost": XGBoostModel,
    "LightGBM_DART": LightGBMModel,
    "CatBoost": CatBoostModel,
}


def load_model_and_encoder(model_path: Path) -> tuple[Any, Any]:
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None, None

    model_name_key = model_path.stem.split("_")[-1]
    ModelClass = MODEL_CLASS_MAP.get(model_name_key)
    if not ModelClass:
        if "LightGBM" in model_path.stem:
            ModelClass = LightGBMModel
        else:
            logger.error(
                f"Could not determine model class for key '{model_name_key}' from file {model_path.name}"
            )
            return None, None

    model_wrapper = ModelClass.load_model(model_path)

    encoder_path = model_path.with_name(f"{model_path.stem}_label_encoder.pkl")
    if not encoder_path.exists():
        logger.error(f"Label encoder not found for model: {model_path.name}")
        return None, None

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    return model_wrapper, encoder


def main():
    logger.info("--- Starting Ensemble Feature Generation Script ---")
    if INCLUDE_CURRENT_REGIME_FEATURE:
        logger.info("Configuration: 'regime_t' WILL be included as a feature.")
    else:
        logger.info("Configuration: 'regime_t' WILL NOT be included as a feature.")

    db_manager = DatabaseManager(db_path=DB_PATH)

    logger.info("Loading master features table for targets and pillar features...")
    try:
        master_df = pd.read_sql_table(
            "master_features", db_manager.engine, index_col="date", parse_dates=["date"]
        )
        master_df.sort_index(inplace=True)
    except Exception as e:
        logger.error(f"Failed to load master_features table: {e}")
        return

    model_files = list(MODELS_DIR.glob("*.pkl"))
    model_files = [p for p in model_files if "label_encoder" not in p.name]

    if not model_files:
        logger.error(f"No model .pkl files found in {MODELS_DIR}. Cannot generate features.")
        return

    all_model_predictions = {}
    reference_encoder = None

    for model_path in model_files:
        logger.info(f"Processing model: {model_path.name}")
        model, encoder = load_model_and_encoder(model_path)

        if not model or not encoder:
            continue

        if reference_encoder is None:
            reference_encoder = encoder
            logger.info(
                f"Using encoder from {model_path.name} as the reference. Classes: {reference_encoder.classes_}"
            )

        features_for_model = model.feature_names_in_
        if not all(f in master_df.columns for f in features_for_model):
            logger.error(f"Model {model_path.name} requires features not in master_df. Skipping.")
            continue

        X = master_df[features_for_model].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.ffill(inplace=True)

        X.dropna(inplace=True)

        if X.empty:
            logger.warning(f"No valid data rows for {model_path.name} after cleaning. Skipping.")
            continue

        logger.info(f"Generating predictions for {len(X)} rows...")
        probabilities = model.predict_proba(X)

        proba_df = pd.DataFrame(probabilities, index=X.index)

        epsilon = 1e-15
        proba_df = proba_df.clip(epsilon, 1 - epsilon)
        logit_probas = logit(proba_df)

        model_key = model_path.stem
        logit_probas.columns = [f"logit_proba_{model_key}_regime_{c}" for c in encoder.classes_]

        all_model_predictions[model_key] = logit_probas

    if not all_model_predictions:
        logger.error("No model predictions were generated. Aborting.")
        return

    if reference_encoder is None:
        logger.error("Could not load any model's encoder. Cannot calculate disagreement metrics or proceed.")
        return

    logger.info("Assembling final ensemble feature set...")

    ensemble_features_df = pd.concat(all_model_predictions.values(), axis=1)

    for i, class_name in enumerate(reference_encoder.classes_):
        class_probas_list = []
        for _, logit_proba_df in all_model_predictions.items():
            if logit_proba_df.shape[1] > i:
                proba_series = pd.Series(
                    1 / (1 + np.exp(-logit_proba_df.iloc[:, i])), index=logit_proba_df.index
                )
                class_probas_list.append(proba_series)

        if class_probas_list:
            class_probas_df = pd.concat(class_probas_list, axis=1)
            ensemble_features_df[f"disagreement_std_regime_{class_name}"] = class_probas_df.std(axis=1)

    pillar_features_df = master_df[PILLAR_FEATURES].copy()
    ensemble_features_df = ensemble_features_df.join(pillar_features_df, how="left")

    if INCLUDE_CURRENT_REGIME_FEATURE:
        logger.info("Adding 'regime_t' to the feature set.")
        ensemble_features_df = ensemble_features_df.join(master_df[["regime_t"]], how="left")

    ensemble_features_df = ensemble_features_df.join(master_df[["regime_t_plus_6m"]], how="left")

    cols_to_ffill = PILLAR_FEATURES + (["regime_t"] if INCLUDE_CURRENT_REGIME_FEATURE else [])
    ensemble_features_df[cols_to_ffill] = ensemble_features_df[cols_to_ffill].ffill()

    initial_rows = len(ensemble_features_df)
    ensemble_features_df.dropna(inplace=True)
    logger.info(f"Dropped {initial_rows - len(ensemble_features_df)} rows with NaNs in final assembly.")

    ensemble_features_df.to_csv(OUTPUT_PATH)
    logger.info(f"Successfully generated and saved ensemble feature set to: {OUTPUT_PATH}")
    logger.info(f"Final shape of the feature set: {ensemble_features_df.shape}")
    logger.info("--- Ensemble Feature Generation Finished ---")


if __name__ == "__main__":
    main()
