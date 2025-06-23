import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from regime_predictor_lib.supervised_learning.models import CatBoostModel, LightGBMModel, XGBoostModel
from regime_predictor_lib.supervised_learning.training.pipeline_config_manager import PipelineConfigManager
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

USE_THEMATIC_MODELS_WITH_REGIME_T = True

DB_PATH = PROJECT_ROOT_PATH / "data" / "db" / "volume" / "quant.db"
DEFAULT_MODEL_PARAMS_PATH = PROJECT_ROOT_PATH / "config" / "supervised_learning" / "default_model_params.yaml"
THEMATIC_PIPELINES_PATH = PROJECT_ROOT_PATH / "config" / "supervised_learning" / "thematic_pipelines.yaml"
THEMATIC_FEATURE_LISTS_DIR = (
    PROJECT_ROOT_PATH / "data" / "processed" / "feature_selection" / "thematic_feature_lists"
)
HPO_RESULTS_DIR = PROJECT_ROOT_PATH / "data" / "hpo_results"

PROCESSED_DATA_DIR = PROJECT_ROOT_PATH / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

output_experiment_name = ("oof_from_tuned_models_"
                          f"{'with_regime_t' if USE_THEMATIC_MODELS_WITH_REGIME_T else 'no_regime_t'}")
OUTPUT_FILENAME = f"ensemble_features_{output_experiment_name}.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / OUTPUT_FILENAME

PILLAR_FEATURES = ["sp500_ret_252d", "cpi_yoy_change", "vol_vix"]

MODEL_CLASS_MAP = {
    "xgboost": XGBoostModel,
    "lightgbm_dart": LightGBMModel,
    "catboost": CatBoostModel,
}


def get_oof_predictions_for_pipeline(
        config: dict, db_manager: DatabaseManager, label_encoder: LabelEncoder
) -> pd.DataFrame | None:
    theme_name = config["theme_name"]
    model_type = config["model_type"]
    theme_table_name = f"theme_{theme_name.split('theme_')[-1]}"

    logger.info(f"--- Generating OOF preds for: Theme='{theme_name}', Model='{model_type}' ---")

    hpo_params_path = HPO_RESULTS_DIR / f"{theme_table_name}_{model_type}_best_params.json"
    if not hpo_params_path.exists():
        logger.warning(f"HPO result file not found: {hpo_params_path}. Skipping.")
        return None
    with open(hpo_params_path, "r") as f:
        best_hpo_params = json.load(f)

    final_model_params = config["model_params"].copy()
    final_model_params.update(best_hpo_params)
    logger.info("Loaded and applied tuned hyperparameters.")

    try:
        df_theme = pd.read_sql_table(theme_table_name, db_manager.engine, parse_dates=["date"])
        df_theme.set_index("date", inplace=True)
        df_theme.sort_index(inplace=True)
    except Exception as e:
        logger.error(f"Failed to load table '{theme_table_name}': {e}. Skipping.")
        return None

    with open(config["feature_list_path"], "r") as f:
        features_to_use = [line.strip() for line in f if line.strip() and line.strip() in df_theme.columns]

    if (USE_THEMATIC_MODELS_WITH_REGIME_T and "regime_t" in df_theme.columns
            and "regime_t" not in features_to_use):
        features_to_use.append("regime_t")

    target_col = config["target_column"]
    df_model_data = df_theme[features_to_use + [target_col]].dropna(subset=[target_col])

    y_encoded = pd.Series(label_encoder.transform(df_model_data[target_col]), index=df_model_data.index,
                          name=target_col)
    X = df_model_data[features_to_use].copy()

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.ffill(inplace=True)

    common_index = X.index.intersection(y_encoded.index).dropna()
    X = X.loc[common_index]
    y_encoded = y_encoded.loc[common_index]

    X.dropna(inplace=True)
    y_encoded = y_encoded.loc[X.index]

    if X.empty:
        logger.warning(f"No data remains for {theme_name} after cleaning. Skipping.")
        return None

    ModelClass = MODEL_CLASS_MAP[model_type]
    n_splits = config.get("cv_params", {}).get("n_splits", 5)
    cv_splitter = TimeSeriesSplit(n_splits=n_splits)

    oof_probas_list = []

    for fold_idx, (train_indices, val_indices) in enumerate(cv_splitter.split(X, y_encoded)):
        if len(train_indices) == 0:
            logger.warning(f"Fold {fold_idx + 1} has an empty training set. Skipping.")
            continue

        X_train_fold, X_val_fold = X.iloc[train_indices], X.iloc[val_indices]
        y_train_fold = y_encoded.iloc[train_indices]

        try:
            model_wrapper = ModelClass(model_params=final_model_params)

            model_wrapper.fit(X_train_fold, y_train_fold, label_encoder=label_encoder)

            y_proba_fold = model_wrapper.predict_proba(X_val_fold)

            proba_df = pd.DataFrame(
                y_proba_fold,
                index=X_val_fold.index,
                columns=[f"proba_class_{c}" for c in label_encoder.classes_]
            )

            oof_probas_list.append(proba_df)
        except Exception as e:
            logger.error(f"CV fold {fold_idx + 1} failed for {theme_name} - {model_type}: {e}", exc_info=True)
            return None

    if not oof_probas_list:
        logger.warning(f"No OOF predictions generated for {theme_name} - {model_type}. Skipping.")
        return None

    oof_df = pd.concat(oof_probas_list)
    return oof_df


def main():
    logger.info("--- Starting OOF Ensemble Feature Generation (from Tuned Models) ---")

    db_manager = DatabaseManager(db_path=DB_PATH)
    config_manager = PipelineConfigManager(
        default_model_params_path=DEFAULT_MODEL_PARAMS_PATH,
        thematic_pipelines_path=THEMATIC_PIPELINES_PATH,
        thematic_feature_lists_dir=THEMATIC_FEATURE_LISTS_DIR,
    )
    all_pipeline_configs = config_manager.get_pipeline_configs()

    master_df = pd.read_sql_table("master_features", db_manager.engine)
    label_encoder = LabelEncoder().fit(master_df['regime_t_plus_6m'].dropna())
    logger.info(f"Fitted a shared LabelEncoder on the full target variable. Classes: {label_encoder.classes_}")

    all_model_predictions = []

    for config in all_pipeline_configs:
        oof_df = get_oof_predictions_for_pipeline(config, db_manager, label_encoder)
        if oof_df is not None and not oof_df.empty:
            epsilon = 1e-15
            logit_probas = logit(oof_df.clip(epsilon, 1 - epsilon))
            theme_table_name = f"theme_{config['theme_name'].split('theme_')[-1]}"
            run_name = f"{theme_table_name}_{config['model_type']}"
            logit_probas.columns = [f"logit_proba_{run_name}_regime_{i}" for i in label_encoder.classes_]
            all_model_predictions.append(logit_probas)

    if not all_model_predictions:
        logger.error("No valid OOF predictions were generated across all pipelines. Aborting.")
        return

    logger.info("Assembling final ensemble feature set...")

    ensemble_features_df = pd.concat(all_model_predictions, axis=1, join='outer')

    cols_to_join = PILLAR_FEATURES + ["regime_t", "regime_t_plus_6m"]
    ensemble_features_df = ensemble_features_df.join(master_df.set_index('date')[cols_to_join], how="left")

    ffill_cols = PILLAR_FEATURES + (["regime_t"] if USE_THEMATIC_MODELS_WITH_REGIME_T else [])
    if ffill_cols:
        ensemble_features_df[ffill_cols] = ensemble_features_df[ffill_cols].ffill()

    # Final cleanup
    ensemble_features_df.dropna(subset=['regime_t_plus_6m'], inplace=True)
    ensemble_features_df.ffill(inplace=True)
    ensemble_features_df.dropna(inplace=True)

    ensemble_features_df.to_csv(OUTPUT_PATH)
    logger.info(f"Successfully generated LEAKAGE-FREE ensemble feature set to: {OUTPUT_PATH}")
    logger.info(f"Final shape of the feature set: {ensemble_features_df.shape}")
    logger.info("--- Ensemble Feature Generation Finished ---")


if __name__ == "__main__":
    main()
