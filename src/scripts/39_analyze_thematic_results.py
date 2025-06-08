import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from regime_predictor_lib.supervised_learning.models import CatBoostModel, LightGBMModel, XGBoostModel
from regime_predictor_lib.supervised_learning.results import plotting_utils
from regime_predictor_lib.supervised_learning.results.result_saver import ResultSaver

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "thematic_models"

BASE_REPORT_DIR = PROJECT_ROOT_PATH / "data" / "reports" / "supervised_learning"
THEMATIC_MODELS_DIR = BASE_REPORT_DIR / EXPERIMENT_NAME
BASE_MODEL_DIR = PROJECT_ROOT_PATH / "data" / "models" / "supervised" / EXPERIMENT_NAME

MODEL_CLASS_MAP = {
    "XGBoost": XGBoostModel,
    "LightGBM_DART": LightGBMModel,
    "CatBoost": CatBoostModel,
}


def analyze_single_run(run_dir: Path, result_saver: ResultSaver):
    logger.info(f"Analyzing results in: {run_dir.name}")

    run_name_str = run_dir.name
    model_type_key = None
    ModelClass = None

    for key, MClass in MODEL_CLASS_MAP.items():
        if run_name_str.endswith(f"_{key}"):
            model_type_key = key
            ModelClass = MClass
            break

    if not model_type_key:
        logger.warning(f"Could not determine model type for directory '{run_name_str}'. Skipping.")
        return

    theme_name = run_name_str.removesuffix(f"_{model_type_key}")
    model_name_display = ModelClass(model_params={}, model_name="temp").model_name

    oof_path = run_dir / "cv_results" / "oof_predictions.csv"
    if oof_path.exists():
        oof_df = pd.read_csv(oof_path)
        if not oof_df.empty and "true_label" in oof_df and "predicted_label" in oof_df:
            y_true = oof_df["true_label"]
            y_pred = oof_df["predicted_label"]
            class_labels = sorted(y_true.unique())
            class_names = [f"Regime {i}" for i in class_labels]

            cm = confusion_matrix(y_true, y_pred, labels=class_labels)
            fig_cm = plotting_utils.plot_confusion_matrix(
                cm, class_names=class_names, title=f"Confusion Matrix\n{theme_name} - {model_name_display}"
            )
            result_saver.save_plot(
                theme_name, model_name_display, "confusion_matrix", fig_cm, plot_subdir="analysis_plots"
            )

            fig_cm_norm = plotting_utils.plot_confusion_matrix(
                cm,
                class_names=class_names,
                title=f"Normalized Confusion Matrix\n{theme_name} - {model_name_display}",
                normalize=True,
            )
            result_saver.save_plot(
                theme_name,
                model_name_display,
                "confusion_matrix_normalized",
                fig_cm_norm,
                plot_subdir="analysis_plots",
            )

            proba_cols = [col for col in oof_df.columns if col.startswith("proba_class_")]
            if len(proba_cols) == len(class_labels):
                y_proba = oof_df[proba_cols].values
                fig_roc = plotting_utils.plot_roc_curves(
                    y_true, {model_name_display: y_proba}, class_labels, f"ROC Curves: {theme_name}"
                )
                result_saver.save_plot(
                    theme_name, model_name_display, "roc_curve", fig_roc, plot_subdir="analysis_plots"
                )
    else:
        logger.warning(f"OOF predictions not found for {run_dir.name}. Skipping plots.")

    model_path = BASE_MODEL_DIR / f"{run_dir.name}.pkl"
    if model_path.exists() and ModelClass:
        try:
            model_wrapper = ModelClass.load_model(model_path)
            importance_series = model_wrapper.get_feature_importance()

            if importance_series is not None and not importance_series.empty:
                importance_df = importance_series.reset_index()
                importance_df.columns = ["feature", "importance"]
                fig_builtin = plotting_utils.plot_feature_importances(
                    importance_df,
                    top_n=25,
                    title=f"Built-in Feature Importance\n{theme_name} - {model_name_display}",
                )
                result_saver.save_plot(
                    theme_name,
                    model_name_display,
                    "builtin_feature_importance",
                    fig_builtin,
                    plot_subdir="analysis_plots",
                )
        except Exception as e:
            logger.error(f"Failed to load model or get importance from {model_path}: {e}", exc_info=True)

    mda_path = run_dir / "mda_results" / "mda_scores.csv"
    if mda_path.exists():
        mda_per_fold_df = pd.read_csv(mda_path)

        fig_stability = plotting_utils.plot_feature_importance_stability(
            mda_per_fold_df, top_n=25, title=f"MDA Importance Stability\n{theme_name} - {model_name_display}"
        )
        result_saver.save_plot(
            theme_name,
            model_name_display,
            "mda_feature_importance_stability",
            fig_stability,
            plot_subdir="analysis_plots",
        )

        aggregated_mda = mda_per_fold_df.groupby("feature")["importance"].mean().reset_index()
        aggregated_mda.columns = ["feature", "importance"]
        fig_mda_agg = plotting_utils.plot_feature_importances(
            aggregated_mda,
            top_n=25,
            title=f"Mean Decrease Accuracy (Aggregated)\n{theme_name} - {model_name_display}",
        )
        result_saver.save_plot(
            theme_name,
            model_name_display,
            "mda_feature_importance_aggregated",
            fig_mda_agg,
            plot_subdir="analysis_plots",
        )
    else:
        logger.warning(f"MDA results not found for {run_dir.name}")


def main():
    logger.info("--- Starting Analysis of Thematic Model Results ---")

    result_saver = ResultSaver(base_report_dir=BASE_REPORT_DIR, base_model_dir=BASE_MODEL_DIR)

    run_dirs = [d for d in THEMATIC_MODELS_DIR.glob("*_*") if d.is_dir()]

    if not run_dirs:
        logger.error(f"No result directories found in {THEMATIC_MODELS_DIR}. Please run training first.")
        return

    for run_dir in run_dirs:
        try:
            analyze_single_run(run_dir, result_saver)
        except Exception as e:
            logger.error(f"Failed to analyze run directory {run_dir.name}: {e}", exc_info=True)

    logger.info("--- Analysis Script Finished ---")


if __name__ == "__main__":
    main()
