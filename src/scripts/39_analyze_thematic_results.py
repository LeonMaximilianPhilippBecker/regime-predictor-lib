import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from regime_predictor_lib.supervised_learning.results import plotting_utils
from regime_predictor_lib.supervised_learning.results.result_saver import ResultSaver

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_REPORT_DIR = PROJECT_ROOT_PATH / "data" / "reports" / "supervised_learning"
THEMATIC_MODELS_DIR = BASE_REPORT_DIR / "thematic_models/thematic_models"
BASE_MODEL_DIR = PROJECT_ROOT_PATH / "data" / "models" / "supervised"


def analyze_single_run(run_dir: Path, result_saver: ResultSaver):
    logger.info(f"Analyzing results in: {run_dir.name}")

    parts = run_dir.name.split("_")
    model_name = parts[-1]
    theme_name = "_".join(parts[:-1])

    oof_path = run_dir / "cv_results" / "oof_predictions.csv"
    if not oof_path.exists():
        logger.warning(f"OOF predictions not found for {run_dir.name}. Skipping plots.")
        return

    oof_df = pd.read_csv(oof_path)
    if oof_df.empty or "true_label" not in oof_df or "predicted_label" not in oof_df:
        logger.warning(f"OOF predictions file for {run_dir.name} is empty or malformed.")
        return

    y_true = oof_df["true_label"]
    y_pred = oof_df["predicted_label"]
    class_labels = sorted(y_true.unique())
    class_names = [f"Regime {i}" for i in class_labels]

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    fig_cm = plotting_utils.plot_confusion_matrix(
        cm, class_names=class_names, title=f"Confusion Matrix\n{theme_name} - {model_name}"
    )
    result_saver.save_plot(theme_name, model_name, "confusion_matrix", fig_cm)

    fig_cm_norm = plotting_utils.plot_confusion_matrix(
        cm,
        class_names=class_names,
        title=f"Normalized Confusion Matrix\n{theme_name} - {model_name}",
        normalize=True,
    )
    result_saver.save_plot(theme_name, model_name, "confusion_matrix_normalized", fig_cm_norm)

    proba_cols = [col for col in oof_df.columns if col.startswith("proba_class_")]
    if len(proba_cols) == len(class_labels):
        y_proba = oof_df[proba_cols].values
        fig_roc = plotting_utils.plot_roc_curves(
            y_true, {model_name: y_proba}, class_labels, f"ROC Curves: {theme_name}"
        )
        result_saver.save_plot(theme_name, model_name, "roc_curve", fig_roc)


def main():
    logger.info("--- Starting Analysis of Thematic Model Results ---")

    result_saver = ResultSaver(base_report_dir=BASE_REPORT_DIR, base_model_dir=BASE_MODEL_DIR)

    if not THEMATIC_MODELS_DIR.exists() or not any(THEMATIC_MODELS_DIR.iterdir()):
        logger.error(f"No result directories found in {THEMATIC_MODELS_DIR}. Please run training first.")
        return

    for run_dir in THEMATIC_MODELS_DIR.iterdir():
        if run_dir.is_dir():
            analyze_single_run(run_dir, result_saver)

    logger.info("--- Analysis Script Finished ---")


if __name__ == "__main__":
    main()
