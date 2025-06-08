import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_REPORT_DIR = PROJECT_ROOT_PATH / "data" / "reports" / "supervised_learning"
THEMATIC_MODELS_DIR = BASE_REPORT_DIR / "thematic_models/thematic_models"
OUTPUT_FILENAME = BASE_REPORT_DIR / "master_results_summary.csv"


def main():
    logger.info("--- Starting Results Aggregation Script ---")

    if not THEMATIC_MODELS_DIR.exists():
        logger.error(f"Thematic models directory not found at: {THEMATIC_MODELS_DIR}")
        logger.error("Please run the training script first.")
        return

    all_results = []
    json_files = list(THEMATIC_MODELS_DIR.glob("**/cv_aggregated_metrics.json"))

    if not json_files:
        logger.warning(f"No 'cv_aggregated_metrics.json' files found in {THEMATIC_MODELS_DIR}.")
        return

    logger.info(f"Found {len(json_files)} result files to aggregate.")

    for file_path in json_files:
        try:
            parts = file_path.parent.parent.name.split("_")
            model_name = parts[-1]
            theme_name = "_".join(parts[:-1])

            with open(file_path, "r") as f:
                data = json.load(f)

            flat_data = {
                "theme": theme_name,
                "model": model_name,
            }
            flat_data.update(data)
            all_results.append(flat_data)

        except (IndexError, FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not process file {file_path}: {e}")
            continue

    if not all_results:
        logger.error("No valid results were parsed. Exiting.")
        return

    results_df = pd.DataFrame(all_results)

    key_metrics = [
        "f1_macro_mean",
        "f1_macro_std",
        "mcc_mean",
        "mcc_std",
        "auc_roc_ovr_macro_mean",
        "auc_roc_ovr_macro_std",
        "log_loss_mean",
        "log_loss_std",
        "accuracy_mean",
        "accuracy_std",
        "precision_macro_mean",
        "recall_macro_mean",
    ]

    id_cols = ["theme", "model"]
    ordered_cols = id_cols + key_metrics

    remaining_cols = sorted([col for col in results_df.columns if col not in ordered_cols])
    final_cols = ordered_cols + remaining_cols

    final_cols_exist = [col for col in final_cols if col in results_df.columns]

    results_df = results_df[final_cols_exist]

    OUTPUT_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILENAME, index=False, float_format="%.4f")
    logger.info(f"Aggregated results saved to: {OUTPUT_FILENAME}")
    logger.info("--- Results Aggregation Finished ---")


if __name__ == "__main__":
    main()
