import logging
import sys
from pathlib import Path

from regime_predictor_lib.supervised_learning.feature_importance import MdaFeatureRefiner

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT_PATH / "data" / "reports" / "supervised_learning" / "thematic_models"

FEATURE_LISTS_DIR = PROJECT_ROOT_PATH / "data" / "processed" / "feature_selection" / "thematic_feature_lists"

MIN_MEAN_IMPORTANCE = 0.0

MIN_POSITIVE_FOLD_FRACTION = 0.6


def main():
    logger.info("--- Starting Automated Feature Refinement Script ---")

    if not REPORTS_DIR.exists():
        logger.error(f"Reports directory not found: {REPORTS_DIR}")
        logger.error(
            "Please run the training script (36_train_thematic_models.py) first to generate MDA results."
        )
        sys.exit(1)

    if not FEATURE_LISTS_DIR.exists():
        logger.error(f"Feature lists directory not found: {FEATURE_LISTS_DIR}")
        sys.exit(1)

    refiner = MdaFeatureRefiner(
        reports_dir=REPORTS_DIR,
        feature_lists_dir=FEATURE_LISTS_DIR,
        min_mean_importance=MIN_MEAN_IMPORTANCE,
        min_positive_fold_fraction=MIN_POSITIVE_FOLD_FRACTION,
    )

    refiner.run()

    logger.info("--- Automated Feature Refinement Script Finished ---")


if __name__ == "__main__":
    main()
