import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class MdaFeatureRefiner:
    def __init__(
        self,
        reports_dir: Path,
        feature_lists_dir: Path,
        min_mean_importance: float = 0.0,
        min_positive_fold_fraction: float = 0.6,
    ):
        self.reports_dir = reports_dir
        self.feature_lists_dir = feature_lists_dir
        self.min_mean_importance = min_mean_importance
        self.min_positive_fold_fraction = min_positive_fold_fraction
        logger.info("MdaFeatureRefiner initialized with pruning rules:")
        logger.info(f"  - Minimum Mean MDA Importance: {self.min_mean_importance}")
        logger.info(f"  - Minimum Positive Fold Fraction: {self.min_positive_fold_fraction}")

    def _find_and_group_mda_files(self) -> Dict[str, List[Path]]:
        mda_files = list(self.reports_dir.glob("**/mda_results/mda_scores.csv"))
        themes: Dict[str, List[Path]] = {}
        for file_path in mda_files:
            run_name = file_path.parent.parent.name

            theme_name = "_".join(run_name.split("_")[:-1])
            if "thematic_models" in theme_name:
                theme_name = "_".join(run_name.split("_")[:-1])
            else:
                theme_name = run_name.rsplit("_", 1)[0]

            if theme_name not in themes:
                themes[theme_name] = []
            themes[theme_name].append(file_path)
        logger.info(f"Found {len(mda_files)} MDA result files across {len(themes)} themes.")
        return themes

    def _load_and_combine_mda_data(self, mda_files: List[Path]) -> pd.DataFrame:
        all_dfs = [pd.read_csv(f) for f in mda_files if f.exists()]
        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    def _apply_pruning_rules(self, combined_mda_df: pd.DataFrame) -> List[str]:
        if combined_mda_df.empty:
            return []

        stats = combined_mda_df.groupby("feature")["importance"].agg(
            mean_importance="mean", positive_fold_fraction=lambda x: (x > 0).mean()
        )

        kept_features = stats[
            (stats["mean_importance"] > self.min_mean_importance)
            & (stats["positive_fold_fraction"] >= self.min_positive_fold_fraction)
        ]

        logger.debug(f"Feature stats:\n{stats.sort_values('mean_importance', ascending=False).to_string()}")

        return sorted(kept_features.index.tolist())

    def _write_refined_feature_list(self, theme_name: str, features: List[str]):
        filename = f"{theme_name}_selected_features.txt"
        output_path = self.feature_lists_dir / filename

        try:
            with open(output_path, "w") as f:
                for feature in features:
                    f.write(f"{feature}\n")
            logger.info(f"Successfully updated feature list: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write updated feature list to {output_path}: {e}")

    def run(self):
        logger.info("Starting automated feature list refinement based on MDA results...")

        themes_with_mda = self._find_and_group_mda_files()

        if not themes_with_mda:
            logger.warning("No MDA result files found. Cannot perform refinement.")
            return

        for theme_name, mda_files in themes_with_mda.items():
            logger.info(f"--- Processing theme: {theme_name} ---")

            combined_mda_df = self._load_and_combine_mda_data(mda_files)
            if combined_mda_df.empty:
                logger.warning(f"No MDA data could be loaded for theme '{theme_name}'. Skipping.")
                continue

            initial_feature_count = len(combined_mda_df["feature"].unique())

            refined_features = self._apply_pruning_rules(combined_mda_df)

            logger.info(f"Feature count for '{theme_name}': {initial_feature_count} -> {len(refined_features)}")

            self._write_refined_feature_list(theme_name, refined_features)

        logger.info("Automated feature list refinement complete.")
