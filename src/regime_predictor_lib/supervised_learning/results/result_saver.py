import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ResultSaver:
    def __init__(self, base_report_dir: Path | str, base_model_dir: Path | str):
        self.base_report_dir = Path(base_report_dir)
        self.base_model_dir = Path(base_model_dir)

        self.base_report_dir.mkdir(parents=True, exist_ok=True)
        self.base_model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ResultSaver initialized. Reports: {self.base_report_dir}, Models: {self.base_model_dir}")

    def _get_output_path(
        self,
        theme_name: str,
        model_name: str,
        artifact_subdir: str = "",
        is_model_artifact: bool = False,
    ) -> Path:
        if is_model_artifact:
            theme_dir = (
                self.base_model_dir / "thematic_models"
                if "theme_" in theme_name
                else self.base_model_dir / "gp_ensemble"
            )
            theme_dir.mkdir(parents=True, exist_ok=True)
            return theme_dir
        else:
            if "theme_" in theme_name:
                model_run_name = f"{theme_name}_{model_name}"
                report_subdir = self.base_report_dir / "thematic_models" / model_run_name
            else:
                report_subdir = self.base_report_dir / theme_name / model_name

            if artifact_subdir:
                report_subdir = report_subdir / artifact_subdir

            report_subdir.mkdir(parents=True, exist_ok=True)
            return report_subdir

    def save_cv_metrics(
        self,
        theme_name: str,
        model_name: str,
        fold_metrics_df: pd.DataFrame,
        aggregated_metrics_dict: Dict[str, Any],
        oof_predictions_df: Optional[pd.DataFrame] = None,
    ):
        output_dir = self._get_output_path(theme_name, model_name, artifact_subdir="cv_results")

        fold_metrics_df.to_csv(output_dir / "cv_fold_metrics.csv", index_label="fold")
        logger.info(f"Saved CV fold metrics to {output_dir / 'cv_fold_metrics.csv'}")

        with open(output_dir / "cv_aggregated_metrics.json", "w") as f:
            json.dump(aggregated_metrics_dict, f, indent=4)
        logger.info(f"Saved CV aggregated metrics to {output_dir / 'cv_aggregated_metrics.json'}")

        if oof_predictions_df is not None and not oof_predictions_df.empty:
            oof_predictions_df.to_csv(output_dir / "oof_predictions.csv", index=False)
            logger.info(f"Saved OOF predictions to {output_dir / 'oof_predictions.csv'}")

    def save_mda_results(self, theme_name: str, model_name: str, mda_df: pd.DataFrame):
        output_dir = self._get_output_path(theme_name, model_name, artifact_subdir="mda_results")
        mda_df.to_csv(output_dir / "mda_scores.csv", index=False)
        logger.info(f"Saved MDA scores to {output_dir / 'mda_scores.csv'}")

    def save_model_artifact(
        self, theme_name: str, model_name: str, model_object: Any, filename: Optional[str] = None
    ):
        model_dir = self._get_output_path(theme_name, model_name, is_model_artifact=True)

        if filename is None:
            filename = f"{theme_name}_{model_name}.pkl"

        filepath = model_dir / filename

        if hasattr(model_object, "save_model") and callable(getattr(model_object, "save_model")):
            model_object.save_model(filepath)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(model_object, f)
        logger.info(f"Saved model artifact to {filepath}")

    def save_plot(
        self, theme_name: str, model_name: str, plot_name: str, fig: plt.Figure, plot_subdir: str = "plots"
    ):
        output_dir = self._get_output_path(theme_name, model_name, artifact_subdir=plot_subdir)
        plot_path = output_dir / f"{plot_name}.png"
        try:
            fig.savefig(plot_path, bbox_inches="tight", dpi=150)
            logger.info(f"Saved plot to {plot_path}")
        except Exception as e:
            logger.error(f"Error saving plot {plot_name} to {plot_path}: {e}")
        plt.close(fig)

    def save_config_summary(self, theme_name: str, model_name: str, config_dict: Dict[str, Any]):
        output_dir = self._get_output_path(theme_name, model_name)
        filepath = output_dir / "run_config_summary.yaml"
        try:
            with open(filepath, "w") as f:
                yaml.dump(config_dict, f, sort_keys=False, indent=2, default_flow_style=False)
            logger.info(f"Saved run configuration summary to {filepath}")
        except Exception as e:
            logger.error(f"Error saving config summary to {filepath}: {e}")

    def save_training_log(self, theme_name: str, model_name: str, log_content: str):
        output_dir = self._get_output_path(theme_name, model_name)
        filepath = output_dir / "training_log.txt"
        try:
            with open(filepath, "w") as f:
                f.write(log_content)
            logger.info(f"Saved training log to {filepath}")
        except Exception as e:
            logger.error(f"Error saving training log to {filepath}: {e}")
