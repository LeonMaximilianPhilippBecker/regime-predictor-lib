import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


class PipelineConfigManager:
    def __init__(
        self,
        default_model_params_path: Path | str,
        thematic_pipelines_path: Path | str,
        thematic_feature_lists_dir: Path | str,
    ):
        self.default_model_params_path = Path(default_model_params_path)
        self.thematic_pipelines_path = Path(thematic_pipelines_path)
        self.thematic_feature_lists_dir = Path(thematic_feature_lists_dir)

        self._default_params = self._load_yaml(self.default_model_params_path)
        self._pipeline_definitions = self._load_yaml(self.thematic_pipelines_path)
        logger.info("PipelineConfigManager initialized and YAML files loaded.")

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            logger.error(f"Configuration file not found: {path}")
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML file {path}: {e}")
                raise

    def get_pipeline_configs(self) -> List[Dict[str, Any]]:
        if not self._pipeline_definitions or not self._default_params:
            logger.error("Pipeline definitions or default model params not loaded.")
            return []

        resolved_configs = []
        for theme_name, theme_config in self._pipeline_definitions.items():
            logger.info(f"Processing configuration for theme: {theme_name}")

            feature_list_filename = theme_config.get("features_from_list")
            if not feature_list_filename:
                logger.warning(f"Theme '{theme_name}' is missing 'features_from_list'. Skipping.")
                continue

            feature_list_path = self.thematic_feature_lists_dir / feature_list_filename
            if not feature_list_path.exists():
                logger.warning(
                    f"Feature list file '{feature_list_path}' "
                    f"for theme '{theme_name}' not found. Skipping theme."
                )
                continue

            models_to_run = theme_config.get("models_to_run", [])
            if not models_to_run:
                logger.warning(f"No models specified for theme '{theme_name}'. Skipping.")
                continue

            for model_type in models_to_run:
                if model_type not in self._default_params:
                    logger.warning(
                        f"Model type '{model_type}' for "
                        f"theme '{theme_name}' not found in "
                        "default_model_params.yaml. Skipping model."
                    )
                    continue

                model_params = self._default_params[model_type].copy()
                theme_overrides = theme_config.get("model_params_override", {}).get(model_type, {})
                model_params.update(theme_overrides)

                config = {
                    "theme_name": theme_name,
                    "model_type": model_type,
                    "description": theme_config.get("description", "No description."),
                    "target_column": theme_config.get("target_column"),
                    "feature_list_path": str(feature_list_path),
                    "cv_params": theme_config.get("cv_params", {}),
                    "model_params": model_params,
                }
                resolved_configs.append(config)
                logger.debug(f"Added resolved config for {theme_name} - {model_type}")

        logger.info(f"Generated {len(resolved_configs)} total pipeline configurations to run.")
        return resolved_configs
