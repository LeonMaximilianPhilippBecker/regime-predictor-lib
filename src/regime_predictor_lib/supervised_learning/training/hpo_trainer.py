import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.class_weight import compute_sample_weight

from ..models.base import AbstractModel

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    def __init__(
        self,
        model_class: Callable[..., AbstractModel],
        X: pd.DataFrame,
        y: pd.Series,
        cv_splitter: BaseCrossValidator,
        search_space: Dict[str, Any],
        base_model_params: Dict[str, Any],
        use_class_weights: bool = True,
    ):
        self.model_class = model_class
        self.X = X
        self.y = y
        self.cv_splitter = cv_splitter
        self.search_space = search_space
        self.base_model_params = base_model_params
        self.use_class_weights = use_class_weights
        self.study: Optional[optuna.Study] = None
        logger.info(f"HyperparameterTuner initialized for model: {model_class.__name__}")

    def _get_suggested_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        for name, config in self.search_space.items():
            param_type = config.get("type", "float")
            param_config = config.get("params", {})

            if param_type == "categorical":
                params[name] = trial.suggest_categorical(name, **param_config)
            elif param_type == "int":
                params[name] = trial.suggest_int(name, **param_config)
            elif param_type == "float":
                for k, v in param_config.items():
                    if isinstance(v, str):
                        try:
                            param_config[k] = float(v)
                        except ValueError:
                            logger.error(
                                f"Cannot convert param '{k}' with value '{v}' "
                                f"to float for hyperparameter '{name}'"
                            )
                            raise
                params[name] = trial.suggest_float(name, **param_config)
            else:
                raise ValueError(f"Unknown parameter type '{param_type}' for hyperparameter '{name}'")

        if "bagging_temperature" in params and params.get("bagging_temperature") == 0:
            if "subsample" in params:
                params["subsample"] = 1.0

        return params

    def _objective(self, trial: optuna.Trial) -> float:
        suggested_params = self._get_suggested_params(trial)
        current_model_params = {**self.base_model_params, **suggested_params}

        fold_scores = []
        for fold_idx, (train_indices, val_indices) in enumerate(self.cv_splitter.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_indices], self.X.iloc[val_indices]
            y_train, y_val = self.y.iloc[train_indices], self.y.iloc[val_indices]

            model = self.model_class(model_params=current_model_params)

            fit_params = {}
            if self.use_class_weights:
                fit_params["sample_weight_train"] = compute_sample_weight("balanced", y=y_train)

            try:
                model.fit(X_train, y_train, **fit_params)
                y_proba = model.predict_proba(X_val)
                score = log_loss(y_val, y_proba, labels=sorted(self.y.unique()))
                fold_scores.append(score)
            except Exception as e:
                logger.warning(f"Trial {trial.number}, fold {fold_idx + 1} failed: {e}")
                return float("inf")

        avg_score = np.mean(fold_scores)
        logger.debug(f"Trial {trial.number} finished. Avg LogLoss: {avg_score:.4f}. Params: {suggested_params}")
        return avg_score

    def tune(self, n_trials: int, direction: str = "minimize") -> Dict[str, Any]:
        study = optuna.create_study(direction=direction)
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)

        self.study = study

        logger.info(f"Optimization finished. Number of trials: {len(study.trials)}")
        logger.info(f"Best trial value (avg log_loss): {study.best_value:.4f}")
        logger.info("Best parameters found:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")

        return study.best_params
