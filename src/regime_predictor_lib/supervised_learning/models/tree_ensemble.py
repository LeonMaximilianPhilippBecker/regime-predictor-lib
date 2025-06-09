import logging
from typing import Any, Dict, Optional

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from .base import AbstractModel

logger = logging.getLogger(__name__)


class XGBoostModel(AbstractModel):
    def __init__(self, model_params: Dict[str, Any], model_name: str = "XGBoost"):
        super().__init__(model_params=model_params, model_name=model_name)
        if "use_label_encoder" not in self.model_params:
            self.model_params["use_label_encoder"] = False
        self.model: Optional[xgb.XGBClassifier] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight_train: Optional[pd.Series] = None,
        sample_weight_val: Optional[pd.Series] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> "XGBoostModel":
        super().fit(X_train, y_train)

        self.model = xgb.XGBClassifier(**self.model_params)

        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
        sw_train_np = (
            sample_weight_train.values if isinstance(sample_weight_train, pd.Series) else sample_weight_train
        )

        eval_set = []
        fit_params = {}

        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
            eval_set.append((X_val_np, y_val_np))

            if sample_weight_val is not None:
                sw_val_np = (
                    sample_weight_val.values if isinstance(sample_weight_val, pd.Series) else sample_weight_val
                )
                fit_params["eval_sample_weight"] = [sw_val_np]

            if early_stopping_rounds is not None:
                fit_params["early_stopping_rounds"] = early_stopping_rounds

        if sw_train_np is not None:
            fit_params["sample_weight"] = sw_train_np

        logger.info(
            f"Fitting {self.model_name} with params: {self.model_params} "
            f"and fit_params: {list(fit_params.keys())}"
        )
        try:
            self.model.fit(
                X_train_np,
                y_train_np,
                eval_set=eval_set if eval_set else None,
                verbose=kwargs.get("verbose", False),
                **fit_params,
            )
        except Exception as e:
            logger.error(f"Error during {self.model_name} fitting: {e}", exc_info=True)
            raise
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.feature_names_in_:
            X = X[self.feature_names_in_]
        return self.model.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.feature_names_in_:
            X = X[self.feature_names_in_]
        return self.model.predict_proba(X.values)

    def get_feature_importance(self, importance_type: str = "gain", **kwargs) -> Optional[pd.Series]:
        if self.model is None or not self.feature_names_in_:
            logger.warning("Model not fitted or feature names not available.")
            return None
        try:
            if hasattr(self.model, "get_booster"):
                booster = self.model.get_booster()
                fscore = booster.get_score(importance_type=importance_type)
                importances = pd.Series(fscore, name="importance").reindex(
                    self.feature_names_in_, fill_value=0.0
                )
                return importances.sort_values(ascending=False)
            elif hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                return pd.Series(importances, index=self.feature_names_in_).sort_values(ascending=False)
            return None
        except Exception as e:
            logger.error(f"Error getting feature importance for {self.model_name}: {e}", exc_info=True)
            return None


class LightGBMModel(AbstractModel):
    def __init__(self, model_params: Dict[str, Any], model_name: str = "LightGBM_DART"):
        super().__init__(model_params=model_params, model_name=model_name)
        if "boosting_type" not in self.model_params:
            self.model_params["boosting_type"] = "dart"
        if "verbose" not in self.model_params:
            self.model_params["verbose"] = -1
        self.model: Optional[lgb.LGBMClassifier] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight_train: Optional[pd.Series] = None,
        sample_weight_val: Optional[pd.Series] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> "LightGBMModel":
        super().fit(X_train, y_train)

        fit_model_params = self.model_params.copy()
        if fit_model_params.get("objective") in ["multiclass", "multi_logloss"]:
            fit_model_params["num_class"] = y_train.nunique()

        self.model = lgb.LGBMClassifier(**fit_model_params)

        eval_set = []
        fit_params = {}

        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            if sample_weight_val is not None:
                fit_params["eval_sample_weight"] = [sample_weight_val]
            if early_stopping_rounds:
                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds, verbose=kwargs.get("verbose", False)
                    )
                ]
                fit_params["callbacks"] = callbacks

        if sample_weight_train is not None:
            fit_params["sample_weight"] = sample_weight_train

        logger.info(
            f"Fitting {self.model_name} with params: {fit_model_params} "
            f"and fit_params: {list(fit_params.keys())}"
        )
        try:
            self.model.fit(X_train, y_train, eval_set=eval_set if eval_set else None, **fit_params)
        except Exception as e:
            logger.error(f"Error during {self.model_name} fitting: {e}", exc_info=True)
            raise
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.feature_names_in_:
            X = X[self.feature_names_in_]
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.feature_names_in_:
            X = X[self.feature_names_in_]
        return self.model.predict_proba(X)

    def get_feature_importance(self, **kwargs) -> Optional[pd.Series]:
        if self.model is None or not hasattr(self.model, "feature_importances_") or not self.feature_names_in_:
            logger.warning("Model not fitted or feature importances not available.")
            return None
        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_names_in_).sort_values(ascending=False)


class CatBoostModel(AbstractModel):
    def __init__(self, model_params: Dict[str, Any], model_name: str = "CatBoost"):
        super().__init__(model_params=model_params, model_name=model_name)
        if "verbose" not in self.model_params:
            self.model_params["verbose"] = 0
        self.model: Optional[cb.CatBoostClassifier] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight_train: Optional[pd.Series] = None,
        sample_weight_val: Optional[pd.Series] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> "CatBoostModel":
        super().fit(X_train, y_train)

        fit_model_params = self.model_params.copy()

        if fit_model_params.get("subsample", 1.0) < 1.0 and fit_model_params.get("bagging_temperature", 0) > 0:
            if "bootstrap_type" not in fit_model_params or fit_model_params.get("bootstrap_type") == "Bayesian":
                fit_model_params["bootstrap_type"] = "Bernoulli"
                logger.debug("Set CatBoost bootstrap_type to Bernoulli to support subsample parameter.")

        self.model = cb.CatBoostClassifier(**fit_model_params)

        eval_set = None
        fit_params = {}

        if X_val is not None and y_val is not None:
            if sample_weight_val is not None:
                eval_set = cb.Pool(data=X_val, label=y_val, weight=sample_weight_val)
            else:
                eval_set = (X_val, y_val)

            if early_stopping_rounds:
                fit_params["early_stopping_rounds"] = early_stopping_rounds

        if sample_weight_train is not None:
            fit_params["sample_weight"] = sample_weight_train

        logger.info(
            f"Fitting {self.model_name} with params: {fit_model_params} "
            f"and fit_params: {list(fit_params.keys())}"
        )
        try:
            self.model.fit(X_train, y_train, eval_set=eval_set, **fit_params)
        except Exception as e:
            logger.error(f"Error during {self.model_name} fitting: {e}", exc_info=True)
            raise
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.feature_names_in_:
            X = X[self.feature_names_in_]
        return self.model.predict(X).flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.feature_names_in_:
            X = X[self.feature_names_in_]
        return self.model.predict_proba(X)

    def get_feature_importance(self, **kwargs) -> Optional[pd.Series]:
        if (
            self.model is None
            or not hasattr(self.model, "get_feature_importance")
            or not self.feature_names_in_
        ):
            logger.warning("Model not fitted or feature importances not available.")
            return None
        importances = self.model.get_feature_importance()
        return pd.Series(importances, index=self.feature_names_in_).sort_values(ascending=False)
