import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import AbstractModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(AbstractModel):

    def __init__(self, model_params: Dict[str, Any], model_name: str = "LogisticRegression"):
        super().__init__(model_params=model_params, model_name=model_name)
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[LogisticRegression] = None

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            sample_weight_train: Optional[pd.Series] = None,
            sample_weight_val: Optional[pd.Series] = None,
            **kwargs,
    ) -> "LogisticRegressionModel":
        super().fit(X_train, y_train)

        # 1. Scale the data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 2. Initialize and fit the model
        self.model = LogisticRegression(**self.model_params)
        logger.info(f"Fitting {self.model_name} with parameters: {self.model.get_params()}")
        self.model.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.feature_names_in_:
            X = X[self.feature_names_in_]

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.feature_names_in_:
            X = X[self.feature_names_in_]

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self, **kwargs) -> Optional[pd.Series]:
        if self.model is None or not hasattr(self.model, "coef_") or not self.feature_names_in_:
            return None

        if self.model.coef_.shape[0] > 1:
            importances = np.linalg.norm(self.model.coef_, axis=0)
        else:
            importances = np.abs(self.model.coef_.flatten())

        return pd.Series(importances, index=self.feature_names_in_).sort_values(ascending=False)
