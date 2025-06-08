import abc
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class AbstractModel(abc.ABC):
    def __init__(self, model_params: Dict[str, Any], model_name: str):
        self.model_params = model_params
        self.model_name = model_name
        self.model: Optional[Any] = None
        self.feature_names_in_: Optional[list[str]] = None
        self.n_features_in_: Optional[int] = None

    @abc.abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight_train: Optional[pd.Series] = None,
        sample_weight_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "AbstractModel":
        self.feature_names_in_ = X_train.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        pass

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def get_feature_importance(self, importance_type: str = "gain", **kwargs) -> Optional[pd.Series]:
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            if self.feature_names_in_ and len(self.feature_names_in_) == len(importances):
                return pd.Series(importances, index=self.feature_names_in_).sort_values(ascending=False)
        return None

    def save_model(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "model_params": self.model_params,
            "model_name": self.model_name,
            "model_artifact": self.model,
            "feature_names_in_": self.feature_names_in_,
            "n_features_in_": self.n_features_in_,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filepath: Path) -> "AbstractModel":
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(model_params=model_data["model_params"], model_name=model_data["model_name"])
        instance.model = model_data["model_artifact"]
        instance.feature_names_in_ = model_data.get("feature_names_in_")
        instance.n_features_in_ = model_data.get("n_features_in_")
        return instance

    def get_params(self) -> Dict[str, Any]:
        return self.model_params

    def set_params(self, **params) -> "AbstractModel":
        self.model_params.update(params)
        return self
