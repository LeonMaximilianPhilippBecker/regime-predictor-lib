# PATH: src/regime_predictor_lib/supervised_learning/models/gaussian_process.py
import logging
from typing import Any, Dict, List, Optional

import GPy
import numpy as np
import pandas as pd
from GPy.models import SparseGPClassification
from sklearn.preprocessing import LabelBinarizer

from .base import AbstractModel

logger = logging.getLogger(__name__)


class _SingleBinaryGP(AbstractModel):
    def __init__(self, model_params: Dict[str, Any]):
        super().__init__(model_params=model_params, model_name="BinarySparseGP")
        self.kernel: Optional[GPy.kern.Kern] = None
        self.model: Optional[SparseGPClassification] = None

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            **kwargs,
    ) -> "_SingleBinaryGP":
        super().fit(X_train, y_train)

        kernel_name = self.model_params.get("kernel", "RBF")
        kernel_params = self.model_params.get("kernel_params", {})
        input_dim = X_train.shape[1]

        if kernel_name.upper() == "RBF":
            self.kernel = GPy.kern.RBF(input_dim=input_dim, **kernel_params)
        elif kernel_name.upper() == "MATERN52":
            self.kernel = GPy.kern.Matern52(input_dim=input_dim, **kernel_params)
        else:
            self.kernel = GPy.kern.RBF(input_dim=input_dim)

        y_train_reshaped = y_train.values.reshape(-1, 1)
        num_inducing = self.model_params.get("num_inducing", 100)

        self.model = SparseGPClassification(
            X=X_train.values, Y=y_train_reshaped, kernel=self.kernel, num_inducing=num_inducing
        )

        optimizer = self.model_params.get("optimizer", "scg")
        max_iters = self.model_params.get("max_iters", 1000)
        messages = self.model_params.get("messages", False)

        self.model.optimize(optimizer, max_iters=max_iters, messages=messages)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Predict should be called from the OneVsRestGPModel wrapper.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.feature_names_in_:
            X = X[self.feature_names_in_]
        # This returns the probability of Y=1
        prob_class_1, _ = self.model.predict_quantiles(X.values, quantiles=(50,))
        return prob_class_1


class OneVsRestGPModel(AbstractModel):
    def __init__(self, model_params: Dict[str, Any], model_name: str = "OneVsRestGP"):
        super().__init__(model_params=model_params, model_name=model_name)
        self.estimators_: List[_SingleBinaryGP] = []
        self.label_binarizer_: Optional[LabelBinarizer] = None

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            sample_weight_train: Optional[pd.Series] = None,
            sample_weight_val: Optional[pd.Series] = None,
            **kwargs,
    ) -> "OneVsRestGPModel":
        super().fit(X_train, y_train)
        self.label_binarizer_ = LabelBinarizer(sparse_output=False)
        Y_one_hot = self.label_binarizer_.fit_transform(y_train)

        self.estimators_ = []
        for i, class_label in enumerate(self.label_binarizer_.classes_):
            logger.info(
                f"--- Fitting binary classifier for class {class_label} "
                f"(model {i + 1}/{Y_one_hot.shape[1]}) ---")

            estimator = _SingleBinaryGP(self.model_params)

            y_train_binary = pd.Series(Y_one_hot[:, i], index=y_train.index)
            estimator.fit(X_train, y_train_binary)

            self.estimators_.append(estimator)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Model has not been fitted yet.")

        all_probas = np.zeros((len(X), len(self.estimators_)))
        for i, estimator in enumerate(self.estimators_):
            proba_class_1 = estimator.predict_proba(X)
            all_probas[:, i] = proba_class_1.flatten()

        sum_probas = all_probas.sum(axis=1)
        sum_probas[sum_probas == 0] = 1

        normalized_probas = all_probas / sum_probas[:, np.newaxis]
        return normalized_probas

    def get_ard_lengthscales(self) -> pd.DataFrame:
        if not self.estimators_:
            return pd.DataFrame()

        lengthscales_dict = {}
        for i, estimator in enumerate(self.estimators_):
            class_label = self.label_binarizer_.classes_[i]
            if hasattr(estimator.model.kern, 'lengthscale'):
                lengthscales_dict[
                    f'lengthscale_class_{class_label}'] = estimator.model.kern.lengthscale.values.flatten()

        return pd.DataFrame(lengthscales_dict, index=self.feature_names_in_)
