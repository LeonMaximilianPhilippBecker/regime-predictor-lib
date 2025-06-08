import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from regime_predictor_lib.supervised_learning.models.base import AbstractModel

logger = logging.getLogger(__name__)


def calculate_mean_decrease_accuracy(
    model: AbstractModel,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric_func: Callable,
    baseline_score: float,
    labels: Optional[List[int]] = None,
    n_repeats: int = 5,
    higher_is_better: bool = True,
) -> pd.Series:
    feature_importances: Dict[str, float] = {}

    for col in X_val.columns:
        shuffled_scores = []
        for _ in range(n_repeats):
            X_val_shuffled = X_val.copy()
            shuffled_col_values = X_val_shuffled[col].sample(frac=1, replace=False).values
            X_val_shuffled[col] = shuffled_col_values

            y_proba_shuffled = model.predict_proba(X_val_shuffled)

            score = metric_func(y_val, y_proba_shuffled, labels=labels)
            shuffled_scores.append(score)

        avg_shuffled_score = np.mean(shuffled_scores)

        if higher_is_better:
            importance = baseline_score - avg_shuffled_score
        else:
            importance = avg_shuffled_score - baseline_score

        feature_importances[col] = importance

    importance_series = pd.Series(feature_importances).sort_values(ascending=False)
    return importance_series
