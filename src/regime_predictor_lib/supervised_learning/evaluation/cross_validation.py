import logging
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

logger = logging.getLogger(__name__)


class PurgedKFold(_BaseKFold):
    def __init__(
        self,
        n_splits: int = 5,
        purge_length: int = 0,
        embargo_length: int = 0,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise ValueError("n_splits must be an integer greater than 1.")
        if not isinstance(purge_length, int) or purge_length < 0:
            raise ValueError("purge_length must be a non-negative integer.")
        if not isinstance(embargo_length, int) or embargo_length < 0:
            raise ValueError("embargo_length must be a non-negative integer.")

        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Optional[pd.Series | np.ndarray] = None,
        groups: Optional[pd.Series | np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if isinstance(X, pd.DataFrame):
            indices = np.arange(X.shape[0])
        elif isinstance(X, np.ndarray):
            indices = np.arange(X.shape[0])
        else:
            raise ValueError("X must be a pandas DataFrame or a numpy array.")

        n_samples = len(indices)

        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than " f"the number of samples={n_samples}."
            )

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current_sample_idx = 0
        for fold_i in range(self.n_splits):
            test_start_idx = current_sample_idx
            test_end_idx = test_start_idx + fold_sizes[fold_i]
            test_indices = indices[test_start_idx:test_end_idx]
            train_end_purged = test_start_idx - self.purge_length
            train_indices_before = indices[indices < train_end_purged]
            train_indices = train_indices_before
            if len(train_indices) == 0:
                logger.warning(
                    f"Fold {fold_i + 1}/{self.n_splits} has an empty training set due to purging. "
                    f"Consider reducing purge_length or n_splits, or ensure enough data."
                )
            if len(test_indices) == 0:
                logger.warning(f"Fold {fold_i + 1}/{self.n_splits} has an empty test set.")

            yield train_indices, test_indices

            current_sample_idx = test_end_idx
