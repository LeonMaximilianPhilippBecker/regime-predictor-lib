# PATH: src/regime_predictor_lib/supervised_learning/training/trainer.py
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from ..evaluation.metrics import calculate_classification_metrics
from ..feature_importance.mda import calculate_mean_decrease_accuracy
from ..models.base import AbstractModel
from ..results.result_saver import ResultSaver

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
            self,
            model_wrapper: AbstractModel,
            cv_splitter: BaseCrossValidator,
            result_saver: ResultSaver,
            theme_name: str,
            label_encoder: Optional[LabelEncoder] = None,
    ):
        self.model_wrapper = model_wrapper
        self.cv_splitter = cv_splitter
        self.result_saver = result_saver
        self.theme_name = theme_name
        self.model_config_name = model_wrapper.model_name
        self.label_encoder = label_encoder

        self.cv_fold_metrics: List[Dict[str, Any]] = []
        self.mda_fold_results: List[pd.DataFrame] = []
        self.oof_predictions: Optional[pd.DataFrame] = None
        self.training_logs: str = ""

        logger.info(f"ModelTrainer initialized for {self.theme_name} - {self.model_config_name}")

    def _log(self, message: str, level: int = logging.INFO):
        logger.log(level, message)
        self.training_logs += (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {logging.getLevelName(level)} - {message}\n"
        )

    def _encode_target(self, y: pd.Series) -> Tuple[pd.Series, List[int]]:
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            self._log(f"Fitted LabelEncoder. Classes: {self.label_encoder.classes_}")
        else:
            y_encoded = self.label_encoder.transform(y)
            self._log(f"Used existing LabelEncoder. Classes: {self.label_encoder.classes_}")
        return pd.Series(y_encoded, index=y.index, name=y.name), list(range(len(self.label_encoder.classes_)))

    def run_cross_validation(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            groups: Optional[pd.Series] = None,
            fit_params: Optional[Dict[str, Any]] = None,
            use_class_weights: bool = True,
            run_mda: bool = True,
            mda_n_repeats: int = 5,
    ) -> Dict[str, Any]:
        self._log(f"Starting cross-validation for {self.model_config_name} on {self.theme_name}...")
        if fit_params is None:
            fit_params = {}

        oof_index = X.index
        if not isinstance(oof_index, pd.DatetimeIndex):
            self._log(
                f"Warning: X.index is not a DatetimeIndex (type: {type(oof_index).__name__})."
                f" OOF results will use this index.",
                logging.WARNING,
            )

        y_processed, class_labels_encoded = y, sorted(y.unique().tolist())
        if self.label_encoder:
            class_labels_encoded = list(range(len(self.label_encoder.classes_)))

        oof_preds_list, oof_probas_list, oof_indices_list, oof_true_list = [], [], [], []
        self.cv_fold_metrics = []
        self.mda_fold_results = []

        for fold_idx, (train_indices, val_indices) in enumerate(
                self.cv_splitter.split(X, y_processed, groups=groups)
        ):
            self._log(f"--- Fold {fold_idx + 1}/{self.cv_splitter.get_n_splits()} ---")

            if len(train_indices) == 0 or len(val_indices) == 0:
                self._log(
                    f"Skipping fold {fold_idx + 1} due to empty train or validation set.", logging.WARNING
                )
                continue

            X_train_fold, X_val_fold = X.iloc[train_indices], X.iloc[val_indices]
            y_train_fold, y_val_fold = y_processed.iloc[train_indices], y_processed.iloc[val_indices]

            sample_weight_train_fold = None
            if use_class_weights:
                sample_weight_train_fold = pd.Series(
                    compute_sample_weight(class_weight="balanced", y=y_train_fold), index=y_train_fold.index
                )

            start_time = time.time()
            current_model = self.model_wrapper
            current_model.set_params(**self.model_wrapper.get_params())
            current_model.fit(
                X_train_fold,
                y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                sample_weight_train=sample_weight_train_fold,
                **fit_params,
            )
            train_time = time.time() - start_time
            self._log(f"Fold {fold_idx + 1} training time: {train_time:.2f}s")

            y_pred_val = current_model.predict(X_val_fold)
            y_proba_val = current_model.predict_proba(X_val_fold)

            # --- THIS IS THE KEY CHANGE ---
            # Get the classes the model was *actually* trained on in this fold
            fold_trained_classes = current_model.model.classes_

            # Create the full-sized DataFrame first
            current_val_indices = oof_index[val_indices]
            oof_proba_df = pd.DataFrame(
                0.0,
                index=current_val_indices,
                columns=[f"proba_class_{i}" for i in class_labels_encoded]
            )

            # Create a temporary DataFrame with the probabilities we got
            temp_proba_df = pd.DataFrame(
                y_proba_val,
                index=current_val_indices,
                columns=[f"proba_class_{c}" for c in fold_trained_classes]
            )

            oof_proba_df.update(temp_proba_df)

            oof_indices_list.append(current_val_indices)
            oof_true_list.append(y_val_fold)
            oof_preds_list.append(pd.Series(y_pred_val, index=current_val_indices))
            oof_probas_list.append(oof_proba_df)  # Append the correctly shaped DataFrame

            fold_metrics_results = calculate_classification_metrics(
                y_val_fold, y_pred_val, y_proba_val, labels=class_labels_encoded
            )
            fold_metrics_results["fold"] = fold_idx + 1
            fold_metrics_results["training_time_seconds"] = train_time
            self.cv_fold_metrics.append(fold_metrics_results)

            if run_mda:
                try:
                    baseline_logloss = log_loss(y_val_fold, y_proba_val, labels=class_labels_encoded)
                    self._log(f"Fold {fold_idx + 1} baseline log_loss for MDA: {baseline_logloss:.4f}")

                    mda_scores_fold = calculate_mean_decrease_accuracy(
                        model=current_model,
                        X_val=X_val_fold,
                        y_val=y_val_fold,
                        metric_func=log_loss,
                        baseline_score=baseline_logloss,
                        labels=class_labels_encoded,
                        n_repeats=mda_n_repeats,
                        higher_is_better=False,
                    )

                    mda_df_fold = mda_scores_fold.reset_index()
                    mda_df_fold.columns = ["feature", "importance"]
                    mda_df_fold["fold"] = fold_idx + 1
                    self.mda_fold_results.append(mda_df_fold)
                    self._log(f"Fold {fold_idx + 1} MDA calculated. Top 3: {mda_scores_fold.head(3).to_dict()}")
                except ValueError as e:
                    self._log(f"Could not run MDA for fold {fold_idx + 1}: {e}", logging.ERROR)

        # Aggregate and Save Results
        if self.mda_fold_results:
            all_mda_df = pd.concat(self.mda_fold_results, ignore_index=True)
            self.result_saver.save_mda_results(
                theme_name=self.theme_name, model_name=self.model_config_name, mda_df=all_mda_df
            )

        if oof_preds_list:
            oof_true_combined = pd.concat(oof_true_list).rename("true_label")
            oof_pred_combined = pd.concat(oof_preds_list).rename("predicted_label")
            oof_proba_combined = pd.concat(oof_probas_list)
            self.oof_predictions = pd.concat([oof_true_combined, oof_pred_combined, oof_proba_combined], axis=1)

            if self.label_encoder:
                self.oof_predictions["true_label_orig"] = self.label_encoder.inverse_transform(
                    self.oof_predictions["true_label"].astype(int)
                )
                self.oof_predictions["predicted_label_orig"] = self.label_encoder.inverse_transform(
                    self.oof_predictions["predicted_label"].astype(int)
                )

        # The rest of the function remains the same...
        cv_metrics_df = pd.DataFrame(self.cv_fold_metrics)
        aggregated_metrics = {}
        for col in cv_metrics_df.columns:
            if col != "fold" and pd.api.types.is_numeric_dtype(cv_metrics_df[col]):
                aggregated_metrics[f"{col}_mean"] = cv_metrics_df[col].mean()
                aggregated_metrics[f"{col}_std"] = cv_metrics_df[col].std()

        self.result_saver.save_cv_metrics(
            theme_name=self.theme_name,
            model_name=self.model_config_name,
            fold_metrics_df=cv_metrics_df,
            aggregated_metrics_dict=aggregated_metrics,
            oof_predictions_df=self.oof_predictions.reset_index() if self.oof_predictions is not None else None,
        )
        self.result_saver.save_training_log(
            theme_name=self.theme_name, model_name=self.model_config_name, log_content=self.training_logs
        )

        return aggregated_metrics

    def train_final_model(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            sample_weight: Optional[pd.Series] = None,
            fit_params: Optional[Dict[str, Any]] = None,
    ) -> AbstractModel:
        self._log(f"Training final model {self.model_config_name} on {self.theme_name}...")
        if fit_params is None:
            fit_params = {}

        y_processed = y.copy()
        if self.label_encoder:
            if not all(item in self.label_encoder.classes_ for item in y.unique()):
                self._log("New classes found in final training set. Refitting encoder.", logging.WARNING)
                self.label_encoder.fit(y)
            y_processed = pd.Series(self.label_encoder.transform(y), index=y.index, name=y.name)

        start_time = time.time()
        final_model_instance = self.model_wrapper
        final_model_instance.set_params(**self.model_wrapper.get_params())

        final_model_instance.fit(X, y_processed, sample_weight_train=sample_weight, **fit_params)
        train_time = time.time() - start_time
        self._log(f"Final model training time: {train_time:.2f}s")

        self.trained_final_model = final_model_instance

        model_filename = f"{self.theme_name}_{self.model_config_name}.pkl"
        self.result_saver.save_model_artifact(
            theme_name=self.theme_name,
            model_name=self.model_config_name,
            model_object=self.trained_final_model,
            filename=model_filename,
        )

        if self.label_encoder:
            le_filename = f"{self.theme_name}_{self.model_config_name}_label_encoder.pkl"
            le_dir = self.result_saver._get_output_path(
                self.theme_name, self.model_config_name, is_model_artifact=True
            )
            le_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            with open(le_dir / le_filename, "wb") as f:
                pickle.dump(self.label_encoder, f)
            self._log(f"Saved LabelEncoder to {le_dir / le_filename}")

        self.result_saver.save_training_log(
            theme_name=self.theme_name,
            model_name=self.model_config_name,
            log_content=self.training_logs,
        )
        return self.trained_final_model
