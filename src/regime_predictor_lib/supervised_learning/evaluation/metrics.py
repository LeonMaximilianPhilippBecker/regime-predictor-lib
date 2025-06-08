import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def calculate_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: Optional[pd.DataFrame | np.ndarray] = None,
    labels: Optional[List[int]] = None,
    average_method: str = "macro",
    regime_importance_weights: Optional[Dict[int, float]] = None,
) -> Dict[str, float]:
    metrics = {}

    if labels is None:
        labels = sorted(list(set(np.concatenate((y_true, y_pred)).astype(int))))

    num_classes = len(labels)

    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics[f"f1_{average_method}"] = f1_score(
            y_true, y_pred, average=average_method, labels=labels, zero_division=0
        )
        metrics[f"precision_{average_method}"] = precision_score(
            y_true, y_pred, average=average_method, labels=labels, zero_division=0
        )
        metrics[f"recall_{average_method}"] = recall_score(
            y_true, y_pred, average=average_method, labels=labels, zero_division=0
        )
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

        if num_classes > 1:
            f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
            precision_per_class = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
            for i, label in enumerate(labels):
                metrics[f"f1_class_{label}"] = f1_per_class[i]
                metrics[f"precision_class_{label}"] = precision_per_class[i]
                metrics[f"recall_class_{label}"] = recall_per_class[i]

            if regime_importance_weights:
                weights_array = np.array([regime_importance_weights.get(label, 1.0) for label in labels])
                if weights_array.sum() > 0:
                    metrics["f1_custom_weighted"] = np.average(f1_per_class, weights=weights_array)
                    metrics["precision_custom_weighted"] = np.average(
                        precision_per_class, weights=weights_array
                    )
                    metrics["recall_custom_weighted"] = np.average(recall_per_class, weights=weights_array)
                else:
                    logger.warning("Sum of custom weights is zero, cannot compute weighted metrics.")

        if y_proba is not None:
            if y_proba.ndim == 1 and num_classes == 2:
                y_proba_adjusted = np.vstack([1 - y_proba, y_proba]).T
            elif y_proba.shape[1] != num_classes:
                logger.warning("y_proba shape mismatch. Skipping AUC and LogLoss.")
                y_proba_adjusted = None
            else:
                y_proba_adjusted = y_proba

            if y_proba_adjusted is not None:
                try:
                    metrics["log_loss"] = log_loss(y_true, y_proba_adjusted, labels=labels)
                except ValueError as e:
                    logger.warning(f"Could not calculate log_loss: {e}")

                if num_classes > 1 and len(np.unique(y_true)) > 1:
                    try:
                        metrics[f"auc_roc_ovr_{average_method}"] = roc_auc_score(
                            y_true, y_proba_adjusted, multi_class="ovr", average=average_method, labels=labels
                        )
                        auc_ovr_per_class = roc_auc_score(
                            y_true, y_proba_adjusted, multi_class="ovr", average=None, labels=labels
                        )
                        for i, label in enumerate(labels):
                            metrics[f"auc_roc_ovr_class_{label}"] = auc_ovr_per_class[i]
                    except ValueError as e:
                        logger.warning(f"Could not calculate multi-class AUC OvR: {e}")

    except Exception as e:
        logger.error(f"Error calculating multi-class metrics: {e}", exc_info=True)

    try:
        binary_metrics = calculate_binary_regime0_metrics(y_true, y_pred, y_proba, regime0_label=0)
        metrics.update(binary_metrics)
    except Exception as e:
        logger.error(f"Failed to generate binary metrics for Regime 0: {e}", exc_info=True)

    return metrics


def calculate_binary_regime0_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    regime0_label: int = 0,
) -> Dict[str, float]:
    metrics = {}
    prefix = f"binary_r{regime0_label}_"

    y_true_binary = (y_true == regime0_label).astype(int)
    y_pred_binary = (y_pred == regime0_label).astype(int)

    try:
        metrics[f"{prefix}f1"] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics[f"{prefix}precision"] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics[f"{prefix}recall"] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics[f"{prefix}accuracy"] = accuracy_score(y_true_binary, y_pred_binary)

        if y_proba is not None and y_proba.shape[1] > regime0_label:
            y_proba_binary = y_proba[:, regime0_label]

            if len(np.unique(y_true_binary)) > 1:
                metrics[f"{prefix}auc_roc"] = roc_auc_score(y_true_binary, y_proba_binary)
                metrics[f"{prefix}log_loss"] = log_loss(y_true_binary, y_proba_binary)
            else:
                metrics[f"{prefix}auc_roc"] = np.nan
                metrics[f"{prefix}log_loss"] = np.nan

    except Exception as e:
        logger.error(f"Could not calculate binary Regime {regime0_label} metrics: {e}", exc_info=True)

    return metrics


def calculate_regime_entry_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    target_regime_value: int = 0,
    early_tolerance_periods: int = 5,
    late_tolerance_periods: int = 10,
) -> Dict[str, Any]:
    if not y_true.index.equals(y_pred.index):
        logger.warning("Indices of y_true and y_pred do not match. Aligning.")
        common_index = y_true.index.intersection(y_pred.index)
        y_true, y_pred = y_true.loc[common_index], y_pred.loc[common_index]

    true_entries = y_true.index[(y_true == target_regime_value) & (y_true.shift(1) != target_regime_value)]
    pred_entries = y_pred.index[(y_pred == target_regime_value) & (y_pred.shift(1) != target_regime_value)]

    num_true_entries = len(true_entries)
    num_pred_entries = len(pred_entries)
    hits, lead_lags = [], []
    matched_preds = set()

    for true_time in true_entries:
        if isinstance(y_true.index, pd.DatetimeIndex):
            start_window = true_time - pd.Timedelta(days=early_tolerance_periods)
            end_window = true_time + pd.Timedelta(days=late_tolerance_periods)
        else:
            start_window = true_time - early_tolerance_periods
            end_window = true_time + late_tolerance_periods

        potential_matches = [
            p for p in pred_entries if start_window <= p <= end_window and p not in matched_preds
        ]

        if not potential_matches:
            continue

        if isinstance(y_true.index, pd.DatetimeIndex):
            time_diffs = [abs((p - true_time).total_seconds()) for p in potential_matches]
        else:
            time_diffs = [abs(p - true_time) for p in potential_matches]

        best_pred = potential_matches[np.argmin(time_diffs)]

        if isinstance(y_true.index, pd.DatetimeIndex):
            lead_lag = (best_pred - true_time).days
        else:
            lead_lag = best_pred - true_time

        hits.append(true_time)
        lead_lags.append(lead_lag)
        matched_preds.add(best_pred)

    num_hits = len(hits)
    precision = num_hits / num_pred_entries if num_pred_entries > 0 else 0.0
    recall = num_hits / num_true_entries if num_true_entries > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        f"num_true_regime{target_regime_value}_entries": num_true_entries,
        f"num_pred_regime{target_regime_value}_entries": num_pred_entries,
        f"num_hit_regime{target_regime_value}_entries": num_hits,
        f"precision_regime{target_regime_value}_entry": precision,
        f"recall_regime{target_regime_value}_entry": recall,
        f"f1_regime{target_regime_value}_entry": f1,
        "mean_lead_lag_hit_entries_periods": np.mean(lead_lags) if lead_lags else 0.0,
    }


def _find_spells(series: pd.Series, target_value: int, min_duration: int) -> List[Tuple[Any, Any, int]]:
    is_target = series == target_value
    blocks = (is_target.diff() != 0).cumsum()
    spells = []
    for _, group in series[is_target].groupby(blocks):
        if not group.empty:
            start, end = group.index[0], group.index[-1]
            duration = (
                (end - start).days + 1 if isinstance(series.index, pd.DatetimeIndex) else (end - start) + 1
            )
            if duration >= min_duration:
                spells.append((start, end, duration))
    return spells


def calculate_regime_spell_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    target_regime_value: int = 0,
    min_spell_duration: int = 5,
    start_early_tolerance: int = 10,
    end_late_tolerance: int = 10,
) -> Dict[str, Any]:
    if not y_true.index.equals(y_pred.index):
        logger.warning("Indices of y_true and y_pred do not match for spell metrics. Aligning.")
        common_index = y_true.index.intersection(y_pred.index)
        y_true, y_pred = y_true.loc[common_index], y_pred.loc[common_index]

    true_spells = _find_spells(y_true, target_regime_value, min_spell_duration)
    pred_spells = _find_spells(y_pred, target_regime_value, min_spell_duration)

    num_true_spells = len(true_spells)
    num_pred_spells = len(pred_spells)

    if num_true_spells == 0:
        return {
            f"num_true_regime{target_regime_value}_spells": 0,
            f"num_pred_regime{target_regime_value}_spells": num_pred_spells,
        }

    matched_spells = []
    available_preds = set(range(num_pred_spells))

    for i, (true_start, true_end, _) in enumerate(true_spells):
        best_overlap, best_match_idx = 0, -1
        for j in available_preds:
            pred_start, pred_end, _ = pred_spells[j]
            overlap_start, overlap_end = max(true_start, pred_start), min(true_end, pred_end)

            if overlap_end >= overlap_start:
                overlap_duration = (
                    (overlap_end - overlap_start).days + 1
                    if isinstance(y_true.index, pd.DatetimeIndex)
                    else (overlap_end - overlap_start) + 1
                )
                if overlap_duration > best_overlap:
                    best_overlap, best_match_idx = overlap_duration, j

        if best_match_idx != -1:
            matched_spells.append({"true_spell": true_spells[i], "pred_spell": pred_spells[best_match_idx]})
            available_preds.remove(best_match_idx)

    num_spells_detected = len(matched_spells)
    recall = num_spells_detected / num_true_spells if num_true_spells > 0 else 0.0
    precision = num_spells_detected / num_pred_spells if num_pred_spells > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    start_diffs, end_diffs, duration_ratios, starts_in_tol, ends_in_tol = [], [], [], 0, 0
    for match in matched_spells:
        true_start, true_end, true_dur = match["true_spell"]
        pred_start, pred_end, pred_dur = match["pred_spell"]

        is_dt_index = isinstance(y_true.index, pd.DatetimeIndex)
        start_diffs.append((pred_start - true_start).days if is_dt_index else (pred_start - true_start))
        end_diffs.append((pred_end - true_end).days if is_dt_index else (pred_end - true_end))
        duration_ratios.append(pred_dur / true_dur if true_dur > 0 else 0)

        start_tol_begin = (
            true_start - pd.Timedelta(days=start_early_tolerance)
            if is_dt_index
            else true_start - start_early_tolerance
        )
        if start_tol_begin <= pred_start <= true_start:
            starts_in_tol += 1

        end_tol_end = (
            true_end + pd.Timedelta(days=end_late_tolerance) if is_dt_index else true_end + end_late_tolerance
        )
        if true_end <= pred_end <= end_tol_end:
            ends_in_tol += 1

    return {
        f"num_true_regime{target_regime_value}_spells": num_true_spells,
        f"num_pred_regime{target_regime_value}_spells": num_pred_spells,
        "num_spells_detected": num_spells_detected,
        "spell_detection_recall": recall,
        "spell_detection_precision": precision,
        "spell_detection_f1": f1,
        "avg_start_diff_periods": np.mean(start_diffs) if start_diffs else 0.0,
        "avg_end_diff_periods": np.mean(end_diffs) if end_diffs else 0.0,
        "avg_duration_ratio": np.mean(duration_ratios) if duration_ratios else 0.0,
        "prop_starts_within_tolerance": starts_in_tol / num_spells_detected if num_spells_detected > 0 else 0.0,
        "prop_ends_within_tolerance": ends_in_tol / num_spells_detected if num_spells_detected > 0 else 0.0,
    }


def get_sklearn_classification_report(y_true, y_pred, labels=None, target_names=None, **kwargs):
    return classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0, **kwargs
    )


def get_sklearn_confusion_matrix(y_true, y_pred, labels=None, normalize=None):
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
