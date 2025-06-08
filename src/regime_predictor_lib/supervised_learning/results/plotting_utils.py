import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve


def plot_confusion_matrix(cm, class_names, title, normalize=False, cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize=(10, 8))

    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        annot = np.array(
            [f"{val_norm:.2f}\n({val_raw})" for val_norm, val_raw in zip(cm_normalized.flatten(), cm.flatten())]
        )
        annot = annot.reshape(cm.shape)
        fmt = ""
        heatmap_data = cm_normalized
        cbar_label = "Normalized Proportion"
    else:
        annot = cm
        fmt = "d"
        heatmap_data = cm
        cbar_label = "Raw Count"

    sns.heatmap(heatmap_data, annot=annot, fmt=fmt, cmap=cmap, ax=ax, cbar=True, cbar_kws={"label": cbar_label})

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel("True label", fontsize=13)
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=0)
    plt.tight_layout()
    return fig


def plot_roc_curves(y_true, y_probas_dict, class_labels, title):
    fig, axes = plt.subplots(1, len(class_labels), figsize=(7 * len(class_labels), 6), sharey=True)
    if len(class_labels) <= 1:
        axes = [axes]

    fig.suptitle(title, fontsize=16)

    for i, class_label in enumerate(class_labels):
        ax = axes[i]
        for model_name, y_proba in y_probas_dict.items():
            y_true_binary = (y_true == class_label).astype(int)

            if np.sum(y_true_binary) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No true samples for Regime {class_label}\nin this data split.",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red",
                )
                ax.set_title(f"ROC for Regime {class_label}")
                continue

            fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:0.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Chance (AUC = 0.500)")
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC for Regime {class_label}")
        ax.legend(loc="lower right")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_feature_importances(importance_df, top_n=20, title="Feature Importances"):
    if importance_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No feature importances available.", ha="center", va="center")
        ax.set_title(title)
        return fig

    top_features = importance_df.nlargest(top_n, "importance").sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) / 2.5)))
    ax.barh(
        top_features["feature"],
        top_features["importance"],
        color=sns.color_palette("viridis", len(top_features)),
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


def plot_feature_importance_stability(
    mda_per_fold_df: pd.DataFrame, top_n: int = 25, title="Feature Importance Stability"
):
    if mda_per_fold_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No MDA results available.", ha="center", va="center")
        ax.set_title(title)
        return fig

    mean_importances = mda_per_fold_df.groupby("feature")["importance"].mean().sort_values(ascending=False)
    top_features = mean_importances.head(top_n).index.tolist()

    df_top = mda_per_fold_df[mda_per_fold_df["feature"].isin(top_features)]

    fig, ax = plt.subplots(figsize=(12, max(8, len(top_features) * 0.4)))
    sns.boxplot(
        x="importance",
        y="feature",
        data=df_top,
        order=top_features,
        orient="h",
        palette="viridis",
        ax=ax,
    )

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("MDA Importance (Increase in LogLoss)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.axvline(x=0, color="r", linestyle="--", linewidth=1.2, label="Zero Importance")
    ax.legend(loc="lower right")

    plt.tight_layout()
    return fig
