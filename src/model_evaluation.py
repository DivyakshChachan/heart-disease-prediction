from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_fscore_support, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classifier(model, X_test, y_test) -> Dict[str, Any]:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred)

    return {
        "roc_auc": float(roc_auc),
        "fpr": fpr,
        "tpr": tpr,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cm": cm,
        "report": classification_report(y_test, y_pred, output_dict=True)
    }


def plot_roc_curve(metrics: Dict[str, Any], ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(metrics["fpr"], metrics["tpr"], label=f"ROC AUC={metrics['roc_auc']:.3f}")
    ax.plot([0,1], [0,1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return ax


def plot_confusion_matrix(metrics: Dict[str, Any], ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return ax
