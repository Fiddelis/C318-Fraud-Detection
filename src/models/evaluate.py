"""Métricas e escolha de limiar para classificação desbalanceada."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_threshold(y_true, proba, min_recall: float | None = None) -> float:
    """Escolhe o limiar de decisão a partir das probabilidades previstas.

    - ``min_recall=None``: limiar que maximiza o F1.
    - ``min_recall=0.8`` (por exemplo): maior limiar - logo, maior precisão -
      que ainda garante pelo menos esse recall.

    Deve ser chamado sobre a **validação**, nunca sobre o teste.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall têm um ponto a mais que thresholds; o último é (1, 0).
    precision, recall = precision[:-1], recall[:-1]

    if min_recall is not None:
        ok = np.flatnonzero(recall >= min_recall)
        if ok.size == 0:
            raise ValueError(f"Nenhum limiar atinge recall >= {min_recall}.")
        return float(thresholds[ok[-1]])

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(thresholds[np.argmax(f1)])


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    """Métricas independentes de limiar (PR-AUC, ROC-AUC) + métricas no limiar."""
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Converte {nome_modelo: métricas} em tabela ordenada por PR-AUC."""
    return pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)
