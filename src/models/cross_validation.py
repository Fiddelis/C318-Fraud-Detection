"""Validação cruzada temporal (forward-chaining) dos modelos já ajustados.

Uso (a partir da raiz do projeto, depois de ``python -m src.models.train``)::

    python -m src.models.cross_validation

Por que temporal e não K-fold comum: o K-fold embaralha as transações, então o
modelo treinaria com meses posteriores aos que avalia - um cenário que não
existe em produção e que infla as métricas. O ``TimeSeriesSplit`` treina sempre
com o passado e testa no bloco seguinte, com a janela de treino crescendo.

A validação cruzada usa apenas treino + validação (85% iniciais da linha do
tempo); o teste final continua intocado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models.evaluate import classification_metrics
from src.models.train import MODELS, REPORTS_DIR, load_processed, split_xy, temporal_split

INT_PARAMS = {"max_depth", "min_samples_leaf", "max_leaf_nodes", "n_estimators"}


def _clean_value(key: str, value):
    """Desfaz o que o CSV faz com os hiperparâmetros (None -> NaN, int -> float, 0.5 -> '0.5')."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value  # ex.: 'sqrt', 'balanced'
    if key in INT_PARAMS:
        return int(value)
    return value


def best_params(tuning: pd.DataFrame, name: str) -> dict:
    """Melhor configuração de cada modelo segundo o PR-AUC de validação."""
    row = tuning[tuning["model"] == name].sort_values("val_pr_auc", ascending=False).iloc[0]
    _, space = MODELS[name]
    return {key: _clean_value(key, row[key]) for key in space}


def validation_thresholds(comparison: pd.DataFrame) -> dict[str, float]:
    """Limiares escolhidos na validação durante o treino (mantidos fixos aqui)."""
    val = comparison[comparison["split"] == "validation"].set_index("model")
    return val["threshold"].to_dict()


def temporal_cross_validation(
    df: pd.DataFrame,
    params: dict[str, dict],
    thresholds: dict[str, float],
    n_splits: int = 5,
) -> pd.DataFrame:
    train, val, _test = temporal_split(df)
    dev = pd.concat([train, val])  # já em ordem cronológica
    X, y = split_xy(dev)
    times = dev["transaction_time"]

    rows = []
    for fold, (tr_idx, te_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        for name, (builder, _space) in MODELS.items():
            model = builder(X_tr, **params[name]).fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            metrics = classification_metrics(y_te, proba, thresholds[name])
            rows.append(
                {
                    "fold": fold,
                    "model": name,
                    "train_rows": len(tr_idx),
                    "train_until": times.iloc[tr_idx[-1]].date(),
                    "test_from": times.iloc[te_idx[0]].date(),
                    "test_until": times.iloc[te_idx[-1]].date(),
                    "test_frauds": int(y_te.sum()),
                    **metrics,
                }
            )
            print(
                f"  fold {fold} {name}: PR-AUC={metrics['pr_auc']:.4f} "
                f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    cols = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    summary = folds.groupby("model")[cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.sort_values("pr_auc_mean", ascending=False)


def main(reports_dir: str | Path = REPORTS_DIR) -> None:
    reports_dir = Path(reports_dir)
    tuning = pd.read_csv(reports_dir / "model_tuning.csv")
    comparison = pd.read_csv(reports_dir / "model_comparison.csv")

    params = {name: best_params(tuning, name) for name in MODELS}
    thresholds = validation_thresholds(comparison)
    for name in MODELS:
        print(f"{name}: {params[name]} | limiar={thresholds[name]:.4f}")

    folds = temporal_cross_validation(load_processed(), params, thresholds)
    summary = summarize(folds)

    folds.round(4).to_csv(reports_dir / "cross_validation_folds.csv", index=False)
    summary.round(4).to_csv(reports_dir / "cross_validation_summary.csv")

    pd.set_option("display.width", 160)
    print("\n== Resumo (média e desvio entre folds) ==\n", summary.round(4))
    # Diferença pareada fold a fold: o empate entre os modelos se mantém?
    wide = folds.pivot(index="fold", columns="model", values="pr_auc")
    diff = wide["random_forest"] - wide["hist_gradient_boosting"]
    print(
        f"\nPR-AUC RF - HGB por fold: {np.round(diff.values, 4)} "
        f"(média {diff.mean():+.4f}, RF vence {int((diff > 0).sum())}/{len(diff)})"
    )


if __name__ == "__main__":
    main()
