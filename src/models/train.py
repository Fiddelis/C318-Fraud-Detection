"""Treino, comparação e persistência dos modelos de detecção de fraude.

Uso via linha de comando (a partir da raiz do projeto)::

    python -m src.models.train

Fluxo:
1. carrega a base processada e ordena por ``transaction_time``;
2. divisão temporal 70/15/15 (treino / validação / teste);
3. treina Random Forest (bagging) e HistGradientBoosting (boosting), cada um
   com busca aleatória de hiperparâmetros avaliada na validação;
4. escolhe o modelo pelo PR-AUC na validação e define o limiar de decisão
   também na validação;
5. avalia **uma única vez** no teste e salva o modelo em ``models/``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.data.cleaning import PROCESSED_DEFAULT, RAW_DEFAULT, build_processed
from src.features.selection import TARGET, feature_matrix
from src.models.evaluate import best_threshold, classification_metrics, metrics_table

MODEL_DEFAULT = Path("models/fraud_model.joblib")
REPORTS_DIR = Path("reports")
TIME_COLUMN = "transaction_time"
RANDOM_STATE = 42

CATEGORICAL = ["channel", "merchant_category", "country", "bin_country"]

# Agregados por usuário calculados pelo dataset sobre o período inteiro
# (avg_amount_user ~ média de amount do usuário em todos os 10 meses).
# Carregam um pouco de informação "do futuro"; usados no teste de robustez.
USER_AGGREGATES = ["avg_amount_user", "total_transactions_user", "amount_to_avg_ratio"]

N_JOBS = 4  # limitado de propósito: a máquina de desenvolvimento tem pouca memória livre

RF_SEARCH_SPACE = {
    "max_depth": [None, 12, 20],
    "min_samples_leaf": [1, 5, 20, 50],
    "max_features": ["sqrt", 0.5],
    "class_weight": [None, "balanced", "balanced_subsample"],
}

HGB_SEARCH_SPACE = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 50, 100, 200],
    "l2_regularization": [0.0, 0.1, 1.0],
    "max_depth": [None, 6, 10],
    "class_weight": [None, "balanced"],
}


# --------------------------------------------------------------------------- dados


def load_processed(path: str | Path = PROCESSED_DEFAULT) -> pd.DataFrame:
    """Carrega a base processada (gera a partir do bruto se ainda não existir)."""
    path = Path(path)
    if not path.exists():
        build_processed(RAW_DEFAULT, path)
    df = pd.read_csv(path, parse_dates=[TIME_COLUMN])
    return df.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame, train: float = 0.70, val: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide em ordem cronológica: o modelo nunca treina com o 'futuro'."""
    df = df.sort_values(TIME_COLUMN, kind="stable")
    i, j = int(len(df) * train), int(len(df) * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def split_xy(df: pd.DataFrame, drop: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    X, y = feature_matrix(df, target=TARGET)
    if drop:
        X = X.drop(columns=[c for c in drop if c in X.columns])
    return X, y


# --------------------------------------------------------------------------- modelos


def _columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [c for c in CATEGORICAL if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    return categorical, numeric


def _tree_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """Árvores não precisam de escala: categóricas viram códigos, numéricas passam direto."""
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )
    return ColumnTransformer([("cat", encoder, categorical), ("num", "passthrough", numeric)])


def build_random_forest(X: pd.DataFrame, **params) -> Pipeline:
    """Random Forest: muitas árvores em amostras bootstrap, decisão por votação."""
    categorical, numeric = _columns(X)
    defaults = dict(n_estimators=300, n_jobs=N_JOBS, random_state=RANDOM_STATE)
    clf = RandomForestClassifier(**{**defaults, **params})
    return Pipeline([("pre", _tree_preprocessor(categorical, numeric)), ("clf", clf)])


def build_hgb(X: pd.DataFrame, **params) -> Pipeline:
    """Gradient boosting com suporte nativo a categóricas e valores ausentes."""
    categorical, numeric = _columns(X)
    pre = _tree_preprocessor(categorical, numeric)
    defaults = dict(
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )
    clf = HistGradientBoostingClassifier(
        # As categóricas saem primeiro do ColumnTransformer.
        categorical_features=list(range(len(categorical))),
        **{**defaults, **params},
    )
    return Pipeline([("pre", pre), ("clf", clf)])


MODELS = {
    "random_forest": (build_random_forest, RF_SEARCH_SPACE),
    "hist_gradient_boosting": (build_hgb, HGB_SEARCH_SPACE),
}


def tune(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int,
) -> tuple[Pipeline, pd.DataFrame]:
    """Busca aleatória de hiperparâmetros avaliada pelo PR-AUC na validação temporal.

    Usamos a validação temporal (e não K-fold aleatório) para que a escolha dos
    hiperparâmetros respeite a ordem do tempo, como o modelo será usado.
    """
    builder, space = MODELS[name]
    trials = []
    best_model, best_score = None, -np.inf
    for i, params in enumerate(ParameterSampler(space, n_iter=n_iter, random_state=RANDOM_STATE)):
        start = time.perf_counter()
        model = builder(X_train, **params).fit(X_train, y_train)
        score = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
        elapsed = time.perf_counter() - start
        trials.append({"model": name, **params, "val_pr_auc": score, "fit_seconds": elapsed})
        print(f"  [{name} {i + 1}/{n_iter}] PR-AUC val={score:.4f} ({elapsed:.0f}s) {params}", flush=True)
        if score > best_score:
            best_model, best_score = model, score
    history = pd.DataFrame(trials).sort_values("val_pr_auc", ascending=False)
    return best_model, history


# --------------------------------------------------------------------------- execução


@dataclass
class TrainingResult:
    models: dict[str, Pipeline]
    thresholds: dict[str, float]
    val_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    best_name: str
    tuning_history: pd.DataFrame
    splits: dict[str, tuple[pd.DataFrame, pd.Series]] = field(repr=False)
    ablation: pd.DataFrame | None = None


def train_and_compare(
    df: pd.DataFrame,
    drop: list[str] | None = None,
    n_iter: int = 20,
    min_recall: float | None = None,
) -> TrainingResult:
    """Treina LogReg e HGB, escolhe limiar e modelo na validação, avalia no teste."""
    train, val, test = temporal_split(df)
    X_tr, y_tr = split_xy(train, drop)
    X_va, y_va = split_xy(val, drop)
    X_te, y_te = split_xy(test, drop)

    models, histories = {}, []
    for name in MODELS:
        models[name], hist = tune(name, X_tr, y_tr, X_va, y_va, n_iter=n_iter)
        histories.append(hist)
    history = pd.concat(histories, ignore_index=True)

    thresholds, val_res, test_res = {}, {}, {}
    for name, model in models.items():
        p_va = model.predict_proba(X_va)[:, 1]
        thresholds[name] = best_threshold(y_va, p_va, min_recall=min_recall)
        val_res[name] = classification_metrics(y_va, p_va, thresholds[name])
        p_te = model.predict_proba(X_te)[:, 1]
        test_res[name] = classification_metrics(y_te, p_te, thresholds[name])

    val_metrics = metrics_table(val_res)
    return TrainingResult(
        models=models,
        thresholds=thresholds,
        val_metrics=val_metrics,
        test_metrics=metrics_table(test_res),
        best_name=val_metrics.index[0],  # escolha feita na validação
        tuning_history=history,
        splits={"train": (X_tr, y_tr), "val": (X_va, y_va), "test": (X_te, y_te)},
    )


def save_model(result: TrainingResult, path: str | Path = MODEL_DEFAULT) -> Path:
    """Salva o modelo escolhido + limiar + lista de features num único artefato."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = result.best_name
    artifact = {
        "model": result.models[name],
        "model_name": name,
        "threshold": result.thresholds[name],
        "features": list(result.splits["train"][0].columns),
        "test_metrics": result.test_metrics.loc[name].to_dict(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(artifact, path)
    return path


def load_model(path: str | Path = MODEL_DEFAULT) -> dict:
    return joblib.load(path)


def predict_fraud(artifact: dict, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Aplica o modelo salvo a transações já limpas (saída de ``clean_transactions``)."""
    X = df_clean[artifact["features"]]
    proba = artifact["model"].predict_proba(X)[:, 1]
    return pd.DataFrame(
        {"fraud_probability": proba, "is_fraud_pred": (proba >= artifact["threshold"]).astype(int)},
        index=df_clean.index,
    )


def save_reports(result: TrainingResult, reports_dir: str | Path = REPORTS_DIR) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.concat(
        {"validation": result.val_metrics, "test": result.test_metrics}, names=["split", "model"]
    )
    if result.ablation is not None:
        comparison = pd.concat([comparison, result.ablation])
    comparison.round(4).to_csv(reports_dir / "model_comparison.csv")
    result.tuning_history.round(4).to_csv(reports_dir / "model_tuning.csv", index=False)
    summary = {
        "chosen_model": result.best_name,
        "threshold": result.thresholds[result.best_name],
        "test": result.test_metrics.loc[result.best_name].round(4).to_dict(),
    }
    (reports_dir / "model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    df = load_processed()
    print("== Treino com todas as features ==", flush=True)
    result = train_and_compare(df)

    # Robustez: quanto o desempenho depende dos agregados de período inteiro?
    print("\n== Treino sem agregados de usuário (robustez) ==", flush=True)
    ablation = train_and_compare(df, drop=USER_AGGREGATES, n_iter=8)
    result.ablation = pd.concat({"test_no_user_aggregates": ablation.test_metrics}, names=["split", "model"])

    path = save_model(result)
    save_reports(result)

    cols = ["pr_auc", "roc_auc", "threshold", "precision", "recall", "f1", "tp", "fp", "fn"]
    pd.set_option("display.width", 140)
    print("\n== Validação ==\n", result.val_metrics[cols].round(4))
    print("\n== Teste ==\n", result.test_metrics[cols].round(4))
    print("\n== Teste sem agregados de usuário ==\n", ablation.test_metrics[cols].round(4))
    print(f"\nModelo escolhido: {result.best_name} -> {path}")


if __name__ == "__main__":
    main()
