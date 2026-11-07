"""Análise e ranking de importância de features para detecção de fraude."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder

TARGET = "is_fraud"
DROP_FROM_FEATURES = ("transaction_time",)


def feature_matrix(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X/y removendo alvo e colunas não preditivas (ex.: timestamp bruto)."""
    cols_to_drop = [c for c in (target, *DROP_FROM_FEATURES) if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target].astype(int)
    return X, y


def encode_for_scoring(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot nas categóricas; numéricas passam adiante sem alteração."""
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    if not categorical:
        return X.copy()

    encoder = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    transformed = encoder.fit_transform(X)
    cat_names = encoder.named_transformers_["cat"].get_feature_names_out(categorical).tolist()
    columns = cat_names + numeric
    return pd.DataFrame(transformed, columns=columns, index=X.index)


def correlation_with_target(
    df: pd.DataFrame,
    target: str = TARGET,
    method: str = "pearson",
) -> pd.Series:
    """Correlação de features numéricas com o alvo."""
    numeric = df.select_dtypes(include="number")
    if target not in numeric.columns:
        raise ValueError(f"Coluna alvo '{target}' não encontrada entre as numéricas.")
    return (
        numeric.corr(method=method)[target]
        .drop(labels=[target])
        .sort_values(key=abs, ascending=False)
    )


def mutual_information_scores(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> pd.Series:
    """Mutual information entre features (já numéricas/encoded) e o alvo."""
    scores = mutual_info_classif(X, y, random_state=random_state)
    return pd.Series(scores, index=X.columns, name="mutual_info").sort_values(ascending=False)


def fraud_rate_by_category(df: pd.DataFrame, column: str, target: str = TARGET) -> pd.DataFrame:
    """Taxa de fraude e volume por categoria - útil para features nominais."""
    summary = (
        df.groupby(column, dropna=False)
        .agg(n=(target, "size"), fraud_rate=(target, "mean"))
        .sort_values("fraud_rate", ascending=False)
    )
    summary["fraud_rate"] = summary["fraud_rate"].round(4)
    return summary


def rank_features(df: pd.DataFrame, target: str = TARGET, random_state: int = 42) -> pd.DataFrame:
    """Combina |correlação| (numéricas) e mutual information (matriz encoded)."""
    X, y = feature_matrix(df, target=target)
    X_enc = encode_for_scoring(X)

    corr = correlation_with_target(pd.concat([X.select_dtypes("number"), y], axis=1), target=target)
    mi = mutual_information_scores(X_enc, y, random_state=random_state)

    ranking = pd.DataFrame({"mutual_info": mi})
    ranking["abs_corr"] = corr.reindex(ranking.index).abs()
    return ranking.sort_values(["mutual_info", "abs_corr"], ascending=False)
