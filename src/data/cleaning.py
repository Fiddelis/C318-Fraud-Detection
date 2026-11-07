"""Carregamento, limpeza e preparação do dataset de transações."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DEFAULT = Path("data/raw/transactions.csv")
PROCESSED_DEFAULT = Path("data/processed/transactions_clean.csv")

ID_COLUMNS = ("transaction_id", "user_id")
TARGET = "is_fraud"


def load_raw(path: str | Path = RAW_DEFAULT) -> pd.DataFrame:
    """Carrega o CSV bruto de transações."""
    return pd.read_csv(path)


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo de valores ausentes por coluna."""
    missing = df.isna().sum()
    percent = (missing / len(df) * 100).round(4)
    return (
        pd.DataFrame({"missing": missing, "percent": percent})
        .query("missing > 0")
        .sort_values("missing", ascending=False)
    )


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza mínima e cria features derivadas úteis à modelagem.

    O dataset já vem sem nulos e sem duplicatas. A limpeza aqui foca em:
    - tipagem correta de data/hora e flags binárias;
    - features derivadas com sinal comportamental conhecido;
    - remoção de identificadores que não devem entrar no modelo.

    Valores extremos de ``amount`` e ``shipping_distance_km`` são mantidos:
    correlacionam fortemente com fraude e não devem ser tratados como erro.
    """
    out = df.copy()

    out["transaction_time"] = pd.to_datetime(out["transaction_time"], utc=True)

    for col in ("promo_used", "avs_match", "cvv_result", "three_ds_flag", TARGET):
        out[col] = out[col].astype("int8")

    out["hour"] = out["transaction_time"].dt.hour.astype("int8")
    out["day_of_week"] = out["transaction_time"].dt.day_of_week.astype("int8")
    out["is_night"] = out["hour"].between(0, 5).astype("int8")
    out["country_bin_mismatch"] = (out["country"] != out["bin_country"]).astype("int8")
    out["amount_to_avg_ratio"] = (
        out["amount"] / out["avg_amount_user"].replace(0, np.nan)
    ).astype("float64")

    # Identificadores não entram como feature preditiva.
    out = out.drop(columns=list(ID_COLUMNS))

    return out.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: str | Path = PROCESSED_DEFAULT) -> Path:
    """Persiste a base limpa em CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def build_processed(
    raw_path: str | Path = RAW_DEFAULT,
    processed_path: str | Path = PROCESSED_DEFAULT,
) -> pd.DataFrame:
    """Pipeline completo: carrega, limpa e salva a base processada."""
    raw = load_raw(raw_path)
    clean = clean_transactions(raw)
    save_processed(clean, processed_path)
    return clean
