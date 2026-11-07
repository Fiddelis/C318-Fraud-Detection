"""Módulos de engenharia e seleção de features."""

from .selection import (
    correlation_with_target,
    encode_for_scoring,
    feature_matrix,
    fraud_rate_by_category,
    mutual_information_scores,
    rank_features,
)

__all__ = [
    "correlation_with_target",
    "encode_for_scoring",
    "feature_matrix",
    "fraud_rate_by_category",
    "mutual_information_scores",
    "rank_features",
]
