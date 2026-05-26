from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from .data import GraphBundle
from .types import EvidenceRecord


def normalize_scores(scores: np.ndarray, fit_mask: np.ndarray | None = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    fit_scores = scores if fit_mask is None else scores[np.asarray(fit_mask, dtype=bool)]
    if fit_scores.size == 0:
        fit_scores = scores
    lo = float(np.min(fit_scores))
    hi = float(np.max(fit_scores))
    if hi - lo < 1e-8:
        return np.zeros_like(scores)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


def _row_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    degree = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv[nonzero] = 1.0 / degree[nonzero]
    return sp.diags(inv).dot(adj).tocsr()


def degree_anomaly(bundle: GraphBundle, adj: sp.csr_matrix, fit_mask: np.ndarray | None = None) -> np.ndarray:
    degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    scope = degree if fit_mask is None else degree[np.asarray(fit_mask, dtype=bool)]
    if scope.size == 0:
        scope = degree
    z = np.abs((degree - scope.mean()) / (scope.std() + 1e-6))
    return normalize_scores(z, fit_mask=fit_mask)


def relation_degree_profile(bundle: GraphBundle, _adj: sp.csr_matrix, fit_mask: np.ndarray | None = None) -> np.ndarray:
    adjs = [bundle.homo_adj] + [bundle.relation_adjs[name] for name in sorted(bundle.relation_adjs)]
    if len(adjs) == 1:
        return degree_anomaly(bundle, bundle.homo_adj, fit_mask=fit_mask)
    profiles = [normalize_scores(np.asarray(adj.sum(axis=1)).reshape(-1), fit_mask=fit_mask) for adj in adjs]
    stacked = np.stack(profiles, axis=1)
    return normalize_scores(np.std(stacked, axis=1), fit_mask=fit_mask)


def relation_disagreement(bundle: GraphBundle, _adj: sp.csr_matrix, fit_mask: np.ndarray | None = None) -> np.ndarray:
    if not bundle.relation_adjs:
        return np.zeros(bundle.num_nodes, dtype=np.float32)
    homo_degree = normalize_scores(np.asarray(bundle.homo_adj.sum(axis=1)).reshape(-1), fit_mask=fit_mask)
    relation_degree = normalize_scores(
        np.mean(
            np.stack(
                [np.asarray(adj.sum(axis=1)).reshape(-1) for adj in bundle.relation_adjs.values()],
                axis=1,
            ),
            axis=1,
        ),
        fit_mask=fit_mask,
    )
    return normalize_scores(np.abs(homo_degree - relation_degree), fit_mask=fit_mask)


def neighbor_feature_deviation(bundle: GraphBundle, adj: sp.csr_matrix, fit_mask: np.ndarray | None = None) -> np.ndarray:
    row_norm = _row_normalize(adj)
    neighbor_mean = row_norm.dot(bundle.features)
    deviation = np.linalg.norm(bundle.features - neighbor_mean, axis=1)
    return normalize_scores(deviation, fit_mask=fit_mask)


def feature_smoothness(bundle: GraphBundle, adj: sp.csr_matrix, fit_mask: np.ndarray | None = None) -> np.ndarray:
    row_norm = _row_normalize(adj)
    propagated = row_norm.dot(bundle.features)
    smoothness = np.mean(np.abs(bundle.features - propagated), axis=1)
    return normalize_scores(smoothness, fit_mask=fit_mask)


def feature_reconstruction_residual(bundle: GraphBundle, _adj: sp.csr_matrix, fit_mask: np.ndarray | None = None) -> np.ndarray:
    fit_mask = bundle.train_mask if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    train_count = int(np.sum(fit_mask))
    n_components = max(1, min(8, bundle.num_features - 1, train_count - 1))
    if train_count < 2 or bundle.num_features < 2:
        return np.zeros(bundle.num_nodes, dtype=np.float32)
    model = TruncatedSVD(n_components=n_components, random_state=0)
    model.fit(bundle.features[fit_mask])
    reduced = model.transform(bundle.features)
    reconstructed = model.inverse_transform(reduced)
    residual = np.linalg.norm(bundle.features - reconstructed, axis=1)
    return normalize_scores(residual, fit_mask=fit_mask)


TOOL_REGISTRY: dict[str, Callable[[GraphBundle, sp.csr_matrix, np.ndarray | None], np.ndarray]] = {
    "degree_anomaly": degree_anomaly,
    "relation_degree_profile": relation_degree_profile,
    "relation_disagreement": relation_disagreement,
    "neighbor_feature_deviation": neighbor_feature_deviation,
    "feature_smoothness": feature_smoothness,
    "feature_reconstruction_residual": feature_reconstruction_residual,
}


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "top1": float(np.max(scores)),
    }


def run_toolkit(
    bundle: GraphBundle,
    graph_view: str,
    tool_names: list[str],
    fit_mask: np.ndarray | None = None,
) -> list[EvidenceRecord]:
    adj = bundle.get_view(graph_view)
    fit_mask = bundle.train_mask if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    records: list[EvidenceRecord] = []
    for tool_name in tool_names:
        scores = TOOL_REGISTRY[tool_name](bundle, adj, fit_mask)
        records.append(
            EvidenceRecord(
                tool_name=tool_name,
                scores=scores,
                summary=summarize_scores(scores),
                confidence=float(np.std(scores)),
                metadata={"graph_view": graph_view},
            )
        )
    return records
