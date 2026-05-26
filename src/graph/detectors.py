from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import GraphBundle
from .tools import normalize_scores


def _adj_cache_key(prefix: str, adj: sp.csr_matrix, *parts: object) -> str:
    return "|".join(
        [
            prefix,
            str(adj.shape[0]),
            str(adj.nnz),
            f"{float(adj.sum()):.4f}",
            *[str(part) for part in parts],
        ]
    )


def _global_cache_key(prefix: str, bundle: GraphBundle, *parts: object) -> str:
    return "|".join(
        [
            prefix,
            bundle.dataset_id,
            str(bundle.num_nodes),
            str(bundle.num_features),
            *[str(part) for part in parts],
        ]
    )


def _mask_cache_token(mask: np.ndarray) -> str:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)
    digest = hashlib.sha1(indices.tobytes()).hexdigest()[:12]
    return f"{indices.size}:{digest}"


def _row_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    inv = np.zeros_like(degree)
    valid = degree > 0
    inv[valid] = 1.0 / degree[valid]
    return sp.diags(inv).dot(adj).tocsr()


def _neighbor_mean_features(bundle: GraphBundle, adj: sp.csr_matrix) -> np.ndarray:
    return _row_normalize(adj).dot(bundle.features)


def _build_graph_feature_matrix(bundle: GraphBundle, adj: sp.csr_matrix) -> np.ndarray:
    cache_key = _adj_cache_key("feature_matrix", adj)
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    features = [bundle.features]
    degree = np.asarray(adj.sum(axis=1)).reshape(-1, 1).astype(np.float32)
    features.append(np.log1p(degree))

    agg1 = _neighbor_mean_features(bundle, adj)
    agg2 = _row_normalize(adj).dot(agg1)
    features.extend([agg1, agg2, np.abs(bundle.features - agg1), np.abs(agg1 - agg2)])

    if bundle.relation_adjs:
        for relation_adj in bundle.relation_adjs.values():
            relation_degree = np.asarray(relation_adj.sum(axis=1)).reshape(-1, 1).astype(np.float32)
            features.append(np.log1p(relation_degree))

    feature_matrix = np.concatenate(features, axis=1).astype(np.float32)
    bundle.runtime_cache[cache_key] = feature_matrix
    return feature_matrix


def _graph_feature_lr_scores(
    bundle: GraphBundle,
    adj: sp.csr_matrix,
    feature_matrix: np.ndarray,
    train_mask: np.ndarray,
    c_value: float = 0.1,
) -> np.ndarray:
    cache_key = _adj_cache_key("graph_feature_lr_scores", adj, c_value, _mask_cache_token(train_mask))
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=c_value,
            solver="lbfgs",
        ),
    )
    model.fit(feature_matrix[train_mask], bundle.labels[train_mask])
    scores = model.predict_proba(feature_matrix)[:, 1]
    bundle.runtime_cache[cache_key] = scores
    return scores


def graph_feature_lr_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    feature_matrix = _build_graph_feature_matrix(bundle, adj)
    scores = _graph_feature_lr_scores(bundle, adj, feature_matrix, train_mask)
    return normalize_scores(scores, fit_mask=train_mask)


def graph_feature_lr_blend_030_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    fused_adj = bundle.get_view("relation_fused")
    fused_scores = _graph_feature_lr_scores(bundle, fused_adj, _build_graph_feature_matrix(bundle, fused_adj), train_mask)
    homo_scores = _graph_feature_lr_scores(bundle, bundle.homo_adj, _build_graph_feature_matrix(bundle, bundle.homo_adj), train_mask)
    return normalize_scores(0.30 * fused_scores + 0.70 * homo_scores, fit_mask=train_mask)


def graph_feature_et_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    cache_key = _adj_cache_key("graph_feature_et_scores", adj, _mask_cache_token(train_mask))
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    feature_matrix = _build_graph_feature_matrix(bundle, adj)
    model = ExtraTreesClassifier(
        n_estimators=800,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )
    model.fit(feature_matrix[train_mask], bundle.labels[train_mask])
    scores = normalize_scores(model.predict_proba(feature_matrix)[:, 1], fit_mask=train_mask)
    bundle.runtime_cache[cache_key] = scores
    return scores


def _graph_feature_et_blend_detector(
    bundle: GraphBundle,
    adj: sp.csr_matrix,
    train_mask: np.ndarray,
    et_weight: float,
) -> np.ndarray:
    et_scores = graph_feature_et_detector(bundle, adj, train_mask)
    lr_scores = graph_feature_lr_blend_030_detector(bundle, adj, train_mask)
    return normalize_scores(et_weight * et_scores + (1.0 - et_weight) * lr_scores, fit_mask=train_mask)


def graph_feature_et_blend_020_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    return _graph_feature_et_blend_detector(bundle, adj, train_mask, et_weight=0.20)


def graph_feature_et_blend_025_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    return _graph_feature_et_blend_detector(bundle, adj, train_mask, et_weight=0.25)


def graph_feature_et_blend_030_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    return _graph_feature_et_blend_detector(bundle, adj, train_mask, et_weight=0.30)


def _meta_stack_feature_matrix(bundle: GraphBundle, train_mask: np.ndarray) -> np.ndarray:
    mask_token = _mask_cache_token(train_mask)
    cache_key = _global_cache_key("meta_stack_feature_matrix", bundle, mask_token)
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    base_names = [
        "GraphFeatureLR",
        "GraphFeatureLRBlend030",
        "GraphFeatureETBlend020",
        "GraphFeatureETBlend025",
        "GraphFeatureETBlend030",
    ]
    columns = []
    for adj in [bundle.get_view("relation_fused"), bundle.homo_adj]:
        for detector_name in base_names:
            scores = DETECTOR_REGISTRY[detector_name](bundle, adj, train_mask)
            columns.append(scores.reshape(-1, 1))

    feature_matrix = np.concatenate(columns, axis=1).astype(np.float32)
    bundle.runtime_cache[cache_key] = feature_matrix
    return feature_matrix


def _stacked_representation(bundle: GraphBundle, train_mask: np.ndarray) -> np.ndarray:
    mask_token = _mask_cache_token(train_mask)
    cache_key = _global_cache_key("stacked_representation", bundle, mask_token)
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    fused_graph_features = _build_graph_feature_matrix(bundle, bundle.get_view("relation_fused"))
    meta_stack_features = _meta_stack_feature_matrix(bundle, train_mask)
    feature_matrix = np.concatenate([fused_graph_features, meta_stack_features], axis=1).astype(np.float32)
    bundle.runtime_cache[cache_key] = feature_matrix
    return feature_matrix


def graph_feature_stack_et_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    mask_token = _mask_cache_token(train_mask)
    cache_key = _global_cache_key("graph_feature_stack_et_scores", bundle, mask_token)
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    feature_matrix = _stacked_representation(bundle, train_mask)
    model = ExtraTreesClassifier(
        n_estimators=1200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )
    model.fit(feature_matrix[train_mask], bundle.labels[train_mask])
    scores = normalize_scores(model.predict_proba(feature_matrix)[:, 1], fit_mask=train_mask)
    bundle.runtime_cache[cache_key] = scores
    return scores


def graph_feature_stack_mlp_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    mask_token = _mask_cache_token(train_mask)
    cache_key = _global_cache_key("graph_feature_stack_mlp_scores", bundle, mask_token)
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    feature_matrix = _stacked_representation(bundle, train_mask)
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=42,
        ),
    )
    model.fit(feature_matrix[train_mask], bundle.labels[train_mask])
    scores = normalize_scores(model.predict_proba(feature_matrix)[:, 1], fit_mask=train_mask)
    bundle.runtime_cache[cache_key] = scores
    return scores


def graph_feature_stack_hybrid_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    et_scores = graph_feature_stack_et_detector(bundle, adj, train_mask)
    mlp_scores = graph_feature_stack_mlp_detector(bundle, adj, train_mask)
    return normalize_scores(0.175 * et_scores + 0.825 * mlp_scores, fit_mask=train_mask)


def _tool_feature_matrix(bundle: GraphBundle, train_mask: np.ndarray) -> np.ndarray:
    mask_token = _mask_cache_token(train_mask)
    cache_key = _global_cache_key("tool_feature_matrix", bundle, mask_token)
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    from .tools import TOOL_REGISTRY

    tool_names = [
        "degree_anomaly",
        "relation_disagreement",
        "neighbor_feature_deviation",
        "feature_reconstruction_residual",
        "relation_degree_profile",
        "feature_smoothness",
    ]
    columns = []
    for adj in [bundle.get_view("relation_fused"), bundle.homo_adj]:
        for tool_name in tool_names:
            columns.append(TOOL_REGISTRY[tool_name](bundle, adj, train_mask).reshape(-1, 1))
    feature_matrix = np.concatenate(columns, axis=1).astype(np.float32)
    bundle.runtime_cache[cache_key] = feature_matrix
    return feature_matrix


def _stacked_tool_aware_representation(bundle: GraphBundle, train_mask: np.ndarray) -> np.ndarray:
    mask_token = _mask_cache_token(train_mask)
    cache_key = _global_cache_key("stacked_tool_aware_representation", bundle, mask_token)
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    feature_matrix = np.concatenate(
        [
            _stacked_representation(bundle, train_mask),
            _tool_feature_matrix(bundle, train_mask),
        ],
        axis=1,
    ).astype(np.float32)
    bundle.runtime_cache[cache_key] = feature_matrix
    return feature_matrix


def graph_feature_stack_tool_aware_et_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    mask_token = _mask_cache_token(train_mask)
    cache_key = _global_cache_key("graph_feature_stack_tool_aware_et_scores", bundle, mask_token)
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    feature_matrix = _stacked_tool_aware_representation(bundle, train_mask)
    model = ExtraTreesClassifier(
        n_estimators=1200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )
    model.fit(feature_matrix[train_mask], bundle.labels[train_mask])
    scores = normalize_scores(model.predict_proba(feature_matrix)[:, 1], fit_mask=train_mask)
    bundle.runtime_cache[cache_key] = scores
    return scores


def graph_feature_stack_tool_aware_mlp_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    mask_token = _mask_cache_token(train_mask)
    cache_key = _global_cache_key("graph_feature_stack_tool_aware_mlp_scores", bundle, mask_token)
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    feature_matrix = _stacked_tool_aware_representation(bundle, train_mask)
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(160, 80),
            activation="relu",
            alpha=5e-4,
            learning_rate_init=8e-4,
            max_iter=500,
            random_state=42,
        ),
    )
    model.fit(feature_matrix[train_mask], bundle.labels[train_mask])
    scores = normalize_scores(model.predict_proba(feature_matrix)[:, 1], fit_mask=train_mask)
    bundle.runtime_cache[cache_key] = scores
    return scores


def graph_feature_stack_tool_aware_hybrid_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    et_scores = graph_feature_stack_tool_aware_et_detector(bundle, adj, train_mask)
    mlp_scores = graph_feature_stack_tool_aware_mlp_detector(bundle, adj, train_mask)
    return normalize_scores(0.2 * et_scores + 0.8 * mlp_scores, fit_mask=train_mask)


def _relation_bank_representation(bundle: GraphBundle, train_mask: np.ndarray) -> np.ndarray:
    mask_token = _mask_cache_token(train_mask)
    cache_key = _global_cache_key("relation_bank_representation", bundle, mask_token)
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    if not bundle.relation_adjs:
        return _stacked_tool_aware_representation(bundle, train_mask)

    from .tools import TOOL_REGISTRY

    columns = [_stacked_tool_aware_representation(bundle, train_mask)]
    for relation_name in sorted(bundle.relation_adjs):
        relation_adj = bundle.relation_adjs[relation_name]
        relation_features = _build_graph_feature_matrix(bundle, relation_adj)
        train_count = int(np.sum(train_mask))
        n_components = max(1, min(12, relation_features.shape[1] - 1, train_count - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(relation_features[train_mask])
        reduced = svd.transform(relation_features)
        columns.append(reduced.astype(np.float32))
        columns.append(graph_feature_lr_detector(bundle, relation_adj, train_mask).reshape(-1, 1))
        columns.append(graph_feature_et_detector(bundle, relation_adj, train_mask).reshape(-1, 1))
        columns.append(TOOL_REGISTRY["neighbor_feature_deviation"](bundle, relation_adj, train_mask).reshape(-1, 1))
        columns.append(TOOL_REGISTRY["degree_anomaly"](bundle, relation_adj, train_mask).reshape(-1, 1))
        columns.append(TOOL_REGISTRY["feature_smoothness"](bundle, relation_adj, train_mask).reshape(-1, 1))

    feature_matrix = np.concatenate(columns, axis=1).astype(np.float32)
    bundle.runtime_cache[cache_key] = feature_matrix
    return feature_matrix


def graph_feature_relation_bank_hybrid_detector(bundle: GraphBundle, adj: sp.csr_matrix, train_mask: np.ndarray) -> np.ndarray:
    mask_token = _mask_cache_token(train_mask)
    cache_key = _global_cache_key("graph_feature_relation_bank_hybrid_scores", bundle, mask_token)
    cached = bundle.runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    feature_matrix = _relation_bank_representation(bundle, train_mask)
    et = ExtraTreesClassifier(
        n_estimators=1200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(160, 80),
            activation="relu",
            alpha=5e-4,
            learning_rate_init=8e-4,
            max_iter=500,
            random_state=42,
        ),
    )
    et.fit(feature_matrix[train_mask], bundle.labels[train_mask])
    mlp.fit(feature_matrix[train_mask], bundle.labels[train_mask])
    et_scores = et.predict_proba(feature_matrix)[:, 1]
    mlp_scores = mlp.predict_proba(feature_matrix)[:, 1]
    base_scores = graph_feature_stack_tool_aware_hybrid_detector(bundle, adj, train_mask)
    scores = normalize_scores(0.20 * et_scores + 0.45 * mlp_scores + 0.35 * base_scores, fit_mask=train_mask)
    bundle.runtime_cache[cache_key] = scores
    return scores


DETECTOR_REGISTRY: dict[str, Callable[[GraphBundle, sp.csr_matrix, np.ndarray], np.ndarray]] = {
    "GraphFeatureLR": graph_feature_lr_detector,
    "GraphFeatureLRBlend030": graph_feature_lr_blend_030_detector,
    "GraphFeatureET": graph_feature_et_detector,
    "GraphFeatureETBlend020": graph_feature_et_blend_020_detector,
    "GraphFeatureETBlend025": graph_feature_et_blend_025_detector,
    "GraphFeatureETBlend030": graph_feature_et_blend_030_detector,
    "GraphFeatureStackET": graph_feature_stack_et_detector,
    "GraphFeatureStackMLP": graph_feature_stack_mlp_detector,
    "GraphFeatureStackHybrid": graph_feature_stack_hybrid_detector,
    "GraphFeatureStackToolAwareET": graph_feature_stack_tool_aware_et_detector,
    "GraphFeatureStackToolAwareMLP": graph_feature_stack_tool_aware_mlp_detector,
    "GraphFeatureStackToolAwareHybrid": graph_feature_stack_tool_aware_hybrid_detector,
    "GraphFeatureRelationBankHybrid": graph_feature_relation_bank_hybrid_detector,
}
