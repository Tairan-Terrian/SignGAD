from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from .dataset_downloads import maybe_download_known_graph_dataset


@dataclass
class GraphBundle:
    dataset_id: str
    dataset_path: str
    features: np.ndarray
    labels: np.ndarray
    homo_adj: sp.csr_matrix
    relation_adjs: dict[str, sp.csr_matrix]
    split_config: dict[str, Any]
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    runtime_cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def anomaly_ratio(self) -> float:
        return float(np.mean(self.labels))

    @property
    def graph_views(self) -> list[str]:
        views = ["homo", "relation_fused"]
        views.extend(sorted(self.relation_adjs))
        return views

    def get_view(self, graph_view: str) -> sp.csr_matrix:
        if graph_view == "homo":
            return self.homo_adj
        if graph_view == "relation_fused":
            if not self.relation_adjs:
                return self.homo_adj
            fused = self.homo_adj.copy().astype(np.float32)
            for adj in self.relation_adjs.values():
                fused = fused + adj.astype(np.float32)
            fused.data[:] = 1.0
            return fused.tocsr()
        if graph_view in self.relation_adjs:
            return self.relation_adjs[graph_view]
        raise KeyError(f"Unknown graph view: {graph_view}")

    def split_masks(self) -> dict[str, np.ndarray]:
        return {
            "train": self.train_mask.copy(),
            "val": self.val_mask.copy(),
            "test": self.test_mask.copy(),
        }


@dataclass
class GraphTaskCard:
    dataset_id: str
    dataset_path: str
    domain: str
    node_semantics: str
    relation_semantics: dict[str, str]
    anomaly_objective: str
    available_graph_views: list[str]
    feature_stats: dict[str, Any]
    label_stats: dict[str, Any]
    split_config: dict[str, Any]
    optimization_target: str
    evidence_budget: int
    tool_budget: int
    raw_user_description: str
    structured_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SPLIT_CONFIG = {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "seed": 42}

DATASET_DEFAULTS = {
    "amazon": {
        "domain": "E-commerce fraud detection",
        "node_semantics": "Each node represents a user-review entity in the Amazon review graph.",
        "relation_semantics": {
            "homo": "Homogeneous similarity graph used as the main transaction proximity view.",
            "net_upu": "Users connected through shared product interactions.",
            "net_usu": "Users connected through shared star-rating patterns.",
            "net_uvu": "Users connected through shared review text or voting behavior.",
        },
        "anomaly_objective": "Detect fraudulent or suspicious review-centric nodes.",
    },
    "yelpchi": {
        "domain": "Review spam detection",
        "node_semantics": "Each node represents a reviewer-centric entity in the Yelp review graph.",
        "relation_semantics": {
            "homo": "Homogeneous similarity graph used as the primary reviewer affinity view.",
            "net_rsr": "Reviewers linked by shared star-rating behavior.",
            "net_rtr": "Reviewers linked by temporal review patterns.",
            "net_rur": "Reviewers linked by common users or review neighborhoods.",
        },
        "anomaly_objective": "Detect spam or anomalous reviewer nodes.",
    },
    "tfinance": {
        "domain": "Financial fraud detection",
        "node_semantics": "Each node represents an account in a transaction network.",
        "relation_semantics": {
            "homo": "Accounts connected by transaction records.",
        },
        "anomaly_objective": "Detect anomalous or fraudulent accounts in the transaction graph.",
    },
    "tsocial": {
        "domain": "Social network anomaly detection",
        "node_semantics": "Each node represents a user in a social network.",
        "relation_semantics": {
            "homo": "Users connected by long-standing social interactions.",
        },
        "anomaly_objective": "Detect anomalous users in the social interaction graph.",
    },
}


def _parse_description(user_description: str, defaults: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    relation_semantics = dict(defaults.get("relation_semantics", {}))
    for raw_line in user_description.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        key_lower = key.lower()
        if key_lower in {"domain", "dataset domain"}:
            parsed["domain"] = value
        elif key_lower in {"node", "node semantics", "nodes"}:
            parsed["node_semantics"] = value
        elif key_lower in {"anomaly", "objective", "anomaly objective"}:
            parsed["anomaly_objective"] = value
        elif key_lower in {"evidence budget", "evidence_budget"}:
            parsed["evidence_budget"] = int(value)
        elif key_lower in {"tool budget", "tool_budget"}:
            parsed["tool_budget"] = int(value)
        elif key_lower.startswith("relation "):
            relation_semantics[key.split(" ", 1)[1].strip()] = value
    if relation_semantics:
        parsed["relation_semantics"] = relation_semantics
    return parsed


def _ensure_csr(matrix: Any) -> sp.csr_matrix:
    if sp.issparse(matrix):
        csr = matrix.tocsr().astype(np.float32)
    else:
        csr = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    if csr.shape[0] != csr.shape[1]:
        raise ValueError(f"Adjacency must be square, got {csr.shape}")
    return csr


def _load_labels(raw: Any) -> np.ndarray:
    labels = _to_numpy(raw)
    if labels.ndim == 2 and labels.shape[0] > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    else:
        labels = labels.reshape(-1)
    labels = (labels > 0).astype(np.int64)
    return labels


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _create_masks(labels: np.ndarray, split_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(labels.shape[0])
    train_ratio = float(split_config["train_ratio"])
    val_ratio = float(split_config["val_ratio"])
    seed = int(split_config["seed"])

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )
    val_share = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_share,
        stratify=y_temp,
        random_state=seed,
    )

    train_mask = np.zeros(labels.shape[0], dtype=bool)
    val_mask = np.zeros(labels.shape[0], dtype=bool)
    test_mask = np.zeros(labels.shape[0], dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _load_existing_masks(ndata: dict[str, Any], num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    required = ("train_mask", "val_mask", "test_mask")
    if not all(key in ndata for key in required):
        return None
    masks = tuple(_to_numpy(ndata[key]).reshape(-1).astype(bool) for key in required)
    if any(mask.shape[0] != num_nodes for mask in masks):
        raise ValueError("DGL train/val/test masks must match the number of nodes.")
    return masks


def _get_dgl_num_nodes(graph: Any) -> int:
    num_nodes = getattr(graph, "num_nodes", None)
    if callable(num_nodes):
        return int(num_nodes())
    if num_nodes is not None:
        return int(num_nodes)
    number_of_nodes = getattr(graph, "number_of_nodes", None)
    if callable(number_of_nodes):
        return int(number_of_nodes())
    raise ValueError("DGL graph does not expose num_nodes/number_of_nodes.")


def _dgl_graph_to_csr(graph: Any, num_nodes: int) -> sp.csr_matrix:
    try:
        src, dst = graph.edges(order="eid")
    except TypeError:
        try:
            src, dst = graph.edges()
        except Exception as exc:
            raise ValueError("Only homogeneous DGL graphs with a default edge type are supported.") from exc
    src_np = _to_numpy(src).reshape(-1).astype(np.int64)
    dst_np = _to_numpy(dst).reshape(-1).astype(np.int64)
    data = np.ones(src_np.shape[0], dtype=np.float32)
    return sp.csr_matrix((data, (src_np, dst_np)), shape=(num_nodes, num_nodes))


def _load_mat_graph_bundle(path: Path, split_config: dict[str, Any]) -> GraphBundle:
    raw = scipy.io.loadmat(path)
    if "features" not in raw or "label" not in raw or "homo" not in raw:
        raise ValueError("Expected keys 'features', 'label', and 'homo' in graph dataset.")

    features = raw["features"]
    if sp.issparse(features):
        features = features.toarray()
    features = np.asarray(features, dtype=np.float32)
    labels = _load_labels(raw["label"])
    homo_adj = _ensure_csr(raw["homo"])

    relation_adjs: dict[str, sp.csr_matrix] = {}
    for key, value in raw.items():
        if key.startswith("__") or key in {"features", "label", "homo"}:
            continue
        if sp.issparse(value) or isinstance(value, np.ndarray):
            relation_adjs[key] = _ensure_csr(value)

    train_mask, val_mask, test_mask = _create_masks(labels, split_config)
    return GraphBundle(
        dataset_id=path.stem,
        dataset_path=str(path),
        features=features,
        labels=labels,
        homo_adj=homo_adj,
        relation_adjs=relation_adjs,
        split_config=split_config,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def _load_dgl_graph_bundle(path: Path, split_config: dict[str, Any]) -> GraphBundle:
    try:
        from dgl.data.utils import load_graphs
    except ImportError as exc:
        raise ImportError(
            "DGL is required to load non-.mat graph datasets such as T-Finance/T-Social. "
            "Install dgl in the active environment and retry."
        ) from exc

    graphs, _label_dict = load_graphs(str(path))
    if not graphs:
        raise ValueError(f"No graphs found in DGL dataset: {path}")

    graph = graphs[0]
    ndata = getattr(graph, "ndata", {})
    feature_key = "feature" if "feature" in ndata else "features"
    if feature_key not in ndata:
        raise ValueError("Expected DGL node data key 'feature' or 'features'.")
    if "label" not in ndata:
        raise ValueError("Expected DGL node data key 'label'.")

    features = _to_numpy(ndata[feature_key]).astype(np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    labels = _load_labels(ndata["label"])
    num_nodes = _get_dgl_num_nodes(graph)
    if features.shape[0] != num_nodes or labels.shape[0] != num_nodes:
        raise ValueError("DGL feature/label arrays must match the number of nodes.")

    existing_masks = _load_existing_masks(ndata, num_nodes)
    if existing_masks is None:
        train_mask, val_mask, test_mask = _create_masks(labels, split_config)
    else:
        train_mask, val_mask, test_mask = existing_masks

    return GraphBundle(
        dataset_id=path.stem,
        dataset_path=str(path),
        features=features,
        labels=labels,
        homo_adj=_dgl_graph_to_csr(graph, num_nodes),
        relation_adjs={},
        split_config=split_config,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def build_graph_bundle(dataset_path: str, config: dict[str, Any] | None = None) -> GraphBundle:
    path = maybe_download_known_graph_dataset(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    split_config = dict(DEFAULT_SPLIT_CONFIG)
    if config:
        split_config.update(config.get("split_config", {}))

    if path.suffix.lower() == ".mat":
        return _load_mat_graph_bundle(path, split_config)
    if path.suffix.lower() == ".zip":
        raise ValueError("Please unzip DGL graph archives first and pass the extracted graph file path.")
    return _load_dgl_graph_bundle(path, split_config)


def build_graph_task_card(dataset_path: str, user_description: str, config: dict[str, Any] | None = None) -> GraphTaskCard:
    graph_bundle = build_graph_bundle(dataset_path, config=config)
    dataset_defaults = DATASET_DEFAULTS.get(graph_bundle.dataset_id.lower(), {})
    parsed = _parse_description(user_description, dataset_defaults)

    degree = np.asarray(graph_bundle.homo_adj.sum(axis=1)).reshape(-1)
    feature_stats = {
        "num_nodes": graph_bundle.num_nodes,
        "num_features": graph_bundle.num_features,
        "feature_mean": float(np.mean(graph_bundle.features)),
        "feature_std": float(np.std(graph_bundle.features)),
        "degree_mean": float(np.mean(degree)),
        "degree_std": float(np.std(degree)),
        "num_relations": len(graph_bundle.relation_adjs),
    }
    train_labels = graph_bundle.labels[graph_bundle.train_mask]
    label_stats = {
        "scope": "train",
        "num_train_anomalies": int(np.sum(train_labels)),
        "num_train_normals": int(train_labels.shape[0] - np.sum(train_labels)),
        "train_anomaly_ratio": float(np.mean(train_labels)) if train_labels.size else 0.0,
    }

    domain = parsed.get("domain", dataset_defaults.get("domain", "Graph anomaly detection"))
    node_semantics = parsed.get(
        "node_semantics",
        dataset_defaults.get("node_semantics", "Each node represents an entity in a fraud graph."),
    )
    relation_semantics = parsed.get("relation_semantics", dataset_defaults.get("relation_semantics", {"homo": "Primary homogeneous graph"}))
    anomaly_objective = parsed.get(
        "anomaly_objective",
        dataset_defaults.get("anomaly_objective", "Detect anomalous nodes in the graph."),
    )
    evidence_budget = int(parsed.get("evidence_budget", (config or {}).get("evidence_budget", 4)))
    tool_budget = int(parsed.get("tool_budget", (config or {}).get("tool_budget", 3)))

    return GraphTaskCard(
        dataset_id=graph_bundle.dataset_id,
        dataset_path=graph_bundle.dataset_path,
        domain=domain,
        node_semantics=node_semantics,
        relation_semantics=relation_semantics,
        anomaly_objective=anomaly_objective,
        available_graph_views=graph_bundle.graph_views,
        feature_stats=feature_stats,
        label_stats=label_stats,
        split_config=graph_bundle.split_config,
        optimization_target="AUC + F1-macro",
        evidence_budget=evidence_budget,
        tool_budget=tool_budget,
        raw_user_description=user_description,
        structured_context={
            "graph_summary": {
                "num_nodes": graph_bundle.num_nodes,
                "num_features": graph_bundle.num_features,
                "graph_views": graph_bundle.graph_views,
            },
            "description_overrides": parsed,
        },
    )
