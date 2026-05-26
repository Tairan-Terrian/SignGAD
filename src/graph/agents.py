from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np

from .data import GraphBundle, GraphTaskCard, build_graph_bundle, build_graph_task_card
from .detectors import DETECTOR_REGISTRY
from .tools import run_toolkit
from .types import WorkflowSpec


def _llm_enabled(config: dict | None) -> bool:
    if not config:
        return False
    return bool(config.get("llm", {}).get("enabled", False))


def _extract_json_object(text: str) -> dict:
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _query_llm_json(messages, fallback: dict | None = None, model: str | None = None) -> dict:
    try:
        from utils.openai_client import query_openai

        response = query_openai(messages, model=model)
    except Exception:
        return fallback or {}
    payload = _extract_json_object(response)
    return payload or (fallback or {})


class TaskCardAgent:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def build(self, dataset_path: str, user_description: str, config: dict | None = None) -> tuple[GraphTaskCard, GraphBundle]:
        effective_config = config or self.config
        task_card = build_graph_task_card(dataset_path, user_description, config=effective_config)
        graph_bundle = build_graph_bundle(dataset_path, config=effective_config)
        if _llm_enabled(effective_config):
            task_card = self._enrich_with_llm(task_card, graph_bundle, effective_config)
        return task_card, graph_bundle

    def validate(self, task_card: GraphTaskCard) -> GraphTaskCard:
        if not task_card.available_graph_views:
            raise ValueError("Task card must expose at least one graph view.")
        if task_card.evidence_budget < 1 or task_card.tool_budget < 1:
            raise ValueError("Budgets must be positive integers.")
        return task_card

    def _enrich_with_llm(self, task_card: GraphTaskCard, graph_bundle: GraphBundle, config: dict) -> GraphTaskCard:
        fallback = {
            "domain": task_card.domain,
            "node_semantics": task_card.node_semantics,
            "anomaly_objective": task_card.anomaly_objective,
            "relation_semantics": task_card.relation_semantics,
            "preferred_graph_views": task_card.available_graph_views[:2],
            "suspected_patterns": [],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Graph-Text Information Structuring Agent for node-level graph anomaly detection. "
                    "Convert the user description and graph metadata into a concise JSON task card patch. "
                    "Return JSON only with keys: domain, node_semantics, anomaly_objective, relation_semantics, "
                    "preferred_graph_views, suspected_patterns."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "dataset_id": task_card.dataset_id,
                        "dataset_path": task_card.dataset_path,
                        "user_description": task_card.raw_user_description,
                        "feature_stats": task_card.feature_stats,
                        "train_label_stats": task_card.label_stats,
                        "available_graph_views": task_card.available_graph_views,
                        "relation_names": sorted(graph_bundle.relation_adjs.keys()),
                        "current_task_card": fallback,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        payload = _query_llm_json(messages, fallback=fallback, model=config.get("llm", {}).get("model"))
        relation_semantics = dict(task_card.relation_semantics)
        relation_semantics.update(payload.get("relation_semantics", {}))
        structured_context = dict(task_card.structured_context)
        structured_context["llm_task_card_patch"] = payload
        structured_context["preferred_graph_views"] = payload.get("preferred_graph_views", task_card.available_graph_views[:2])
        structured_context["suspected_patterns"] = payload.get("suspected_patterns", [])
        structured_context["llm_enabled"] = True
        return replace(
            task_card,
            domain=payload.get("domain", task_card.domain),
            node_semantics=payload.get("node_semantics", task_card.node_semantics),
            anomaly_objective=payload.get("anomaly_objective", task_card.anomaly_objective),
            relation_semantics=relation_semantics,
            structured_context=structured_context,
        )


class DesignAgent:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def design_candidates(self, task_card: GraphTaskCard) -> list[WorkflowSpec]:
        description = task_card.raw_user_description.lower()
        prefer_homo = "homogeneous" in description or "homo" in description
        prefer_multi = "multi-relation" in description or "multi relation" in description or "relation" in description

        if prefer_homo:
            candidate_views = ["homo"]
        elif prefer_multi and "relation_fused" in task_card.available_graph_views:
            candidate_views = ["relation_fused", "homo"]
        else:
            candidate_views = [view for view in ["relation_fused", "homo"] if view in task_card.available_graph_views]
            if not candidate_views:
                candidate_views = [task_card.available_graph_views[0]]

        candidate_detectors = [
            "GraphFeatureRelationBankHybrid",
            "GraphFeatureStackToolAwareHybrid",
            "GraphFeatureStackHybrid",
            "GraphFeatureETBlend025",
            "GraphFeatureLRBlend030",
        ]

        candidates: list[WorkflowSpec] = []
        for view in candidate_views:
            for detector_name in candidate_detectors:
                candidates.append(
                    WorkflowSpec(
                        detector_name=detector_name,
                        graph_view=view,
                        tool_names=[],
                        notes={
                            "origin": "design_agent",
                            **(
                                {
                                    "internal_tools": [
                                        "degree_anomaly",
                                        "relation_disagreement",
                                        "neighbor_feature_deviation",
                                        "feature_reconstruction_residual",
                                        "relation_degree_profile",
                                        "feature_smoothness",
                                    ]
                                }
                                if "ToolAware" in detector_name or detector_name == "GraphFeatureRelationBankHybrid"
                                else {}
                            ),
                        },
                    )
                )
        if _llm_enabled(self.config):
            candidates = self._rank_candidates_with_llm(task_card, candidates)
        return candidates

    def _rank_candidates_with_llm(self, task_card: GraphTaskCard, candidates: list[WorkflowSpec]) -> list[WorkflowSpec]:
        candidate_payload = [
            {
                "detector_name": item.detector_name,
                "graph_view": item.graph_view,
            }
            for item in candidates
        ]
        preferred_views = task_card.structured_context.get("preferred_graph_views", task_card.available_graph_views[:2])
        fallback = {
            "ordered_candidates": candidate_payload[: min(10, len(candidate_payload))],
            "detector_only_detectors": [
                item.detector_name for item in candidates if item.detector_name.startswith("GraphFeature")
            ][:8],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Self-Designing Multi-Agent planner for graph anomaly detection. "
                    "Rank workflow candidates for a graph anomaly task. "
                    "Prefer candidates likely to balance AUC and F1-macro under small-label supervision. "
                    "Return JSON only with keys ordered_candidates and detector_only_detectors."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_card": {
                            "dataset_id": task_card.dataset_id,
                            "domain": task_card.domain,
                            "node_semantics": task_card.node_semantics,
                            "anomaly_objective": task_card.anomaly_objective,
                            "feature_stats": task_card.feature_stats,
                            "train_label_stats": task_card.label_stats,
                            "preferred_graph_views": preferred_views,
                            "suspected_patterns": task_card.structured_context.get("suspected_patterns", []),
                        },
                        "candidate_pool": candidate_payload,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        payload = _query_llm_json(messages, fallback=fallback, model=self.config.get("llm", {}).get("model"))
        rank_lookup = {
            (item["detector_name"], item["graph_view"]): idx
            for idx, item in enumerate(payload.get("ordered_candidates", []))
            if "detector_name" in item and "graph_view" in item
        }
        detector_only_set = set(payload.get("detector_only_detectors", []))
        reranked = sorted(
            candidates,
            key=lambda item: (
                rank_lookup.get((item.detector_name, item.graph_view), 10_000),
                candidates.index(item),
            ),
        )
        updated = []
        for candidate in reranked:
            notes = dict(candidate.notes)
            if candidate.detector_name in detector_only_set:
                notes["tool_policy"] = "detector_only"
            notes["llm_ranked"] = True
            updated.append(replace(candidate, notes=notes))
        return updated


class DetectorAgent:
    def score(self, graph_bundle: GraphBundle, workflow_spec: WorkflowSpec) -> np.ndarray:
        adj = graph_bundle.get_view(workflow_spec.graph_view)
        detector_fn = DETECTOR_REGISTRY[workflow_spec.detector_name]
        return detector_fn(graph_bundle, adj, graph_bundle.train_mask)


class EvidenceAgent:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def select_tools(self, task_card: GraphTaskCard, workflow_spec: WorkflowSpec) -> WorkflowSpec:
        if workflow_spec.notes.get("tool_policy") == "detector_only":
            return replace(workflow_spec, tool_names=[])
        tool_names = ["degree_anomaly", "neighbor_feature_deviation", "feature_reconstruction_residual"]
        if task_card.feature_stats["num_relations"] > 0:
            tool_names.insert(1, "relation_disagreement")
            tool_names.append("relation_degree_profile")
        tool_names = tool_names[: task_card.tool_budget]
        candidate = replace(workflow_spec, tool_names=tool_names)
        if _llm_enabled(self.config):
            candidate = self._select_tools_with_llm(task_card, candidate)
        return candidate

    def collect(self, graph_bundle: GraphBundle, workflow_spec: WorkflowSpec):
        return run_toolkit(graph_bundle, workflow_spec.graph_view, workflow_spec.tool_names)

    def _select_tools_with_llm(self, task_card: GraphTaskCard, workflow_spec: WorkflowSpec) -> WorkflowSpec:
        allowed_tools = [
            "degree_anomaly",
            "relation_disagreement",
            "neighbor_feature_deviation",
            "feature_reconstruction_residual",
            "relation_degree_profile",
            "feature_smoothness",
        ]
        fallback = {
            "tool_names": workflow_spec.tool_names,
            "fusion_strategy": workflow_spec.fusion_strategy,
            "rationale": "fallback",
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Tool-Augmented Design Agent for graph anomaly detection. "
                    "Choose a small set of evidence tools for the given detector and graph view. "
                    "Return JSON only with keys tool_names, fusion_strategy, rationale."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_card": {
                            "dataset_id": task_card.dataset_id,
                            "domain": task_card.domain,
                            "anomaly_objective": task_card.anomaly_objective,
                            "feature_stats": task_card.feature_stats,
                            "suspected_patterns": task_card.structured_context.get("suspected_patterns", []),
                        },
                        "workflow_spec": workflow_spec.to_dict(),
                        "allowed_tools": allowed_tools,
                        "tool_budget": task_card.tool_budget,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        payload = _query_llm_json(messages, fallback=fallback, model=self.config.get("llm", {}).get("model"))
        selected_tools = [name for name in payload.get("tool_names", []) if name in allowed_tools][: task_card.tool_budget]
        if not selected_tools:
            selected_tools = workflow_spec.tool_names
        notes = dict(workflow_spec.notes)
        notes["llm_tool_rationale"] = payload.get("rationale", "")
        return replace(
            workflow_spec,
            tool_names=selected_tools,
            fusion_strategy=payload.get("fusion_strategy", workflow_spec.fusion_strategy),
            notes=notes,
        )


class JudgeAgent:
    @staticmethod
    def aggregate_evidence(records) -> tuple[np.ndarray, dict[str, float]]:
        if not records:
            return np.zeros(0, dtype=np.float32), {}
        stacked = np.stack([record.scores for record in records], axis=1)
        weights = {record.tool_name: 1.0 / len(records) for record in records}
        return np.mean(stacked, axis=1), weights


class OptimizerAgent:
    @staticmethod
    def _complexity(trace) -> int:
        detector_name = trace.workflow_spec.detector_name
        complexity = 0
        if "ET" in detector_name:
            complexity += 2
        if "Blend" in detector_name:
            complexity += 1
        if "CrossView" in detector_name:
            complexity += 2
        if "Prop" in detector_name:
            complexity += 2
        complexity += len(trace.workflow_spec.tool_names)
        return complexity

    def choose_best(self, traces) -> int:
        best_index = 0
        best_key = (-1.0, -1.0, 0, -1.0, -1.0)
        for index, trace in enumerate(traces):
            auc = trace.validation_metrics["auc"]
            f1 = trace.validation_metrics["f1_macro"]
            balance = 0.6 * f1 + 0.4 * auc
            key = (
                round(min(auc, f1), 3),
                round(balance, 3),
                -self._complexity(trace),
                auc,
                f1,
            )
            if key > best_key:
                best_key = key
                best_index = index
        return best_index
