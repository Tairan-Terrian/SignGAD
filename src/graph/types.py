from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class WorkflowSpec:
    detector_name: str
    graph_view: str
    tool_names: list[str]
    fusion_strategy: str = "weighted_sum"
    detector_weight: float = 1.0
    tool_weights: dict[str, float] = field(default_factory=dict)
    threshold_strategy: str = "validation_search"
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRecord:
    tool_name: str
    scores: np.ndarray
    summary: dict[str, Any]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = self.scores.tolist()
        return payload


@dataclass
class OptimizationTrace:
    iteration: int
    workflow_spec: WorkflowSpec
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    selected_threshold: float
    detector_summary: dict[str, Any]
    evidence_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["workflow_spec"] = self.workflow_spec.to_dict()
        return payload


@dataclass
class ExperimentResult:
    task_card: Any
    best_workflow: WorkflowSpec
    optimization_trace: list[OptimizationTrace]
    best_validation_metrics: dict[str, float]
    final_test_metrics: dict[str, float]
    selected_threshold: float
    detector_summary: dict[str, Any]
    evidence_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_card": self.task_card.to_dict(),
            "best_workflow": self.best_workflow.to_dict(),
            "optimization_trace": [item.to_dict() for item in self.optimization_trace],
            "best_validation_metrics": self.best_validation_metrics,
            "final_test_metrics": self.final_test_metrics,
            "selected_threshold": self.selected_threshold,
            "detector_summary": self.detector_summary,
            "evidence_summary": self.evidence_summary,
        }
