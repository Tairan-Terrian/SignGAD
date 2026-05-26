from __future__ import annotations

from typing import Any

from graph.pipeline import (
    build_graph_task_card,
    design_workflow,
    evaluate_and_optimize,
    run_agentgad_graph,
    run_toolkit,
)


def build_state() -> dict[str, Any]:
    """Compatibility helper for callers expecting a state-like container."""

    return {
        "dataset_path": "",
        "user_description": "",
        "config": {},
        "task_card": None,
        "workflow_spec": None,
        "result": None,
    }


__all__ = [
    "build_graph_task_card",
    "design_workflow",
    "evaluate_and_optimize",
    "run_agentgad_graph",
    "run_toolkit",
    "build_state",
]
