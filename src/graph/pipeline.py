from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .agents import DesignAgent, DetectorAgent, EvidenceAgent, JudgeAgent, OptimizerAgent, TaskCardAgent
from .types import ExperimentResult, OptimizationTrace, WorkflowSpec
def _log_stage(stage: str, message: str) -> None:
    print(f"[AgentGAD][{stage}] {message}")


def _format_names(names: list[str] | tuple[str, ...]) -> str:
    return ", ".join(names) if names else "none"


def _count_mask(mask: np.ndarray) -> int:
    return int(np.sum(np.asarray(mask, dtype=bool)))


def _normalize(scores: np.ndarray, fit_mask: np.ndarray | None = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    fit_scores = scores if fit_mask is None else scores[np.asarray(fit_mask, dtype=bool)]
    if fit_scores.size == 0:
        fit_scores = scores
    lo = float(np.min(fit_scores))
    hi = float(np.max(fit_scores))
    if hi - lo < 1e-8:
        return np.zeros_like(scores)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


def _search_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = labels.astype(int)
    candidates = np.unique(scores)
    if candidates.size > 4096:
        indices = np.linspace(0, candidates.size - 1, 4096).astype(int)
        candidates = candidates[indices]
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        f1 = float(f1_score(labels, pred, average="macro"))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(int)
    pred = (scores >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "f1_macro": float(f1_score(labels, pred, average="macro")),
    }


def _score_workflow(graph_bundle, workflow_spec: WorkflowSpec) -> tuple[np.ndarray, np.ndarray, dict[str, float], WorkflowSpec]:
    detector_agent = DetectorAgent()
    evidence_agent = EvidenceAgent()
    judge_agent = JudgeAgent()

    detector_scores = detector_agent.score(graph_bundle, workflow_spec)
    evidence_records = evidence_agent.collect(graph_bundle, workflow_spec)
    aggregated_evidence, tool_weights = judge_agent.aggregate_evidence(evidence_records)
    return detector_scores, aggregated_evidence, tool_weights, replace(workflow_spec, tool_weights=tool_weights)


def _fuse_scores(
    detector_scores: np.ndarray,
    evidence_scores: np.ndarray,
    labels: np.ndarray,
    val_mask: np.ndarray,
    workflow_spec: WorkflowSpec,
) -> tuple[np.ndarray, float, dict[str, float], WorkflowSpec]:
    detector_norm = _normalize(detector_scores, fit_mask=val_mask)
    if evidence_scores.size == 0:
        threshold, _ = _search_threshold(labels[val_mask], detector_norm[val_mask])
        val_metrics = _compute_metrics(labels[val_mask], detector_norm[val_mask], threshold)
        return detector_norm, threshold, val_metrics, workflow_spec

    evidence_norm = _normalize(evidence_scores, fit_mask=val_mask)
    best_key = (-1.0, -1.0)
    best_scores = detector_norm
    best_threshold = 0.5
    best_weight = workflow_spec.detector_weight
    for detector_weight in np.linspace(0.2, 0.9, 8):
        fused = detector_weight * detector_norm + (1.0 - detector_weight) * evidence_norm
        threshold, _ = _search_threshold(labels[val_mask], fused[val_mask])
        metrics = _compute_metrics(labels[val_mask], fused[val_mask], threshold)
        key = (metrics["auc"], metrics["f1_macro"])
        if key > best_key:
            best_key = key
            best_scores = fused
            best_threshold = threshold
            best_weight = float(detector_weight)
    updated = replace(workflow_spec, detector_weight=best_weight)
    return best_scores, best_threshold, {"auc": best_key[0], "f1_macro": best_key[1]}, updated


def build_graph_task_card(dataset_path: str, user_description: str, config: dict[str, Any] | None = None):
    agent = TaskCardAgent(config=config)
    task_card, _graph_bundle = agent.build(dataset_path, user_description, config=config)
    return agent.validate(task_card)


def design_workflow(task_card, config: dict[str, Any] | None = None) -> WorkflowSpec:
    design_agent = DesignAgent(config=config)
    evidence_agent = EvidenceAgent(config=config)
    candidate = design_agent.design_candidates(task_card)[0]
    return evidence_agent.select_tools(task_card, candidate)


def run_toolkit(graph_bundle, workflow_spec):
    evidence_agent = EvidenceAgent()
    return evidence_agent.collect(graph_bundle, workflow_spec)


def evaluate_and_optimize(task_card, workflow_spec: WorkflowSpec | None = None, config: dict[str, Any] | None = None) -> ExperimentResult:
    task_agent = TaskCardAgent(config=config)
    detector_agent = DetectorAgent()
    design_agent = DesignAgent(config=config)
    evidence_agent = EvidenceAgent(config=config)
    judge_agent = JudgeAgent()
    optimizer_agent = OptimizerAgent()

    task_card, graph_bundle = task_agent.build(task_card.dataset_path, task_card.raw_user_description, config=config)
    task_card = task_agent.validate(task_card)
    _log_stage(
        "1 TaskCard",
        f"dataset={graph_bundle.dataset_id}, nodes={graph_bundle.num_nodes}, features={graph_bundle.num_features}, "
        f"relations={len(graph_bundle.relation_adjs)}, split={_count_mask(graph_bundle.train_mask)}/"
        f"{_count_mask(graph_bundle.val_mask)}/{_count_mask(graph_bundle.test_mask)}",
    )

    direct_workflow = (config or {}).get("direct_workflow", True)
    if direct_workflow:
        candidate = workflow_spec or design_workflow(task_card, config=config)
        if candidate.detector_name.startswith("GraphFeature") and "tool_policy" not in candidate.notes:
            candidate = replace(candidate, notes={**candidate.notes, "tool_policy": "detector_only"})
        candidates = [candidate]
    else:
        designed_candidates = design_agent.design_candidates(task_card)
        candidates = designed_candidates
        if workflow_spec is not None:
            candidates = [workflow_spec] + designed_candidates

    deduped_candidates: list[WorkflowSpec] = []
    seen = set()
    expanded_candidates: list[WorkflowSpec] = []
    for candidate in candidates:
        key = (candidate.detector_name, candidate.graph_view)
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(candidate)
    for candidate in deduped_candidates:
        if direct_workflow:
            expanded_candidates.append(candidate)
        elif candidate.detector_name.startswith("GraphFeature"):
            expanded_candidates.append(
                replace(
                    candidate,
                    notes={**candidate.notes, "tool_policy": "detector_only"},
                )
            )
            if "ToolAware" in candidate.detector_name or candidate.notes.get("internal_tools"):
                expanded_candidates.append(
                    replace(
                        candidate,
                        notes={**candidate.notes, "tool_policy": "tool_augmented"},
                    )
                )
        else:
            expanded_candidates.append(candidate)
    candidates = expanded_candidates
    _log_stage(
        "2 Design",
        f"candidates={len(candidates)}, first={candidates[0].detector_name}@{candidates[0].graph_view}",
    )
    traces: list[OptimizationTrace] = []
    labels = graph_bundle.labels

    for iteration, candidate in enumerate(candidates, start=1):
        candidate = evidence_agent.select_tools(task_card, candidate)
        _log_stage(
            "3 Evidence",
            f"candidate={iteration}/{len(candidates)}, tools={_format_names(candidate.tool_names)}",
        )
        detector_scores = detector_agent.score(graph_bundle, candidate)
        _log_stage(
            "4 Detector",
            f"{candidate.detector_name}@{candidate.graph_view} scored",
        )
        evidence_records = evidence_agent.collect(graph_bundle, candidate)
        aggregated_evidence, tool_weights = judge_agent.aggregate_evidence(evidence_records)
        fused_scores, threshold, val_metrics, candidate = _fuse_scores(
            detector_scores,
            aggregated_evidence,
            labels,
            graph_bundle.val_mask,
            replace(candidate, tool_weights=tool_weights),
        )
        traces.append(
            OptimizationTrace(
                iteration=iteration,
                workflow_spec=candidate,
                validation_metrics=val_metrics,
                test_metrics={},
                selected_threshold=threshold,
                detector_summary={
                    "detector_name": candidate.detector_name,
                    "graph_view": candidate.graph_view,
                    "score_mean": float(np.mean(detector_scores)),
                    "score_std": float(np.std(detector_scores)),
                },
                evidence_summary={
                    "tool_names": candidate.tool_names,
                    "tool_weights": tool_weights,
                    "num_records": len(evidence_records),
                },
            )
        )
        _log_stage(
            "5 Judge",
            f"val_auc={val_metrics['auc']:.4f}, val_f1={val_metrics['f1_macro']:.4f}, threshold={threshold:.4f}",
        )

    best_index = optimizer_agent.choose_best(traces)
    best = traces[best_index]
    _log_stage(
        "6 Optimizer",
        f"best={best.workflow_spec.detector_name}@{best.workflow_spec.graph_view}, "
        f"val_auc={best.validation_metrics['auc']:.4f}, "
        f"val_f1={best.validation_metrics['f1_macro']:.4f}",
    )
    final_best = best
    final_detector_summary = best.detector_summary
    final_evidence_summary = best.evidence_summary
    final_threshold = best.selected_threshold
    final_test_metrics = {}

    if (config or {}).get("final_refit", True):
        val_indices = np.where(graph_bundle.val_mask)[0]
        can_refit = val_indices.size >= 10 and np.unique(labels[val_indices]).size == 2
        if can_refit:
            refit_val_idx, calibration_idx = train_test_split(
                val_indices,
                train_size=float((config or {}).get("final_refit_train_share", 0.5)),
                stratify=labels[val_indices],
                random_state=int(graph_bundle.split_config.get("seed", 42)) + 17,
            )
            refit_train_mask = graph_bundle.train_mask.copy()
            refit_train_mask[refit_val_idx] = True
            calibration_mask = np.zeros_like(graph_bundle.val_mask, dtype=bool)
            calibration_mask[calibration_idx] = True
        else:
            refit_train_mask = graph_bundle.train_mask
            calibration_mask = graph_bundle.val_mask
        _log_stage(
            "7 FinalRefit",
            f"guard_start can_refit={can_refit}, train={_count_mask(refit_train_mask)}, cal={_count_mask(calibration_mask)}",
        )

        original_scores, original_evidence, _original_tool_weights, original_workflow = _score_workflow(graph_bundle, best.workflow_spec)
        _original_fused, original_cal_threshold, original_cal_metrics, _original_workflow = _fuse_scores(
            original_scores,
            original_evidence,
            labels,
            calibration_mask,
            original_workflow,
        )

        refit_bundle = replace(graph_bundle, train_mask=refit_train_mask, runtime_cache={})
        refit_scores, refit_evidence, refit_tool_weights, refit_workflow = _score_workflow(refit_bundle, best.workflow_spec)
        refit_fused_scores, refit_threshold, refit_calibration_metrics, refit_workflow = _fuse_scores(
            refit_scores,
            refit_evidence,
            labels,
            calibration_mask,
            refit_workflow,
        )
        refit_key = (round(refit_calibration_metrics["auc"], 4), round(refit_calibration_metrics["f1_macro"], 4))
        original_key = (round(original_cal_metrics["auc"], 4), round(original_cal_metrics["f1_macro"], 4))
        accept_refit = refit_key >= original_key
        _log_stage(
            "7 FinalRefit",
            f"accepted={accept_refit}, orig=({original_cal_metrics['auc']:.4f},{original_cal_metrics['f1_macro']:.4f}), "
            f"refit=({refit_calibration_metrics['auc']:.4f},{refit_calibration_metrics['f1_macro']:.4f})",
        )
        if accept_refit:
            refit_test_metrics = _compute_metrics(labels[graph_bundle.test_mask], refit_fused_scores[graph_bundle.test_mask], refit_threshold)
            final_best = replace(
                best,
                workflow_spec=replace(
                    refit_workflow,
                    notes={
                        **refit_workflow.notes,
                        "final_refit": "train_plus_validation_split",
                        "original_calibration_metrics": original_cal_metrics,
                        "refit_calibration_metrics": refit_calibration_metrics,
                    },
                ),
                test_metrics=refit_test_metrics,
                selected_threshold=refit_threshold,
                detector_summary={
                    "detector_name": refit_workflow.detector_name,
                    "graph_view": refit_workflow.graph_view,
                    "score_mean": float(np.mean(refit_scores)),
                    "score_std": float(np.std(refit_scores)),
                    "final_refit": True,
                },
                evidence_summary={
                    "tool_names": refit_workflow.tool_names,
                    "tool_weights": refit_tool_weights,
                    "num_records": len(refit_workflow.tool_weights),
                    "final_refit": True,
                },
            )
            final_detector_summary = final_best.detector_summary
            final_evidence_summary = final_best.evidence_summary
            final_threshold = final_best.selected_threshold
            final_test_metrics = final_best.test_metrics
            _log_stage(
                "7 FinalRefit",
                f"test_auc={refit_test_metrics['auc']:.4f}, test_f1={refit_test_metrics['f1_macro']:.4f}",
            )
        else:
            final_best = replace(
                best,
                workflow_spec=replace(
                    best.workflow_spec,
                    notes={
                        **best.workflow_spec.notes,
                        "final_refit": "rejected",
                        "original_calibration_metrics": original_cal_metrics,
                        "refit_calibration_metrics": refit_calibration_metrics,
                        "refit_rejected_threshold": refit_threshold,
                        "original_calibration_threshold": original_cal_threshold,
                    },
                ),
            )
            _log_stage("7 FinalRefit", "rejected; using original best workflow")

    if not final_test_metrics:
        final_scores, final_evidence, final_tool_weights, final_workflow = _score_workflow(graph_bundle, final_best.workflow_spec)
        final_fused_scores, final_threshold, final_validation_metrics, final_workflow = _fuse_scores(
            final_scores,
            final_evidence,
            labels,
            graph_bundle.val_mask,
            final_workflow,
        )
        final_test_metrics = _compute_metrics(labels[graph_bundle.test_mask], final_fused_scores[graph_bundle.test_mask], final_threshold)
        final_best = replace(
            final_best,
            workflow_spec=replace(
                final_workflow,
                notes={
                    **final_workflow.notes,
                    "final_refit": final_workflow.notes.get("final_refit", "not_applied"),
                    "final_validation_metrics": final_validation_metrics,
                },
            ),
            test_metrics=final_test_metrics,
            selected_threshold=final_threshold,
            detector_summary={
                "detector_name": final_workflow.detector_name,
                "graph_view": final_workflow.graph_view,
                "score_mean": float(np.mean(final_scores)),
                "score_std": float(np.std(final_scores)),
                "final_refit": False,
            },
            evidence_summary={
                "tool_names": final_workflow.tool_names,
                "tool_weights": final_tool_weights,
                "num_records": len(final_workflow.tool_weights),
                "final_refit": False,
            },
        )
        final_detector_summary = final_best.detector_summary
        final_evidence_summary = final_best.evidence_summary
        final_threshold = final_best.selected_threshold
        _log_stage(
            "Final Eval",
            f"test_auc={final_test_metrics['auc']:.4f}, test_f1={final_test_metrics['f1_macro']:.4f}",
        )
    return ExperimentResult(
        task_card=task_card,
        best_workflow=final_best.workflow_spec,
        optimization_trace=traces,
        best_validation_metrics=best.validation_metrics,
        final_test_metrics=final_test_metrics,
        selected_threshold=final_threshold,
        detector_summary=final_detector_summary,
        evidence_summary=final_evidence_summary,
    )


def run_agentgad_graph(dataset_path: str, user_description: str, config: dict[str, Any] | None = None) -> ExperimentResult:
    task_agent = TaskCardAgent(config=config)
    task_card, _graph_bundle = task_agent.build(dataset_path, user_description, config=config)
    task_card = task_agent.validate(task_card)
    initial_workflow = design_workflow(task_card, config=config)
    return evaluate_and_optimize(task_card, workflow_spec=initial_workflow, config=config)
