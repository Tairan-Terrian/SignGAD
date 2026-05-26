from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.pipeline import build_graph_task_card, design_workflow, evaluate_and_optimize
from graph.dataset_downloads import ensure_default_fraud_datasets


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
RUNS_DIR = REPO_ROOT / "runs"


def log_stage(stage: str, message: str) -> None:
    print(f"[AgentGAD][{stage}] {message}")


def _candidate_config_paths(config_arg: str) -> list[Path]:
    raw = Path(config_arg)
    candidates = [raw]
    if not raw.suffix:
        candidates.extend(
            [
                DEFAULT_CONFIG_DIR / f"{config_arg}.yaml",
                DEFAULT_CONFIG_DIR / f"{config_arg}.yml",
                REPO_ROOT / f"{config_arg}.yaml",
                REPO_ROOT / f"{config_arg}.yml",
            ]
        )
    return candidates


def resolve_config_path(config_arg: str) -> Path:
    for candidate in _candidate_config_paths(config_arg):
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in _candidate_config_paths(config_arg))
    raise FileNotFoundError(f"Config file not found. Searched: {searched}")


def load_run_config(config_arg: str) -> dict:
    config_path = resolve_config_path(config_arg)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    if "dataset" not in payload:
        raise ValueError(f"Config file must define 'dataset': {config_path}")
    if "description" not in payload:
        raise ValueError(f"Config file must define 'description': {config_path}")

    payload["_config_path"] = str(config_path)
    return payload


def _slugify_dataset_name(dataset_name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in dataset_name)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "dataset"


def build_run_log_path(dataset_path: str, now: datetime | None = None) -> Path:
    dataset_name = Path(dataset_path).stem
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR / f"{_slugify_dataset_name(dataset_name)}_{timestamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph-only AgentGAD for node-level graph anomaly detection.")
    parser.add_argument("--config", help="Config file path, or a config name to resolve from ./configs/")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_config = load_run_config(args.config)
    ensure_default_fraud_datasets(REPO_ROOT / "data")
    config = {
        "split_config": {
            "train_ratio": run_config.get("train_ratio", 0.01),
            "val_ratio": run_config.get("val_ratio", 0.495),
            "test_ratio": run_config.get("test_ratio", 0.495),
            "seed": run_config.get("seed", 42),
        },
        "evidence_budget": run_config.get("evidence_budget", 4),
        "tool_budget": run_config.get("tool_budget", 3),
        "llm": {
            "enabled": run_config.get("llm_enabled", True),
            "model": run_config.get("llm_model"),
        },
        "final_refit": run_config.get("final_refit", True),
    }
    log_stage(
        "Init",
        f"config={run_config['_config_path']}, dataset={run_config['dataset']}, "
        f"llm={config['llm']['enabled']}, final_refit={config['final_refit']}",
    )

    task_card = build_graph_task_card(run_config["dataset"], run_config["description"], config)
    workflow = design_workflow(task_card, config=config)
    result = evaluate_and_optimize(task_card, workflow_spec=workflow, config=config)

    payload = result.to_dict()
    payload["config_path"] = run_config["_config_path"]
    run_log_path = build_run_log_path(run_config["dataset"])
    payload["run_log_path"] = str(run_log_path)
    run_log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    final_metrics = payload["final_test_metrics"]
    log_stage(
        "Final Result",
        f"detector={payload['best_workflow']['detector_name']}, view={payload['best_workflow']['graph_view']}, "
        f"threshold={payload['selected_threshold']:.4f}, "
        f"test_auc={final_metrics['auc']:.4f}, test_f1={final_metrics['f1_macro']:.4f}, log={run_log_path}",
    )
    print(f"AUC: {final_metrics['auc']:.4f}")
    print(f"F1-Macro: {final_metrics['f1_macro']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
