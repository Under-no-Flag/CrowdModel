from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from crowd_bellman.config import ObjectiveConfig
from crowd_bellman.metrics import evaluate_objective_batch_from_summary


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_objective_configs(path: Path | None, summary: dict[str, object]) -> list[ObjectiveConfig]:
    if path is None:
        objective_config = summary.get("objective_config")
        if not isinstance(objective_config, dict):
            raise ValueError("No weights file provided and summary.json does not contain objective_config")
        return [ObjectiveConfig(**objective_config)]

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    objective_sets = raw.get("objective_sets")
    if isinstance(objective_sets, list) and objective_sets:
        return [ObjectiveConfig(**item) for item in objective_sets if isinstance(item, dict)]

    objective = raw.get("objective")
    if isinstance(objective, dict):
        return [ObjectiveConfig(**objective)]

    raise ValueError("Weights TOML must define [objective] or [[objective_sets]]")


def _evaluate_single(summary: dict[str, object], objective_cfgs: list[ObjectiveConfig]) -> dict[str, object]:
    return {
        "case_id": summary.get("case_id"),
        "title": summary.get("title"),
        "objective_terms": summary.get("objective_terms"),
        "objective_terms_normalized": summary.get("objective_terms_normalized"),
        "evaluations": evaluate_objective_batch_from_summary(summary, objective_cfgs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate J1/J2/J5/J for one or more weight sets from saved simulation results.")
    parser.add_argument("--input", required=True, help="Path to summary.json or comparison_summary.json")
    parser.add_argument("--weights", help="Optional TOML file with [objective] or [[objective_sets]]")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    payload = _load_json(input_path)

    if "cases" in payload:
        cases = payload.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("comparison_summary.json does not contain any cases")
        objective_cfgs = _load_objective_configs(Path(args.weights).resolve() if args.weights else None, cases[0])
        result = {
            "input_path": str(input_path),
            "weights_path": str(Path(args.weights).resolve()) if args.weights else None,
            "evaluations": [_evaluate_single(case, objective_cfgs) for case in cases if isinstance(case, dict)],
        }
    else:
        objective_cfgs = _load_objective_configs(Path(args.weights).resolve() if args.weights else None, payload)
        result = {
            "input_path": str(input_path),
            "weights_path": str(Path(args.weights).resolve()) if args.weights else None,
            "evaluation": _evaluate_single(payload, objective_cfgs),
        }

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_name(f"{input_path.stem}_objective_evaluations.json")

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
