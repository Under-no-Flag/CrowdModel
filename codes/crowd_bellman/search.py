from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from .config import CaseOverrides, ObjectiveConfig, SearchConfig
from .metrics import save_json
from .runner import simulate_case
from .scenes import SimulationConfig, build_case_model, build_scene_for_case, build_three_channel_scene


def _format_probability_tag(probabilities: tuple[float, float, float]) -> str:
    scaled = [int(round(100.0 * value)) for value in probabilities]
    return f"p{scaled[0]:02d}-{scaled[1]:02d}-{scaled[2]:02d}"


def _evaluation_label(case_id: str, overrides: CaseOverrides | None) -> str:
    if overrides is None:
        return case_id

    parts = [case_id]
    if overrides.guidance_eta is not None:
        eta_text = f"{overrides.guidance_eta:.2f}".rstrip("0").rstrip(".")
        parts.append(f"eta-{eta_text}")
    if overrides.split_probabilities is not None:
        parts.append(_format_probability_tag(overrides.split_probabilities))
    return "__".join(parts)


def _evaluate_variant(
    cfg: SimulationConfig,
    case_id: str,
    output_root: Path,
    objective_cfg: ObjectiveConfig,
    overrides: CaseOverrides | None = None,
    cached_scene=None,
) -> dict[str, object]:
    scene = build_scene_for_case(case_id=case_id, cfg=cfg, cached_scene=cached_scene)
    case = build_case_model(case_id, scene, overrides=overrides)
    label = _evaluation_label(case_id, overrides)
    summary = simulate_case(
        cfg=cfg,
        scene=scene,
        case=case,
        output_dir=output_root / label,
        objective_cfg=objective_cfg,
    )
    summary["evaluation_id"] = label
    summary["base_case_id"] = case_id
    if overrides is not None:
        summary["case_overrides"] = asdict(overrides)
    save_json(output_root / label / "summary.json", summary)
    return summary


def _ranking_rows(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(summaries, key=lambda item: float(item.get("objective_value", float("inf"))))
    rows: list[dict[str, object]] = []
    for rank, summary in enumerate(ordered, start=1):
        rows.append(
            {
                "rank": rank,
                "evaluation_id": summary["evaluation_id"],
                "base_case_id": summary["base_case_id"],
                "objective_value": float(summary.get("objective_value", 0.0)),
                "j1_total_travel_time": float(summary.get("j1_total_travel_time", 0.0)),
                "j2_high_density_exposure": float(summary.get("j2_high_density_exposure", 0.0)),
                "j5_channel_flux_variance": float(summary.get("j5_channel_flux_variance", 0.0)),
                "peak_density_max": float(summary.get("peak_density_max", 0.0)),
                "final_sink_cumulative": float(summary.get("final_sink_cumulative", 0.0)),
            }
        )
    return rows


def _save_ranking_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_parameter_search(
    cfg: SimulationConfig,
    objective_cfg: ObjectiveConfig,
    search_cfg: SearchConfig,
    output_root: Path,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    shared_scene = build_three_channel_scene(cfg)

    discrete_results = [
        _evaluate_variant(
            cfg=cfg,
            case_id=case_id,
            output_root=output_root / "discrete",
            objective_cfg=objective_cfg,
            cached_scene=shared_scene,
        )
        for case_id in search_cfg.strategy_case_ids
    ]

    eta_results: list[dict[str, object]] = []
    for case_id in search_cfg.eta_case_ids:
        for eta in search_cfg.eta_values:
            eta_results.append(
                _evaluate_variant(
                    cfg=cfg,
                    case_id=case_id,
                    output_root=output_root / "eta_search",
                    objective_cfg=objective_cfg,
                    overrides=CaseOverrides(guidance_eta=float(eta)),
                    cached_scene=shared_scene,
                )
            )

    split_results = [
        _evaluate_variant(
            cfg=cfg,
            case_id=search_cfg.split_case_id,
            output_root=output_root / "split_search",
            objective_cfg=objective_cfg,
            overrides=CaseOverrides(split_probabilities=probabilities),
        )
        for probabilities in search_cfg.split_probability_candidates
    ]

    all_results = discrete_results + eta_results + split_results
    ranking_rows = _ranking_rows(all_results)
    best_result = ranking_rows[0] if ranking_rows else None

    payload = {
        "simulation_config": asdict(cfg),
        "objective_config": asdict(objective_cfg),
        "search_config": asdict(search_cfg),
        "best_result": best_result,
        "top_results": ranking_rows[: search_cfg.top_k],
        "discrete_results": discrete_results,
        "eta_results": eta_results,
        "split_results": split_results,
    }
    save_json(output_root / "search_summary.json", payload)
    _save_ranking_csv(output_root / "search_ranking.csv", ranking_rows)
    return payload
