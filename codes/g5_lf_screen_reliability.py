from __future__ import annotations

import argparse
import csv
import itertools
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from crowd_bellman.g5_hcmbo import (
    ALL_GATE_IDS,
    CHANNEL_NAMES,
    G5EvaluationCache,
    HCMBOConfig,
    V2ControlVector,
    control_from_capacity_mode,
    generate_direction_candidates,
    make_no_cap_control,
    qbar_from_reference,
)
from crowd_bellman.metrics import save_json
from g5_experiment_matrix import apply_profile_overrides, load_matrix_config, profile_from_name


DEFAULT_CONFIG = Path("codes/scenes/examples/g5_hcmbo_v2_small/g5.toml")
DEFAULT_OUTPUT_ROOT = Path("codes/results/g5_lf_screen_reliability")


@dataclass(frozen=True)
class SimulationBudget:
    label: str
    steps: int
    time_horizon: float
    bellman_every: int
    save_every: int = 100000
    density_contour_levels: int | str = 0

    def to_overrides(self) -> dict[str, object]:
        return {
            "steps": int(self.steps),
            "time_horizon": float(self.time_horizon),
            "bellman_every": int(self.bellman_every),
            "save_every": int(self.save_every),
            "density_contour_levels": self.density_contour_levels,
        }


@dataclass(frozen=True)
class EvalTask:
    index: int
    output_root: str
    baseline_config: str
    config: dict[str, object]
    budget: dict[str, object]
    fidelity_label: str
    fidelity_role: str
    directions: tuple[str, ...]
    mode: str
    q_by_gate: tuple[tuple[float, ...], ...]
    qbar_by_gate: dict[str, float]


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise SystemExit(f"Output root already exists and is not empty: {output_root}. Use --force to overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)

    profile, baseline_config, seed = load_profile(args)
    config = profile.config
    if args.direction_candidate_limit is not None:
        config = replace_config(config, direction_candidate_limit=int(args.direction_candidate_limit))
    if args.shortlist_size is not None:
        config = replace_config(config, shortlist_size=int(args.shortlist_size))
    if args.capacity_modes:
        config = replace_config(config, screen_capacity_modes=tuple(parse_string_list(args.capacity_modes)))

    screen_budgets = build_screen_budgets(args, default_bellman_every=profile.screen.bellman_every)
    target_budget = SimulationBudget(
        label=args.target_label,
        steps=int(args.target_steps),
        time_horizon=float(args.target_time_horizon),
        bellman_every=int(args.target_bellman_every),
        save_every=int(args.target_save_every),
        density_contour_levels=args.target_density_contour_levels,
    )

    rng = np.random.default_rng(seed)
    direction_candidates = generate_direction_candidates(config=config, rng=rng)
    direction_candidates = direction_candidates[: int(config.direction_candidate_limit)]

    reference_record = evaluate_reference(
        baseline_config=baseline_config,
        output_root=output_root,
        config=config,
        target_budget=target_budget,
    )
    qbar_by_gate = qbar_from_reference(reference_record.summary, config=config)

    tasks = build_eval_tasks(
        output_root=output_root,
        baseline_config=baseline_config,
        config=config,
        qbar_by_gate=qbar_by_gate,
        direction_candidates=direction_candidates,
        screen_budgets=screen_budgets,
        target_budget=target_budget,
    )
    rows = run_tasks(tasks, workers=int(args.workers))

    control_rows = [reference_row(reference_record, target_budget)] + rows
    write_csv(output_root / "G5_lf_control_evaluations.csv", control_rows)

    target_best = rank_by_direction(rows, target_budget.label)
    screen_best_by_label = {budget.label: rank_by_direction(rows, budget.label) for budget in screen_budgets}
    summary_rows = build_reliability_summary(
        target_best=target_best,
        screen_best_by_label=screen_best_by_label,
        screen_budgets=screen_budgets,
        shortlist_size=int(config.shortlist_size),
    )
    direction_rows = build_direction_audit(
        direction_candidates=direction_candidates,
        target_best=target_best,
        screen_best_by_label=screen_best_by_label,
        screen_budgets=screen_budgets,
        shortlist_size=int(config.shortlist_size),
    )
    stability_rows = build_shortlist_stability(
        screen_best_by_label=screen_best_by_label,
        screen_budgets=screen_budgets,
        shortlist_size=int(config.shortlist_size),
    )
    write_csv(output_root / "G5_lf_reliability_summary.csv", summary_rows)
    write_csv(output_root / "G5_lf_direction_rank_audit.csv", direction_rows)
    write_csv(output_root / "G5_lf_shortlist_stability.csv", stability_rows)
    write_report(
        output_root=output_root,
        config=config,
        screen_budgets=screen_budgets,
        target_budget=target_budget,
        summary_rows=summary_rows,
        direction_rows=direction_rows,
    )
    if not args.no_plots:
        write_plots(
            output_root=output_root,
            screen_budgets=screen_budgets,
            target_best=target_best,
            screen_best_by_label=screen_best_by_label,
            summary_rows=summary_rows,
        )

    manifest = {
        "status": "completed",
        "config_path": str(Path(args.config).resolve()),
        "baseline_config": str(baseline_config),
        "output_root": str(output_root),
        "seed": int(seed),
        "workers": int(args.workers),
        "direction_candidate_limit": int(config.direction_candidate_limit),
        "shortlist_size": int(config.shortlist_size),
        "screen_capacity_modes": list(config.screen_capacity_modes),
        "screen_budgets": [asdict(item) for item in screen_budgets],
        "target_budget": asdict(target_budget),
        "direction_candidates": [directions_dict(item) for item in direction_candidates],
        "qbar_by_gate": qbar_by_gate,
        "outputs": {
            "control_evaluations": str(output_root / "G5_lf_control_evaluations.csv"),
            "reliability_summary": str(output_root / "G5_lf_reliability_summary.csv"),
            "direction_rank_audit": str(output_root / "G5_lf_direction_rank_audit.csv"),
            "shortlist_stability": str(output_root / "G5_lf_shortlist_stability.csv"),
            "report": str(output_root / "G5_lf_reliability_report.md"),
            "rank_scatter": str(output_root / "G5_lf_rank_scatter.png"),
            "summary_plot": str(output_root / "G5_lf_reliability_summary.png"),
        },
    }
    save_json(output_root / "G5_lf_reliability_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LF screening reliability for G5 direction selection.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--direction-candidate-limit", type=int, default=None)
    parser.add_argument("--shortlist-size", type=int, default=None)
    parser.add_argument("--capacity-modes", default=None, help="Comma-separated screen capacity modes.")
    parser.add_argument("--screen-horizons", default="4,20,60")
    parser.add_argument("--screen-steps", default="60,240,600")
    parser.add_argument("--screen-bellman-every", default=None)
    parser.add_argument("--screen-save-every", type=int, default=100000)
    parser.add_argument("--target-label", default="target_160")
    parser.add_argument("--target-steps", type=int, default=1600)
    parser.add_argument("--target-time-horizon", type=float, default=160.0)
    parser.add_argument("--target-bellman-every", type=int, default=5)
    parser.add_argument("--target-save-every", type=int, default=100000)
    parser.add_argument("--target-density-contour-levels", default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_profile(args: argparse.Namespace):
    loaded = load_matrix_config(Path(args.config))
    seed = int(args.seed if args.seed is not None else loaded.seed if loaded.seed is not None else 23)
    profile = profile_from_name(loaded.profile_name or "full", seed)
    profile = apply_profile_overrides(profile, loaded.profile_overrides or {})
    baseline_config = loaded.baseline_config or Path("codes/scenes/examples/g2_multistage_directional/run_baseline.toml")
    return profile, baseline_config.resolve(), seed


def replace_config(config: HCMBOConfig, **kwargs: object) -> HCMBOConfig:
    data = dict(config.__dict__)
    data.update(kwargs)
    return HCMBOConfig(**data)


def build_screen_budgets(args: argparse.Namespace, *, default_bellman_every: int) -> list[SimulationBudget]:
    horizons = parse_float_list(args.screen_horizons)
    steps = parse_int_list(args.screen_steps)
    if len(horizons) != len(steps):
        raise ValueError("--screen-horizons and --screen-steps must have the same length")
    if args.screen_bellman_every is None:
        bellman_values = [int(default_bellman_every)] * len(horizons)
    else:
        bellman_values = parse_int_list(args.screen_bellman_every)
        if len(bellman_values) == 1:
            bellman_values *= len(horizons)
        if len(bellman_values) != len(horizons):
            raise ValueError("--screen-bellman-every must have length 1 or match --screen-horizons")
    return [
        SimulationBudget(
            label=f"lf_t{format_label_number(horizon)}",
            steps=int(step),
            time_horizon=float(horizon),
            bellman_every=int(bellman),
            save_every=int(args.screen_save_every),
            density_contour_levels=0,
        )
        for horizon, step, bellman in zip(horizons, steps, bellman_values)
    ]


def evaluate_reference(
    *,
    baseline_config: Path,
    output_root: Path,
    config: HCMBOConfig,
    target_budget: SimulationBudget,
):
    evaluator = G5EvaluationCache(
        baseline_config=baseline_config,
        output_root=output_root / "_reference",
        objective_config=config,
        simulation_overrides=target_budget.to_overrides(),
        fidelity="reference",
    )
    return evaluator.evaluate(
        make_no_cap_control(tuple("FREE" for _ in CHANNEL_NAMES), config.time_segments),
        source="qbar_reference_all_free",
        phase="reference",
        qbar_by_gate={gate_id: math.inf for gate_id in ALL_GATE_IDS},
    )


def build_eval_tasks(
    *,
    output_root: Path,
    baseline_config: Path,
    config: HCMBOConfig,
    qbar_by_gate: dict[str, float],
    direction_candidates: list[tuple[str, ...]],
    screen_budgets: list[SimulationBudget],
    target_budget: SimulationBudget,
) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    config_dict = dict(config.__dict__)
    task_index = 0
    for budget in [*screen_budgets, target_budget]:
        role = "target" if budget.label == target_budget.label else "screen"
        for directions in direction_candidates:
            for mode in config.screen_capacity_modes:
                task_index += 1
                control = control_from_capacity_mode(
                    directions=directions,
                    mode=mode,
                    qbar_by_gate=qbar_by_gate,
                    segment_count=config.time_segments,
                )
                tasks.append(
                    EvalTask(
                        index=task_index,
                        output_root=str(output_root / "_evaluations" / budget.label / f"{task_index:04d}_{mode}_{control.digest}"),
                        baseline_config=str(baseline_config),
                        config=config_dict,
                        budget=budget.to_overrides(),
                        fidelity_label=budget.label,
                        fidelity_role=role,
                        directions=control.directions,
                        mode=str(mode),
                        q_by_gate=control.q_by_gate,
                        qbar_by_gate=qbar_by_gate,
                    )
                )
    return tasks


def run_tasks(tasks: list[EvalTask], *, workers: int) -> list[dict[str, object]]:
    if workers <= 1:
        return [evaluate_task(task) for task in tasks]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(evaluate_task, task): task.index for task in tasks}
        for future in as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: int(row["task_index"]))


def evaluate_task(task: EvalTask) -> dict[str, object]:
    config = HCMBOConfig(**task.config)
    budget = dict(task.budget)
    output_root = Path(task.output_root)
    control = V2ControlVector(directions=task.directions, q_by_gate=task.q_by_gate).normalized()
    evaluator = G5EvaluationCache(
        baseline_config=Path(task.baseline_config),
        output_root=output_root,
        objective_config=config,
        simulation_overrides=budget,
        fidelity=task.fidelity_label,
    )
    record = evaluator.evaluate(
        control,
        source=f"{task.fidelity_label}_{task.mode}",
        phase=task.fidelity_role,
        qbar_by_gate=task.qbar_by_gate,
    )
    row = record.to_row()
    row.update(
        {
            "task_index": int(task.index),
            "fidelity_label": task.fidelity_label,
            "fidelity_role": task.fidelity_role,
            "mode": task.mode,
            "direction_key": direction_key(record.control.directions),
            "control_digest": record.control.digest,
            "steps": int(budget["steps"]),
            "time_horizon": float(budget["time_horizon"]),
            "bellman_every": int(budget["bellman_every"]),
            "case_output_root": str(output_root),
        }
    )
    return row


def reference_row(record: Any, target_budget: SimulationBudget) -> dict[str, object]:
    row = record.to_row()
    row.update(
        {
            "task_index": 0,
            "fidelity_label": "reference",
            "fidelity_role": "reference",
            "mode": "no_cap",
            "direction_key": direction_key(record.control.directions),
            "control_digest": record.control.digest,
            "steps": target_budget.steps,
            "time_horizon": target_budget.time_horizon,
            "bellman_every": target_budget.bellman_every,
            "case_output_root": "",
        }
    )
    return row


def rank_by_direction(rows: list[dict[str, object]], fidelity_label: str) -> list[dict[str, object]]:
    best_by_direction: dict[str, dict[str, object]] = {}
    for row in rows:
        if row.get("fidelity_label") != fidelity_label:
            continue
        key = str(row["direction_key"])
        current = best_by_direction.get(key)
        if current is None or to_float(row["objective_value"]) < to_float(current["objective_value"]):
            best_by_direction[key] = row
    ranked = sorted(best_by_direction.values(), key=lambda item: to_float(item["objective_value"]))
    result: list[dict[str, object]] = []
    for rank, row in enumerate(ranked, start=1):
        item = dict(row)
        item["direction_rank"] = rank
        result.append(item)
    return result


def build_reliability_summary(
    *,
    target_best: list[dict[str, object]],
    screen_best_by_label: dict[str, list[dict[str, object]]],
    screen_budgets: list[SimulationBudget],
    shortlist_size: int,
) -> list[dict[str, object]]:
    target_rank = rank_map(target_best)
    target_top = top_direction_keys(target_best, shortlist_size)
    target_best_key = top_direction_keys(target_best, 1)[0]
    rows: list[dict[str, object]] = []
    for budget in screen_budgets:
        screen_best = screen_best_by_label[budget.label]
        screen_rank = rank_map(screen_best)
        common = [key for key in target_rank if key in screen_rank]
        rank_errors = [abs(float(screen_rank[key] - target_rank[key])) for key in common]
        screen_top = top_direction_keys(screen_best, shortlist_size)
        overlap = sorted(set(screen_top) & set(target_top))
        rows.append(
            {
                "screen_label": budget.label,
                "screen_steps": budget.steps,
                "screen_time_horizon": budget.time_horizon,
                "screen_bellman_every": budget.bellman_every,
                "target_direction_count": len(target_rank),
                "spearman": spearman_from_ranks([screen_rank[key] for key in common], [target_rank[key] for key in common]),
                "mean_abs_rank_error": float(np.mean(rank_errors)) if rank_errors else None,
                "max_abs_rank_error": float(np.max(rank_errors)) if rank_errors else None,
                "top_k": shortlist_size,
                "top_k_overlap_count": len(overlap),
                "top_k_overlap_ratio": len(overlap) / max(shortlist_size, 1),
                "target_best_direction": target_best_key,
                "target_best_rank_in_screen": screen_rank.get(target_best_key),
                "target_best_in_screen_top_k": bool(target_best_key in set(screen_top)),
                "screen_top_k_directions": ";".join(screen_top),
                "target_top_k_directions": ";".join(target_top),
                "overlap_directions": ";".join(overlap),
            }
        )
    return rows


def build_direction_audit(
    *,
    direction_candidates: list[tuple[str, ...]],
    target_best: list[dict[str, object]],
    screen_best_by_label: dict[str, list[dict[str, object]]],
    screen_budgets: list[SimulationBudget],
    shortlist_size: int,
) -> list[dict[str, object]]:
    target_by_key = {str(row["direction_key"]): row for row in target_best}
    target_rank = rank_map(target_best)
    rows: list[dict[str, object]] = []
    for directions in direction_candidates:
        key = direction_key(directions)
        target = target_by_key.get(key, {})
        row: dict[str, object] = {
            "direction_key": key,
            **directions_dict(directions),
            "target_rank": target_rank.get(key),
            "target_objective": target.get("objective_value"),
            "target_mode": target.get("mode"),
            "target_feasible": target.get("feasible"),
            "target_j1_eval": target.get("j1_eval"),
            "target_j2_eval": target.get("j2_eval"),
            "target_j5_eval": target.get("j5_eval"),
            "target_jb_normalized": target.get("jb_normalized"),
            "target_gate_rejected": target.get("gate_rejected"),
        }
        for budget in screen_budgets:
            ranked = screen_best_by_label[budget.label]
            by_key = {str(item["direction_key"]): item for item in ranked}
            ranks = rank_map(ranked)
            item = by_key.get(key, {})
            rank = ranks.get(key)
            row[f"{budget.label}_rank"] = rank
            row[f"{budget.label}_objective"] = item.get("objective_value")
            row[f"{budget.label}_mode"] = item.get("mode")
            row[f"{budget.label}_feasible"] = item.get("feasible")
            row[f"{budget.label}_j2_eval"] = item.get("j2_eval")
            row[f"{budget.label}_j5_eval"] = item.get("j5_eval")
            row[f"{budget.label}_gate_rejected"] = item.get("gate_rejected")
            row[f"{budget.label}_rank_error"] = abs(int(rank) - int(target_rank[key])) if rank and key in target_rank else None
            row[f"{budget.label}_in_top_k"] = bool(rank is not None and int(rank) <= shortlist_size)
        rows.append(row)
    return sorted(rows, key=lambda item: int(item["target_rank"]) if item.get("target_rank") else 9999)


def build_shortlist_stability(
    *,
    screen_best_by_label: dict[str, list[dict[str, object]]],
    screen_budgets: list[SimulationBudget],
    shortlist_size: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(screen_budgets, 2):
        left_top = top_direction_keys(screen_best_by_label[left.label], shortlist_size)
        right_top = top_direction_keys(screen_best_by_label[right.label], shortlist_size)
        overlap = sorted(set(left_top) & set(right_top))
        rows.append(
            {
                "left_label": left.label,
                "right_label": right.label,
                "top_k": shortlist_size,
                "overlap_count": len(overlap),
                "overlap_ratio": len(overlap) / max(shortlist_size, 1),
                "left_top_k_directions": ";".join(left_top),
                "right_top_k_directions": ";".join(right_top),
                "overlap_directions": ";".join(overlap),
            }
        )
    return rows


def write_report(
    *,
    output_root: Path,
    config: HCMBOConfig,
    screen_budgets: list[SimulationBudget],
    target_budget: SimulationBudget,
    summary_rows: list[dict[str, object]],
    direction_rows: list[dict[str, object]],
) -> None:
    best_target = direction_rows[0] if direction_rows else {}
    lines = [
        "# G5 LF Screening Reliability Report",
        "",
        "## Scope",
        "",
        "- Purpose: test whether low-fidelity screening ranks the same directions as the 1600-step target fidelity.",
        f"- Direction candidates: `{config.direction_candidate_limit}`.",
        f"- Capacity modes per direction: `{', '.join(config.screen_capacity_modes)}`.",
        f"- Shortlist size: `{config.shortlist_size}`.",
        f"- Target fidelity: `{target_budget.steps}` steps, horizon `{target_budget.time_horizon}`.",
        "",
        "## Target Best Direction",
        "",
        f"- direction: `{best_target.get('direction_key')}`",
        f"- target rank: `{best_target.get('target_rank')}`",
        f"- target objective: `{best_target.get('target_objective')}`",
        f"- best capacity mode: `{best_target.get('target_mode')}`",
        f"- feasible: `{best_target.get('target_feasible')}`",
        "",
        "## LF Reliability Summary",
        "",
        "| LF label | horizon | Spearman | top-k overlap | target best rank in LF | hit target best |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {label} | {horizon} | {spearman:.6g} | {overlap}/{topk} | {best_rank} | {hit} |".format(
                label=row["screen_label"],
                horizon=row["screen_time_horizon"],
                spearman=to_float(row["spearman"]),
                overlap=row["top_k_overlap_count"],
                topk=row["top_k"],
                best_rank=row["target_best_rank_in_screen"],
                hit=row["target_best_in_screen_top_k"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `target_best_rank_in_screen` is the key diagnostic: if it is larger than the shortlist size, LF screening would discard the target-best direction.",
            "- Low Spearman or low top-k overlap means LF ranking is not reliable enough for hard direction pruning.",
            "- Use `G5_lf_direction_rank_audit.csv` to inspect which directions move most between LF and target fidelity.",
            "",
            "## Output Files",
            "",
            f"- `G5_lf_reliability_summary.csv`: `{output_root / 'G5_lf_reliability_summary.csv'}`",
            f"- `G5_lf_direction_rank_audit.csv`: `{output_root / 'G5_lf_direction_rank_audit.csv'}`",
            f"- `G5_lf_control_evaluations.csv`: `{output_root / 'G5_lf_control_evaluations.csv'}`",
            f"- `G5_lf_shortlist_stability.csv`: `{output_root / 'G5_lf_shortlist_stability.csv'}`",
        ]
    )
    (output_root / "G5_lf_reliability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(
    *,
    output_root: Path,
    screen_budgets: list[SimulationBudget],
    target_best: list[dict[str, object]],
    screen_best_by_label: dict[str, list[dict[str, object]]],
    summary_rows: list[dict[str, object]],
) -> None:
    target_rank = rank_map(target_best)
    fig, axes = plt.subplots(1, len(screen_budgets), figsize=(5.0 * len(screen_budgets), 4.2), dpi=150)
    if len(screen_budgets) == 1:
        axes = [axes]
    for ax, budget in zip(axes, screen_budgets):
        screen_rank = rank_map(screen_best_by_label[budget.label])
        common = [key for key in target_rank if key in screen_rank]
        ax.scatter([screen_rank[key] for key in common], [target_rank[key] for key in common], s=32)
        limit = max(len(common), 1)
        ax.plot([1, limit], [1, limit], color="black", linewidth=1, alpha=0.5)
        ax.set_title(budget.label)
        ax.set_xlabel("LF rank")
        ax.set_ylabel("Target rank")
        ax.set_xlim(0.5, limit + 0.5)
        ax.set_ylim(0.5, limit + 0.5)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_root / "G5_lf_rank_scatter.png")
    plt.close(fig)

    labels = [str(row["screen_label"]) for row in summary_rows]
    spearman = [to_float(row["spearman"]) for row in summary_rows]
    overlap = [to_float(row["top_k_overlap_ratio"]) for row in summary_rows]
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2), dpi=150)
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, spearman, width=width, label="Spearman")
    ax.bar(x + width / 2, overlap, width=width, label="Top-k overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_root / "G5_lf_reliability_summary.png")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rank_map(ranked_rows: list[dict[str, object]]) -> dict[str, int]:
    return {str(row["direction_key"]): int(row["direction_rank"]) for row in ranked_rows}


def top_direction_keys(ranked_rows: list[dict[str, object]], limit: int) -> list[str]:
    return [str(row["direction_key"]) for row in sorted(ranked_rows, key=lambda item: int(item["direction_rank"]))[: max(1, int(limit))]]


def spearman_from_ranks(left: list[int], right: list[int]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1.0e-12:
        return None
    return float(np.sum(x * y) / denom)


def directions_dict(directions: tuple[str, ...]) -> dict[str, str]:
    return {f"direction_{name}": state for name, state in zip(CHANNEL_NAMES, directions)}


def direction_key(directions: tuple[str, ...]) -> str:
    return ",".join(f"{name}:{state}" for name, state in zip(CHANNEL_NAMES, directions))


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_string_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def format_label_number(value: float) -> str:
    if abs(value - int(value)) <= 1.0e-12:
        return str(int(value))
    return str(value).replace(".", "p")


def to_float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
