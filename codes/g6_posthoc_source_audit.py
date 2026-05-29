from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from crowd_bellman.g5_hcmbo import ALL_GATE_IDS, CHANNEL_NAMES


DEFAULT_INPUT_ROOT = Path("codes/results/g6_horizontal_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc source audit for completed G6 results; no simulation reruns.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else input_root / "g7a_posthoc_source_audit"
    output_root.mkdir(parents=True, exist_ok=True)

    augmented_hf, optimization_rows = collect_augmented_rows(input_root)
    write_csv(output_root / "G7A_hf_candidates_with_origin.csv", augmented_hf)
    write_csv(output_root / "G7A_optimization_rows_with_source_family.csv", optimization_rows)

    write_csv(output_root / "G7A_source_metric_summary.csv", build_source_metric_summary(optimization_rows, augmented_hf))
    write_csv(output_root / "G7A_hcmbo_source_best_by_seed.csv", build_hcmbo_source_best_by_seed(optimization_rows, augmented_hf))
    write_csv(output_root / "G7A_hcmbo_hf_top10_origin.csv", build_top_hf_origin_rows(augmented_hf, top_k=args.top_k))
    write_csv(output_root / "G7A_hcmbo_vs_tpe_best_feasible.csv", build_hcmbo_vs_tpe_rows(augmented_hf))
    write_csv(output_root / "G7A_same_direction_capacity_differences.csv", build_same_direction_capacity_rows(augmented_hf))
    pareto_rows = build_pareto_rows(augmented_hf)
    write_csv(output_root / "G7A_pareto_j2_j5_gate.csv", pareto_rows)
    write_csv(output_root / "G7A_topk_direction_concentration.csv", build_topk_direction_concentration(augmented_hf))

    draw_pareto_plot(output_root / "G7A_pareto_j2_j5_gate.png", pareto_rows)
    write_report(output_root, augmented_hf, optimization_rows)
    print(f"G7-A posthoc audit written to {output_root}")


def collect_augmented_rows(input_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    augmented_hf: list[dict[str, object]] = []
    optimization_rows: list[dict[str, object]] = []
    for eval_log in sorted(input_root.glob("*/*/G6_evaluation_log.csv")):
        run_dir = eval_log.parent
        method = run_dir.parent.name
        seed = run_dir.name.replace("seed_", "")
        rows = read_csv(eval_log)
        origin_by_signature: dict[str, dict[str, object]] = {}
        for row in rows:
            row_method = str(row.get("method") or method)
            row_seed = str(row.get("seed") or seed)
            fidelity = str(row.get("fidelity", ""))
            phase = str(row.get("phase", ""))
            source = str(row.get("source", ""))
            if fidelity == "mf":
                family = source_family(method=row_method, source=source)
                out = dict(row)
                out["method"] = row_method
                out["seed"] = row_seed
                out["source_family"] = family
                out["directions"] = directions_key(row)
                optimization_rows.append(out)

                signature = control_signature(row)
                previous = origin_by_signature.get(signature)
                if previous is None or to_float(row.get("objective_value")) < to_float(previous.get("objective_value")):
                    origin_by_signature[signature] = out
            elif fidelity not in {"hf", "reference"} and phase not in {"reference", "high_fidelity"}:
                signature = control_signature(row)
                previous = origin_by_signature.get(signature)
                if previous is None or to_float(row.get("objective_value")) < to_float(previous.get("objective_value")):
                    out = dict(row)
                    out["method"] = row_method
                    out["seed"] = row_seed
                    out["source_family"] = source_family(method=row_method, source=source)
                    out["directions"] = directions_key(row)
                    origin_by_signature[signature] = out

        hf_rows = [row for row in rows if str(row.get("fidelity", "")) == "hf"]
        for rank, row in enumerate(sorted(hf_rows, key=lambda item: to_float(item.get("objective_value"))), start=1):
            row_method = str(row.get("method") or method)
            row_seed = str(row.get("seed") or seed)
            origin = origin_by_signature.get(control_signature(row), {})
            out = dict(row)
            out["method"] = row_method
            out["seed"] = row_seed
            out["hf_rank_in_run"] = rank
            out["directions"] = directions_key(row)
            out["original_source"] = origin.get("source", "")
            out["original_phase"] = origin.get("phase", "")
            out["original_case_id"] = origin.get("case_id", "")
            out["original_objective_value"] = origin.get("objective_value", "")
            out["original_source_family"] = origin.get("source_family", source_family(method=row_method, source=str(origin.get("source", ""))))
            augmented_hf.append(out)
    return augmented_hf, optimization_rows


def build_source_metric_summary(
    optimization_rows: list[dict[str, object]],
    augmented_hf: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in optimization_rows:
        groups[(str(row.get("method")), str(row.get("seed")), str(row.get("source_family")))].append(row)
    for (method, seed, family), items in sorted(groups.items()):
        rows.append(metric_summary_row(method=method, seed=seed, source_family=family, stage="optimization", items=items))

    hf_groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in augmented_hf:
        hf_groups[(str(row.get("method")), str(row.get("seed")), str(row.get("original_source_family")))].append(row)
    for (method, seed, family), items in sorted(hf_groups.items()):
        rows.append(metric_summary_row(method=method, seed=seed, source_family=family, stage="high_fidelity_selected", items=items))
    return rows


def metric_summary_row(
    *,
    method: str,
    seed: str,
    source_family: str,
    stage: str,
    items: list[dict[str, object]],
) -> dict[str, object]:
    objectives = finite_values(item.get("objective_value") for item in items)
    j2s = finite_values(item.get("j2_eval") for item in items)
    j5s = finite_values(item.get("j5_eval") for item in items)
    rejected = finite_values(item.get("gate_rejected") for item in items)
    feasible = [parse_bool(item.get("feasible")) for item in items if parse_bool(item.get("feasible")) is not None]
    best = min(items, key=lambda item: to_float(item.get("objective_value"))) if items else {}
    return {
        "method": method,
        "seed": seed,
        "source_family": source_family,
        "stage": stage,
        "count": len(items),
        "best_objective": min(objectives) if objectives else "",
        "mean_objective": float(np.mean(objectives)) if objectives else "",
        "mean_j2_eval": float(np.mean(j2s)) if j2s else "",
        "mean_j5_eval": float(np.mean(j5s)) if j5s else "",
        "mean_gate_rejected": float(np.mean(rejected)) if rejected else "",
        "feasible_rate": sum(1 for item in feasible if item) / len(feasible) if feasible else "",
        "best_case_id": best.get("case_id", ""),
        "best_directions": directions_key(best) if best else "",
    }


def build_hcmbo_source_best_by_seed(
    optimization_rows: list[dict[str, object]],
    augmented_hf: list[dict[str, object]],
) -> list[dict[str, object]]:
    seeds = sorted({str(row.get("seed")) for row in optimization_rows if str(row.get("method")) == "hcmbo_proposed"})
    rows: list[dict[str, object]] = []
    for seed in seeds:
        opt_items = [row for row in optimization_rows if str(row.get("method")) == "hcmbo_proposed" and str(row.get("seed")) == seed]
        hf_items = [row for row in augmented_hf if str(row.get("method")) == "hcmbo_proposed" and str(row.get("seed")) == seed]
        for family in ("hcmbo_structured", "hcmbo_internal_random", "combined_pool"):
            if family == "combined_pool":
                opt_subset = opt_items
                hf_subset = hf_items
            else:
                opt_subset = [row for row in opt_items if str(row.get("source_family")) == family]
                hf_subset = [row for row in hf_items if str(row.get("original_source_family")) == family]
            best_opt = min(opt_subset, key=lambda item: to_float(item.get("objective_value"))) if opt_subset else {}
            best_hf = min(hf_subset, key=lambda item: to_float(item.get("objective_value"))) if hf_subset else {}
            rows.append(
                {
                    "seed": seed,
                    "pool": family,
                    "optimization_count": len(opt_subset),
                    "selected_hf_count": len(hf_subset),
                    "best_mf_objective": best_opt.get("objective_value", ""),
                    "best_mf_case_id": best_opt.get("case_id", ""),
                    "best_mf_source": best_opt.get("source", ""),
                    "best_mf_directions": directions_key(best_opt) if best_opt else "",
                    "best_selected_hf_objective": best_hf.get("objective_value", ""),
                    "best_selected_hf_case_id": best_hf.get("case_id", ""),
                    "best_selected_hf_original_case_id": best_hf.get("original_case_id", ""),
                    "best_selected_hf_feasible": best_hf.get("feasible", ""),
                    "best_selected_hf_gate_rejected": best_hf.get("gate_rejected", ""),
                    "note": "HF values are only available for candidates selected by the original combined-pool top-k.",
                }
            )
    return rows


def build_top_hf_origin_rows(augmented_hf: list[dict[str, object]], *, top_k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in augmented_hf:
        if str(row.get("method")) == "hcmbo_proposed":
            groups[(str(row.get("method")), str(row.get("seed")))].append(row)
    for (_method, seed), items in sorted(groups.items()):
        for rank, row in enumerate(sorted(items, key=lambda item: to_float(item.get("objective_value")))[:top_k], start=1):
            rows.append(
                {
                    "seed": seed,
                    "rank": rank,
                    "hf_objective": row.get("objective_value"),
                    "hf_case_id": row.get("case_id"),
                    "feasible": row.get("feasible"),
                    "directions": row.get("directions"),
                    "original_source_family": row.get("original_source_family"),
                    "original_source": row.get("original_source"),
                    "original_phase": row.get("original_phase"),
                    "original_case_id": row.get("original_case_id"),
                    "original_objective_value": row.get("original_objective_value"),
                    "j2_eval": row.get("j2_eval"),
                    "j5_eval": row.get("j5_eval"),
                    "gate_rejected": row.get("gate_rejected"),
                }
            )
    return rows


def build_hcmbo_vs_tpe_rows(augmented_hf: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seeds = sorted({str(row.get("seed")) for row in augmented_hf if str(row.get("method")) in {"hcmbo_proposed", "tpe_mixed_bo"}})
    for seed in seeds:
        best_by_method: dict[str, dict[str, object]] = {}
        for method in ("hcmbo_proposed", "tpe_mixed_bo"):
            items = [
                row
                for row in augmented_hf
                if str(row.get("seed")) == seed and str(row.get("method")) == method and parse_bool(row.get("feasible")) is True
            ]
            if items:
                best_by_method[method] = min(items, key=lambda item: to_float(item.get("objective_value")))
        hcmbo = best_by_method.get("hcmbo_proposed", {})
        tpe = best_by_method.get("tpe_mixed_bo", {})
        rows.append(
            {
                "seed": seed,
                "hcmbo_best_feasible_hf": hcmbo.get("objective_value", ""),
                "hcmbo_case_id": hcmbo.get("case_id", ""),
                "hcmbo_directions": hcmbo.get("directions", ""),
                "hcmbo_j2_eval": hcmbo.get("j2_eval", ""),
                "hcmbo_gate_rejected": hcmbo.get("gate_rejected", ""),
                "hcmbo_origin": hcmbo.get("original_source_family", ""),
                "tpe_best_feasible_hf": tpe.get("objective_value", ""),
                "tpe_case_id": tpe.get("case_id", ""),
                "tpe_directions": tpe.get("directions", ""),
                "tpe_j2_eval": tpe.get("j2_eval", ""),
                "tpe_gate_rejected": tpe.get("gate_rejected", ""),
                "delta_hcmbo_minus_tpe": to_float(hcmbo.get("objective_value")) - to_float(tpe.get("objective_value")),
            }
        )
    return rows


def build_same_direction_capacity_rows(augmented_hf: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in augmented_hf:
        method = str(row.get("method"))
        if method not in {"hcmbo_proposed", "tpe_mixed_bo"}:
            continue
        grouped[(str(row.get("seed")), method, str(row.get("directions")))].append(row)

    rows: list[dict[str, object]] = []
    keys = sorted({(seed, directions) for seed, method, directions in grouped if method in {"hcmbo_proposed", "tpe_mixed_bo"}})
    for seed, directions in keys:
        h_items = grouped.get((seed, "hcmbo_proposed", directions), [])
        t_items = grouped.get((seed, "tpe_mixed_bo", directions), [])
        if not h_items or not t_items:
            continue
        h = min(h_items, key=lambda item: to_float(item.get("objective_value")))
        t = min(t_items, key=lambda item: to_float(item.get("objective_value")))
        row: dict[str, object] = {
            "seed": seed,
            "directions": directions,
            "hcmbo_objective": h.get("objective_value"),
            "tpe_objective": t.get("objective_value"),
            "delta_hcmbo_minus_tpe": to_float(h.get("objective_value")) - to_float(t.get("objective_value")),
            "hcmbo_case_id": h.get("case_id"),
            "tpe_case_id": t.get("case_id"),
            "hcmbo_gate_rejected": h.get("gate_rejected"),
            "tpe_gate_rejected": t.get("gate_rejected"),
            "hcmbo_j2_eval": h.get("j2_eval"),
            "tpe_j2_eval": t.get("j2_eval"),
        }
        for gate_id in ALL_GATE_IDS:
            safe = gate_id.replace(":", "_")
            h_mean = profile_mean(h.get(f"q_{safe}"))
            t_mean = profile_mean(t.get(f"q_{safe}"))
            row[f"hcmbo_mean_q_{safe}"] = h_mean if math.isfinite(h_mean) else ""
            row[f"tpe_mean_q_{safe}"] = t_mean if math.isfinite(t_mean) else ""
            row[f"delta_mean_q_{safe}"] = h_mean - t_mean if math.isfinite(h_mean) and math.isfinite(t_mean) else ""
        rows.append(row)
    return rows


def build_pareto_rows(augmented_hf: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in augmented_hf:
        rows.append(
            {
                "method": row.get("method"),
                "seed": row.get("seed"),
                "case_id": row.get("case_id"),
                "objective_value": row.get("objective_value"),
                "j2_eval": row.get("j2_eval"),
                "j5_eval": row.get("j5_eval"),
                "gate_rejected": row.get("gate_rejected"),
                "waiting_mass_peak": row.get("waiting_mass_peak"),
                "binding_time_ratio_max": row.get("binding_time_ratio_max"),
                "feasible": row.get("feasible"),
                "directions": row.get("directions"),
                "original_source_family": row.get("original_source_family"),
            }
        )
    return rows


def build_topk_direction_concentration(augmented_hf: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in augmented_hf:
        groups[(str(row.get("method")), str(row.get("seed")))].append(row)
    for (method, seed), items in sorted(groups.items()):
        ordered = sorted(items, key=lambda item: to_float(item.get("objective_value")))
        for k in (3, 5, 10):
            subset = ordered[: min(k, len(ordered))]
            counts = Counter(str(row.get("directions")) for row in subset)
            top_direction, top_count = counts.most_common(1)[0] if counts else ("", 0)
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "top_k": k,
                    "candidate_count": len(subset),
                    "unique_direction_count": len(counts),
                    "top_direction": top_direction,
                    "top_direction_share": top_count / len(subset) if subset else "",
                }
            )
    return rows


def draw_pareto_plot(path: Path, rows: list[dict[str, object]]) -> None:
    plot_rows = [
        row
        for row in rows
        if math.isfinite(to_float(row.get("j2_eval")))
        and math.isfinite(to_float(row.get("j5_eval")))
        and math.isfinite(to_float(row.get("objective_value")))
    ]
    if not plot_rows:
        return
    methods = sorted({str(row.get("method")) for row in plot_rows})
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    cmap = plt.get_cmap("tab10")
    for index, method in enumerate(methods):
        items = [row for row in plot_rows if str(row.get("method")) == method]
        xs = [to_float(row.get("j2_eval")) for row in items]
        ys = [to_float(row.get("j5_eval")) for row in items]
        sizes = [20.0 + min(120.0, max(0.0, to_float(row.get("gate_rejected")) / 30.0)) for row in items]
        ax.scatter(xs, ys, s=sizes, alpha=0.72, label=method, color=cmap(index % 10), edgecolors="none")
    ax.set_xlabel("J2_eval")
    ax.set_ylabel("J5_eval")
    ax.set_title("G6 HF candidates: J2/J5/gate-rejection Pareto view")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(output_root: Path, augmented_hf: list[dict[str, object]], optimization_rows: list[dict[str, object]]) -> None:
    hcmbo_hf = [row for row in augmented_hf if str(row.get("method")) == "hcmbo_proposed"]
    tpe_hf = [row for row in augmented_hf if str(row.get("method")) == "tpe_mixed_bo"]
    best_hcmbo = min(hcmbo_hf, key=lambda row: to_float(row.get("objective_value"))) if hcmbo_hf else {}
    best_tpe = min(tpe_hf, key=lambda row: to_float(row.get("objective_value"))) if tpe_hf else {}
    origin_counts = Counter(str(row.get("original_source_family")) for row in hcmbo_hf)
    lines = [
        "# G7-A Posthoc Source Audit",
        "",
        "This audit reuses completed G6 CSV files only; it does not rerun simulations.",
        "",
        "## Key Findings",
        "",
        f"- HCMBO selected HF candidates by origin: {dict(origin_counts)}.",
        f"- Best HCMBO HF candidate: `{best_hcmbo.get('objective_value', '')}` from `{best_hcmbo.get('original_source_family', '')}`, case `{best_hcmbo.get('case_id', '')}`.",
        f"- Best TPE HF candidate: `{best_tpe.get('objective_value', '')}`, case `{best_tpe.get('case_id', '')}`.",
        "- `structured-only` and `internal-random-only` HF scores are limited to candidates that the original combined top-k actually rechecked.",
        "",
        "## Outputs",
        "",
    ]
    for name in (
        "G7A_hf_candidates_with_origin.csv",
        "G7A_optimization_rows_with_source_family.csv",
        "G7A_source_metric_summary.csv",
        "G7A_hcmbo_source_best_by_seed.csv",
        "G7A_hcmbo_hf_top10_origin.csv",
        "G7A_hcmbo_vs_tpe_best_feasible.csv",
        "G7A_same_direction_capacity_differences.csv",
        "G7A_pareto_j2_j5_gate.csv",
        "G7A_topk_direction_concentration.csv",
        "G7A_pareto_j2_j5_gate.png",
    ):
        lines.append(f"- `{name}`")
    output_root.joinpath("G7A_posthoc_source_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_family(*, method: str, source: str) -> str:
    if method == "hcmbo_proposed" and source.startswith("hcmbo"):
        return "hcmbo_structured"
    if method == "hcmbo_proposed" and source == "random_search":
        return "hcmbo_internal_random"
    if source.startswith("baseline"):
        return "baseline_policy"
    if source:
        return source
    return "unknown"


def control_signature(row: dict[str, object]) -> str:
    parts = [str(row.get(f"direction_{name}", "")) for name in CHANNEL_NAMES]
    for gate_id in ALL_GATE_IDS:
        safe = gate_id.replace(":", "_")
        parts.append(normalize_profile_string(row.get(f"q_{safe}", "")))
    return "|".join(parts)


def normalize_profile_string(value: object) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    parts: list[str] = []
    for item in raw.split(";"):
        token = item.strip().lower()
        if token in {"inf", "infinity"}:
            parts.append("inf")
            continue
        number = to_float(token)
        parts.append(f"{number:.6g}" if math.isfinite(number) else token)
    return ";".join(parts)


def directions_key(row: dict[str, object]) -> str:
    if not row:
        return ""
    return ",".join(f"{name}:{row.get(f'direction_{name}', '')}" for name in CHANNEL_NAMES)


def finite_values(values: Iterable[object]) -> list[float]:
    result = []
    for value in values:
        number = to_float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def profile_mean(value: object) -> float:
    raw = str(value or "")
    if not raw:
        return math.nan
    values = []
    for item in raw.split(";"):
        token = item.strip().lower()
        if token in {"inf", "infinity"}:
            continue
        number = to_float(token)
        if math.isfinite(number):
            values.append(number)
    return float(np.mean(values)) if values else math.nan


def parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


if __name__ == "__main__":
    main()
