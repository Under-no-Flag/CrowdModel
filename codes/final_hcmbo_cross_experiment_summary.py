from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from paper_physical_units import FOUR_CHANNEL


REPO_ROOT = Path(__file__).resolve().parents[1]
G6_ROOT = REPO_ROOT / "codes" / "results" / "g6_horizontal_comparison"
G7C_ROOT = REPO_ROOT / "codes" / "results" / "g7c_hcmbo_tpe_comparison"
G7D_ROOT = REPO_ROOT / "codes" / "results" / "g7d_hcmbo_mainline"
OUTPUT_ROOT = REPO_ROOT / "codes" / "results" / "final_hcmbo_cross_experiment_summary"
PAPER_DPI = 300

METHOD_LABELS = {
    "final_hcmbo": "HCMBO",
    "tpe_mixed_bo": "TPE-Mixed BO",
    "pure_sa": "Pure SA",
    "random_search": "Random Search",
    "enum_de": "Enum-DE",
    "baseline_prior_best": "Prior Baseline",
    "hcmbo_structured_queue_aware": "HCMBO queue-aware",
}
METHOD_COLORS = {
    "final_hcmbo": "#55A868",
    "tpe_mixed_bo": "#8172B3",
    "pure_sa": "#4C78A8",
    "random_search": "#C44E52",
    "enum_de": "#E69F00",
    "baseline_prior_best": "#8C8C8C",
    "hcmbo_structured_queue_aware": "#76B7B2",
}
PAPER_METHOD_ORDER = (
    "baseline_prior_best",
    "random_search",
    "pure_sa",
    "enum_de",
    "tpe_mixed_bo",
    "final_hcmbo",
)
GATE_ROWS = (
    "top:plus",
    "top:minus",
    "middle:plus",
    "middle:minus",
    "lower_middle:plus",
    "lower_middle:minus",
    "bottom:plus",
    "bottom:minus",
)
GATE_LABELS = (
    "top plus",
    "top minus",
    "middle plus",
    "middle minus",
    "lower-mid plus",
    "lower-mid minus",
    "bottom plus",
    "bottom minus",
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    method_rows = build_final_method_rows()
    all_seed_rows = build_final_seed_rows()
    convergence_rows = build_final_convergence_rows()
    seed_rows = build_hcmbo_tpe_seed_rows()
    g7c_rows = build_g7c_focus_rows()
    write_csv(OUTPUT_ROOT / "final_method_summary.csv", method_rows)
    write_csv(OUTPUT_ROOT / "final_seed_summary.csv", all_seed_rows)
    write_csv(OUTPUT_ROOT / "final_convergence_curves.csv", convergence_rows)
    write_csv(OUTPUT_ROOT / "final_hcmbo_tpe_seed_comparison.csv", seed_rows)
    write_csv(OUTPUT_ROOT / "g7c_focus_comparison.csv", g7c_rows)
    save_best_objective_distribution(OUTPUT_ROOT / "final_method_best_objective.png", all_seed_rows, method_rows)
    save_method_objective_feasibility(OUTPUT_ROOT / "final_method_objective_feasibility.png", method_rows)
    save_convergence(OUTPUT_ROOT / "final_method_convergence.png", convergence_rows)
    save_control_profiles(OUTPUT_ROOT / "final_method_control_profiles.png", all_seed_rows)
    manifest = {
        "output_root": str(OUTPUT_ROOT),
        "sources": {
            "g6": str(G6_ROOT),
            "g7c": str(G7C_ROOT),
            "g7d": str(G7D_ROOT),
        },
        "figures": [
            "final_method_best_objective.png",
            "final_method_objective_feasibility.png",
            "final_method_convergence.png",
            "final_method_control_profiles.png",
        ],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote final HCMBO cross-experiment summary to {OUTPUT_ROOT}")


def build_final_method_rows() -> list[dict[str, object]]:
    g6 = {str(row["method"]): row for row in read_csv(G6_ROOT / "G6_method_summary.csv")}
    g7d = read_csv(G7D_ROOT / "G7D_method_summary.csv")[0]
    rows = [
        {
            "method": "final_hcmbo",
            "label": METHOD_LABELS["final_hcmbo"],
            "source": "G7-D",
            "seed_count": g7d["seed_count"],
            "mean_best_hf_objective": g7d["mean_best_hf_objective"],
            "std_best_hf_objective": g7d["std_best_hf_objective"],
            "median_best_hf_objective": g7d["median_best_hf_objective"],
            "best_hf_objective": g7d["best_hf_objective"],
            "worst_hf_objective": g7d["worst_hf_objective"],
            "feasible_rate": g7d["feasible_rate"],
        }
    ]
    for method in ("tpe_mixed_bo", "pure_sa", "random_search", "enum_de", "baseline_prior_best"):
        row = g6[method]
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "source": "G6",
                "seed_count": row["seed_count"],
                "mean_best_hf_objective": row["mean_best_hf_objective"],
                "std_best_hf_objective": row["std_best_hf_objective"],
                "median_best_hf_objective": row["median_best_hf_objective"],
                "best_hf_objective": row["best_hf_objective"],
                "worst_hf_objective": row["worst_hf_objective"],
                "feasible_rate": row["feasible_rate"],
            }
        )
    return sorted(rows, key=lambda item: to_float(item["mean_best_hf_objective"]))


def build_final_seed_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(G6_ROOT / "G6_seed_summary.csv"):
        method = str(row["method"])
        if method not in {"tpe_mixed_bo", "pure_sa", "random_search", "enum_de", "baseline_prior_best"}:
            continue
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "source": "G6",
                "seed": row["seed"],
                "best_hf_objective": row["best_hf_objective_default"],
                "best_case_id": row["best_case_id"],
                "feasible_best": row["feasible_best"],
                "J1_eval": row["J1_eval"],
                "J2_eval": row["J2_eval"],
                "J5_eval": row["J5_eval"],
                "gate_rejected": row["gate_rejected"],
                "output_dir": G6_ROOT / method / f"seed_{int(to_float(row['seed']))}",
            }
        )
    for row in read_csv(G7D_ROOT / "G7D_seed_summary.csv"):
        rows.append(
            {
                "method": "final_hcmbo",
                "label": METHOD_LABELS["final_hcmbo"],
                "source": "G7-D",
                "seed": row["seed"],
                "best_hf_objective": row["best_hf_objective"],
                "best_case_id": row["best_case_id"],
                "feasible_best": row["feasible_best"],
                "J1_eval": row["J1_eval"],
                "J2_eval": row["J2_eval"],
                "J5_eval": row["J5_eval"],
                "gate_rejected": row["gate_rejected"],
                "output_dir": row["output_dir"],
            }
        )
    return order_rows(rows)


def build_final_convergence_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(G6_ROOT / "G6_convergence_curves.csv"):
        method = str(row["method"])
        if method not in {"tpe_mixed_bo", "pure_sa", "random_search", "enum_de"}:
            continue
        item = dict(row)
        item["label"] = METHOD_LABELS[method]
        rows.append(item)
    for seed_dir in sorted((G7D_ROOT / "hcmbo").glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        eval_rows = read_csv(seed_dir / "G7D_evaluation_log.csv")
        best = math.inf
        index = 0
        for row in eval_rows:
            if str(row.get("fidelity")) != "mf":
                continue
            value = to_float(row.get("objective_value"))
            if not math.isfinite(value):
                continue
            index += 1
            best = min(best, value)
            rows.append(
                {
                    "method": "final_hcmbo",
                    "label": METHOD_LABELS["final_hcmbo"],
                    "seed": seed,
                    "evaluation_index": index,
                    "objective_value": value,
                    "best_so_far": best,
                    "source": row.get("source", ""),
                    "phase": row.get("phase", ""),
                    "fidelity": row.get("fidelity", ""),
                }
            )
    return order_rows(rows)


def build_hcmbo_tpe_seed_rows() -> list[dict[str, object]]:
    hcmbo_by_seed = {
        int(to_float(row["seed"])): row
        for row in read_csv(G7D_ROOT / "G7D_seed_summary.csv")
    }
    tpe_by_seed = {
        int(to_float(row["seed"])): row
        for row in read_csv(G6_ROOT / "G6_seed_summary.csv")
        if row["method"] == "tpe_mixed_bo"
    }
    rows = []
    for seed in sorted(set(hcmbo_by_seed) & set(tpe_by_seed)):
        hcmbo = hcmbo_by_seed[seed]
        tpe = tpe_by_seed[seed]
        h_value = to_float(hcmbo["best_hf_objective"])
        t_value = to_float(tpe["best_hf_objective_default"])
        rows.append(
            {
                "seed": seed,
                "hcmbo_objective": h_value,
                "tpe_objective": t_value,
                "delta_hcmbo_minus_tpe": h_value - t_value,
                "hcmbo_feasible": hcmbo["feasible_best"],
                "tpe_feasible": tpe["feasible_best"],
                "hcmbo_j2_eval": hcmbo["J2_eval"],
                "tpe_j2_eval": tpe["J2_eval"],
                "hcmbo_gate_rejected": hcmbo["gate_rejected"],
                "tpe_gate_rejected": tpe["gate_rejected"],
            }
        )
    return rows


def build_g7c_focus_rows() -> list[dict[str, object]]:
    wanted = ("hcmbo_structured_only", "tpe_mixed_bo", "hcmbo_structured_queue_aware")
    rows = []
    for row in read_csv(G7C_ROOT / "G7C_seed_summary.csv"):
        method = str(row["method"])
        if method in wanted:
            rows.append(
                {
                    "method": method if method != "hcmbo_structured_only" else "final_hcmbo",
                    "label": METHOD_LABELS.get(method, METHOD_LABELS.get("final_hcmbo", method)),
                    "seed": row["seed"],
                    "best_hf_objective": row["best_hf_objective"],
                    "feasible": row["feasible_best"],
                    "j2_eval": row["J2_eval"],
                    "gate_rejected": row["gate_rejected"],
                }
            )
    order = {"final_hcmbo": 0, "tpe_mixed_bo": 1, "hcmbo_structured_queue_aware": 2}
    return sorted(rows, key=lambda item: order[str(item["method"])])


def save_best_objective_distribution(path: Path, seed_rows: list[dict[str, object]], method_rows: list[dict[str, object]]) -> None:
    methods = [method for method in PAPER_METHOD_ORDER if any(str(row["method"]) == method for row in seed_rows)]
    means_by_method = {
        str(row["method"]): to_float(row["mean_best_hf_objective"])
        for row in method_rows
    }
    labels = [METHOD_LABELS[method] for method in methods]
    means = np.array([means_by_method.get(method, math.nan) for method in methods], dtype=float)
    y = np.arange(len(methods), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.7), dpi=PAPER_DPI)
    ax.barh(y, means, height=0.58, color=[METHOD_COLORS[method] for method in methods], alpha=0.86)
    for index, method in enumerate(methods):
        values = np.array(
            [
                to_float(row["best_hf_objective"])
                for row in seed_rows
                if str(row["method"]) == method and math.isfinite(to_float(row["best_hf_objective"]))
            ],
            dtype=float,
        )
        if values.size:
            value_min = float(np.min(values))
            value_max = float(np.max(values))
            median = float(np.median(values))
            ax.hlines(y[index], value_min, value_max, color="#303030", linewidth=1.2, alpha=0.72, zorder=3)
            ax.vlines([value_min, value_max], y[index] - 0.12, y[index] + 0.12, color="#303030", linewidth=1.2, alpha=0.72, zorder=3)
            ax.vlines(median, y[index] - 0.18, y[index] + 0.18, color="#FFFFFF", linewidth=2.0, alpha=0.95, zorder=4)
            ax.vlines(median, y[index] - 0.18, y[index] + 0.18, color="#303030", linewidth=0.75, alpha=0.85, zorder=5)
        if math.isfinite(means[index]):
            ax.text(means[index] + 0.035, y[index] + 0.24, f"mean {means[index]:.3f}", va="center", fontsize=8.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Best high-fidelity objective $J$ (dimensionless), lower is better")
    ax.set_title("Best objective across seeds")
    ax.grid(axis="x", alpha=0.24)
    finite = means[np.isfinite(means)]
    if finite.size:
        ax.set_xlim(0.0, float(np.max(finite)) * 1.16)
    save_fig(fig, path)


def save_method_objective_feasibility(path: Path, rows: list[dict[str, object]]) -> None:
    rows_by_method = {str(row["method"]): row for row in rows}
    ordered = [rows_by_method[method] for method in PAPER_METHOD_ORDER if method in rows_by_method]
    fig, (ax, ax_base) = plt.subplots(
        1,
        2,
        figsize=(7.5, 4.4),
        dpi=PAPER_DPI,
        sharey=True,
        gridspec_kw={"width_ratios": (4.3, 1.0), "wspace": 0.08},
    )
    label_offsets = {
        "final_hcmbo": (10, -2),
        "tpe_mixed_bo": (10, -1),
        "pure_sa": (-14, 10),
        "random_search": (-58, -18),
        "enum_de": (12, -18),
        "baseline_prior_best": (-78, 12),
    }
    for row in ordered:
        method = str(row["method"])
        mean_obj = to_float(row["mean_best_hf_objective"])
        feasible_rate = to_float(row["feasible_rate"])
        spread = max(to_float(row["std_best_hf_objective"]), 0.02)
        target_ax = ax_base if method == "baseline_prior_best" else ax
        target_ax.scatter(
            [mean_obj],
            [feasible_rate],
            s=520 * spread + 80,
            color=METHOD_COLORS[method],
            edgecolor="#222222",
            linewidth=0.7,
            alpha=0.86,
            zorder=3,
        )
        target_ax.annotate(
            METHOD_LABELS[method],
            xy=(mean_obj, feasible_rate),
            xytext=label_offsets.get(method, (7, 5)),
            textcoords="offset points",
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "lw": 0.55, "color": "0.35"} if method in {"random_search", "enum_de", "baseline_prior_best", "pure_sa"} else None,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.82},
            zorder=4,
        )
    ax.set_title("Optimizer region", fontsize=10.5, pad=6)
    ax_base.set_title("Prior", fontsize=10.5, pad=6)
    fig.suptitle("Objective-feasibility trade-off", y=0.965, fontsize=12.0)
    fig.supxlabel("Mean best high-fidelity objective, lower is better", y=0.055, fontsize=10.0)
    ax.set_ylabel("Feasible seed rate")
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(2.55, 3.20)
    ax_base.set_xlim(5.40, 5.56)
    ax.set_xticks([2.6, 2.8, 3.0, 3.2])
    ax_base.set_xticks([5.48])
    ax_base.tick_params(axis="y", left=False, labelleft=False)
    for current_ax in (ax, ax_base):
        current_ax.grid(True, alpha=0.24)
        current_ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax_base.spines["left"].set_visible(False)
    break_kwargs = {"marker": [(-1, -0.6), (1, 0.6)], "markersize": 8, "linestyle": "none", "color": "0.25", "mec": "0.25", "mew": 1.0, "clip_on": False}
    ax.plot([1, 1], [0, 1], transform=ax.transAxes, **break_kwargs)
    ax_base.plot([0, 0], [0, 1], transform=ax_base.transAxes, **break_kwargs)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.84, bottom=0.18, wspace=0.08)
    save_fig(fig, path, tight=False)


def save_seed_delta(path: Path, rows: list[dict[str, object]]) -> None:
    seeds = [int(row["seed"]) for row in rows]
    deltas = np.array([to_float(row["delta_hcmbo_minus_tpe"]) for row in rows], dtype=float)
    colors = ["#55A868" if value < 0 else "#C44E52" for value in deltas]
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), dpi=PAPER_DPI)
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.bar([str(seed) for seed in seeds], deltas, color=colors, alpha=0.9, width=0.62)
    for index, value in enumerate(deltas):
        va = "bottom" if value >= 0 else "top"
        offset = 0.012 if value >= 0 else -0.012
        ax.text(index, value + offset, f"{value:+.3f}", ha="center", va=va, fontsize=8.2)
    wins = int(np.sum(deltas < 0.0))
    ax.set_title(f"Paired final HCMBO vs TPE-Mixed BO ({wins}/{len(deltas)} seeds lower)")
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Objective difference: HCMBO - TPE")
    ax.grid(axis="y", alpha=0.22)
    ymin = min(-0.37, float(np.min(deltas)) - 0.04)
    ymax = max(0.09, float(np.max(deltas)) + 0.04)
    ax.set_ylim(ymin, ymax)
    save_fig(fig, path)


def save_g7c_focus(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    objective = np.array([to_float(row["best_hf_objective"]) for row in rows], dtype=float)
    rejected = np.array([to_float(row["gate_rejected"]) for row in rows], dtype=float)
    colors = [METHOD_COLORS[str(row["method"])] for row in rows]
    x = np.arange(len(rows), dtype=float)
    fig, ax1 = plt.subplots(1, 1, figsize=(7.0, 4.2), dpi=PAPER_DPI)
    ax1.bar(x - 0.16, objective, width=0.32, color=colors, alpha=0.88, label="best objective")
    ax1.set_ylabel("Best high-fidelity objective")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(2.3, 3.2)
    ax2 = ax1.twinx()
    ax2.plot(x + 0.16, rejected, color="#4D4D4D", marker="o", linewidth=1.6, label="unreleased flow")
    ax2.set_ylabel("Unreleased attempted flow")
    ax1.set_title("Seed-23 focused comparison and queue-aware trade-off")
    ax1.grid(axis="y", alpha=0.22)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save_fig(fig, path)


def save_convergence(path: Path, curve_rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in curve_rows:
        method = str(row.get("method"))
        if method == "baseline_prior_best":
            continue
        seed_value = to_float(row.get("seed"))
        if not math.isfinite(seed_value):
            continue
        grouped.setdefault((method, int(seed_value)), []).append(row)
    methods = [method for method in PAPER_METHOD_ORDER if method != "baseline_prior_best" and any(key[0] == method for key in grouped)]
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8), dpi=PAPER_DPI)
    plotted_series: list[dict[str, object]] = []
    for method in methods:
        seed_series = []
        max_eval = 0
        for (series_method, _seed), rows in grouped.items():
            if series_method != method:
                continue
            ordered = sorted(rows, key=lambda item: to_float(item.get("evaluation_index")))
            pairs = [
                (int(to_float(item.get("evaluation_index"))), to_float(item.get("best_so_far")))
                for item in ordered
                if math.isfinite(to_float(item.get("evaluation_index"))) and math.isfinite(to_float(item.get("best_so_far")))
            ]
            if not pairs:
                continue
            max_eval = max(max_eval, pairs[-1][0])
            seed_series.append(pairs)
        if not seed_series:
            continue
        x_grid = np.arange(1, min(max_eval, 400) + 1, dtype=int)
        matrix = np.vstack([forward_fill_series(series, x_grid) for series in seed_series])
        center = np.nanmedian(matrix, axis=0)
        q25 = np.nanpercentile(matrix, 25, axis=0)
        q75 = np.nanpercentile(matrix, 75, axis=0)
        color = METHOD_COLORS[method]
        zorder = 5 if method == "final_hcmbo" else 3
        linewidth = 2.3 if method == "final_hcmbo" else 1.8
        band_alpha = 0.24 if method == "final_hcmbo" else 0.12
        ax.plot(x_grid, center, color=color, linewidth=linewidth, label=METHOD_LABELS[method], zorder=zorder)
        ax.fill_between(x_grid, q25, q75, color=color, alpha=band_alpha, linewidth=0, zorder=max(1, zorder - 2))
        plotted_series.append({"method": method, "x": x_grid, "center": center, "q25": q25, "q75": q75, "color": color})
    ax.set_title("Best-so-far convergence")
    ax.set_xlabel("Optimization evaluations")
    ax.set_ylabel("Best objective $J$ so far (dimensionless)")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5)
    if plotted_series:
        axins = ax.inset_axes([0.52, 0.42, 0.43, 0.36])
        for series in plotted_series:
            x_grid = np.asarray(series["x"], dtype=int)
            center = np.asarray(series["center"], dtype=float)
            q25 = np.asarray(series["q25"], dtype=float)
            q75 = np.asarray(series["q75"], dtype=float)
            color = str(series["color"])
            method = str(series["method"])
            zorder = 5 if method == "final_hcmbo" else 3
            linewidth = 2.0 if method == "final_hcmbo" else 1.25
            band_alpha = 0.26 if method == "final_hcmbo" else 0.08
            axins.plot(x_grid, center, color=color, linewidth=linewidth, zorder=zorder)
            axins.fill_between(x_grid, q25, q75, color=color, alpha=band_alpha, linewidth=0, zorder=max(1, zorder - 2))
        axins.set_xlim(240, 400)
        axins.set_ylim(2.45, 3.25)
        axins.set_title("Final-stage detail", fontsize=8.2, pad=2)
        axins.set_xticks([250, 300, 350, 400])
        axins.set_yticks([2.5, 2.7, 2.9, 3.1])
        axins.tick_params(labelsize=7.2)
        axins.grid(True, alpha=0.22)
        ax.indicate_inset_zoom(axins, edgecolor="0.42", alpha=0.75)
    save_fig(fig, path)


def save_control_profiles(path: Path, seed_rows: list[dict[str, object]]) -> None:
    selected_rows: list[dict[str, object]] = []
    for method in PAPER_METHOD_ORDER:
        if method == "baseline_prior_best":
            continue
        candidates = [
            row for row in seed_rows
            if str(row["method"]) == method and math.isfinite(to_float(row["best_hf_objective"]))
        ]
        if not candidates:
            continue
        selected_rows.append(min(candidates, key=lambda item: to_float(item["best_hf_objective"])))
    controls = []
    for row in selected_rows:
        method = str(row["method"])
        seed = int(to_float(row["seed"]))
        if method == "final_hcmbo":
            control_path = G7D_ROOT / "hcmbo" / f"seed_{seed}" / "G7D_best_control.json"
        else:
            control_path = G6_ROOT / method / f"seed_{seed}" / "G6_best_control.json"
        control = load_json(control_path)
        directions = control.get("directions", {}) if isinstance(control, dict) else {}
        if not isinstance(directions, dict):
            directions = {}
        direction_label = " ".join(f"{channel_short(channel)}={directions.get(channel, '?')}" for channel in ("top", "middle", "lower_middle", "bottom"))
        controls.append(
            (
                f"{METHOD_LABELS[method]} | seed {seed} | {direction_label}",
                control_path,
                METHOD_COLORS[method],
            )
        )
    matrices = [control_matrix(load_json(path_item)) for _title, path_item, _color in controls]
    vmax = float(np.nanpercentile(np.vstack(matrices), 95))
    ncols = 2
    nrows = int(math.ceil(len(controls) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.6, max(4.2, 3.2 * nrows + 0.7)), dpi=PAPER_DPI, squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.86, top=0.88, bottom=0.10, wspace=0.28, hspace=0.70)
    axes_array = axes.ravel()
    image = None
    for index, (ax, (title, _path, color), matrix) in enumerate(zip(axes_array, controls, matrices)):
        image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=9.5, color=color, loc="left")
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["1", "2", "3", "4"])
        ax.set_yticks(np.arange(len(GATE_LABELS)))
        ax.set_yticklabels(GATE_LABELS if index % ncols == 0 else [], fontsize=7.2)
        ax.set_xlabel("Gate segment", labelpad=3)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if math.isfinite(value) and value > 0.0:
                    ax.text(col, row, f"{value:.2g}", ha="center", va="center", fontsize=7.4, color="white" if value > vmax * 0.58 else "0.12")
    for ax in axes_array[len(controls):]:
        ax.set_axis_off()
    fig.suptitle("Best-solution entrance-rate profiles", y=0.965, fontsize=12.0)
    if image is not None:
        cbar_ax = fig.add_axes([0.89, 0.17, 0.020, 0.64])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label("Rate upper bound q (ped/s)")
    save_fig(fig, path, tight=False)


def control_matrix(control: dict[str, object]) -> np.ndarray:
    q_by_gate = control.get("q_by_gate", {})
    if not isinstance(q_by_gate, dict):
        q_by_gate = {}
    rate_to_ped_per_sec = FOUR_CHANNEL.ped_per_sec_per_rate_unit
    rows = []
    for gate in GATE_ROWS:
        raw = q_by_gate.get(gate, [])
        values = [to_float(item) * rate_to_ped_per_sec for item in raw] if isinstance(raw, list) else []
        rows.append((values + [math.nan] * 4)[:4])
    return np.array(rows, dtype=float)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def order_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    order = {method: index for index, method in enumerate(PAPER_METHOD_ORDER)}
    return sorted(rows, key=lambda row: (order.get(str(row.get("method")), 99), to_float(row.get("seed")), to_float(row.get("evaluation_index"))))


def forward_fill_series(series: list[tuple[int, float]], x_grid: np.ndarray) -> np.ndarray:
    result = np.full(x_grid.shape, np.nan, dtype=float)
    index = 0
    current = math.nan
    for pos, x_value in enumerate(x_grid):
        while index < len(series) and series[index][0] <= int(x_value):
            current = series[index][1]
            index += 1
        result[pos] = current
    return result


def channel_short(channel: str) -> str:
    return {
        "top": "T",
        "middle": "M",
        "lower_middle": "LM",
        "bottom": "B",
    }.get(channel, channel)


def save_fig(fig: object, path: Path, *, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=PAPER_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


if __name__ == "__main__":
    main()
