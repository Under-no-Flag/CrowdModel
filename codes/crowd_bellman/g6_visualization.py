from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = (
    "baseline_prior_best",
    "random_search",
    "pure_sa",
    "enum_de",
    "hcmbo_proposed",
    "tpe_mixed_bo",
)
METHOD_LABELS = {
    "baseline_prior_best": "Baseline\nprior",
    "random_search": "Random\nsearch",
    "pure_sa": "Pure SA",
    "enum_de": "Enum-DE",
    "hcmbo_proposed": "HCMBO",
    "tpe_mixed_bo": "TPE mixed\nBO",
}
METHOD_COLORS = {
    "baseline_prior_best": "#8C8C8C",
    "random_search": "#C44E52",
    "pure_sa": "#55A868",
    "enum_de": "#E69F00",
    "hcmbo_proposed": "#4C78A8",
    "tpe_mixed_bo": "#8172B3",
}
TERM_COLORS = {
    "J1": "#4C78A8",
    "J2": "#55A868",
    "J5": "#C44E52",
    "JB": "#E69F00",
    "0.1 JR": "#8172B3",
    "Residual": "#9D755D",
}
CHANNELS = ("top", "middle", "lower_middle", "bottom")
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
PAPER_DPI = 300


def build_g6_visual_report(
    output_root: Path,
    *,
    exclude_methods: tuple[str, ...] = ("tpe_mixed_bo",),
    top_n: int = 12,
    figure_dir_name: str = "paper_figures_no_tpe",
) -> dict[str, object]:
    output_root = output_root.resolve()
    seed_rows = _read_csv(output_root / "G6_seed_summary.csv")
    method_rows = _read_csv(output_root / "G6_method_summary.csv")
    candidate_rows = _read_csv(output_root / "G6_hf_candidates.csv")
    curve_rows = _read_csv(output_root / "G6_convergence_curves.csv")
    if not seed_rows:
        raise ValueError(f"No G6_seed_summary.csv rows found under {output_root}")

    excluded = tuple(item.strip() for item in exclude_methods if item.strip())
    seed_rows = _filter_methods(seed_rows, excluded)
    method_rows = _filter_methods(method_rows, excluded)
    candidate_rows = _filter_methods(candidate_rows, excluded)
    curve_rows = _filter_methods(curve_rows, excluded)
    if not seed_rows:
        raise ValueError("No seed rows remain after applying excluded methods")

    paper_dir = output_root / figure_dir_name
    paper_dir.mkdir(parents=True, exist_ok=True)
    best_rows = _best_seed_rows_by_method(seed_rows)
    top_rows = sorted(
        [row for row in candidate_rows if math.isfinite(_float(row.get("objective_value")))],
        key=lambda row: _float(row.get("objective_value")),
    )[: max(1, int(top_n))]

    outputs = {
        "paper_best_objective": str(_save_paper_best_objective(paper_dir / "g6_paper_best_objective.png", seed_rows)),
        "paper_seed_rank_heatmap": str(
            _save_paper_seed_rank_heatmap(paper_dir / "g6_paper_seed_rank_heatmap.png", seed_rows)
        ),
        "paper_objective_feasibility": str(
            _save_paper_objective_feasibility(paper_dir / "g6_paper_objective_feasibility.png", method_rows, seed_rows)
        ),
        "paper_best_terms": str(_save_paper_best_terms(paper_dir / "g6_paper_best_terms.png", best_rows)),
        "paper_convergence": str(_save_paper_convergence(paper_dir / "g6_paper_convergence.png", curve_rows)),
        "paper_control_profiles": str(
            _save_paper_control_profiles(
                paper_dir / "g6_paper_control_profiles.png",
                best_rows,
                output_root=output_root,
            )
        ),
        "paper_paired_delta": str(_save_paper_paired_delta(paper_dir / "g6_paper_paired_delta.png", seed_rows)),
        "paper_feasible_rate": str(
            _save_paper_feasible_rate(paper_dir / "g6_paper_feasible_rate.png", method_rows, seed_rows)
        ),
        "top_candidates_csv": str(_save_top_candidates_csv(output_root / "G6_top_candidates_no_tpe.csv", top_rows)),
    }
    outputs["visual_summary"] = str(
        _save_visual_summary(
            output_root / "G6_visual_summary_no_tpe.md",
            seed_rows=seed_rows,
            method_rows=method_rows,
            outputs=outputs,
            excluded=excluded,
            top_n=len(top_rows),
        )
    )
    manifest = {
        "output_root": str(output_root),
        "figure_dir": str(paper_dir),
        "excluded_methods": list(excluded),
        "method_count": len(_methods_present(seed_rows)),
        "seed_count": len({int(_float(row.get("seed"))) for row in seed_rows if math.isfinite(_float(row.get("seed")))}),
        "top_n": len(top_rows),
        "outputs": outputs,
    }
    manifest_path = output_root / "G6_visual_manifest_no_tpe.json"
    outputs["visual_manifest"] = str(manifest_path)
    manifest["outputs"] = outputs
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _save_paper_best_objective(path: Path, seed_rows: list[dict[str, object]]) -> Path:
    methods = _methods_present(seed_rows)
    means = [_mean(_seed_values(seed_rows, method)) for method in methods]
    labels = [_method_label(method) for method in methods]
    colors = [_method_color(method) for method in methods]
    y = np.arange(len(methods), dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), dpi=PAPER_DPI)
    ax.barh(y, means, height=0.58, color=colors, alpha=0.86)
    for index, method in enumerate(methods):
        values = np.array(_seed_values(seed_rows, method), dtype=float)
        jitter = np.linspace(-0.13, 0.13, values.size) if values.size > 1 else np.array([0.0])
        ax.scatter(values, np.full(values.size, y[index]) + jitter, s=28, color="#222222", alpha=0.75, zorder=3)
        ax.text(means[index], y[index] + 0.24, f"mean {means[index]:.3f}", va="center", fontsize=8.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Best high-fidelity objective, lower is better")
    ax.set_title("G6 best objective across seeds")
    ax.grid(axis="x", alpha=0.24)
    finite = [value for value in means if math.isfinite(value)]
    if finite:
        ax.set_xlim(0.0, max(finite) * 1.16)
    return _save_paper_figure(fig, path)


def _save_paper_seed_rank_heatmap(path: Path, seed_rows: list[dict[str, object]]) -> Path:
    methods = _methods_present(seed_rows)
    seeds = sorted({int(_float(row.get("seed"))) for row in seed_rows if math.isfinite(_float(row.get("seed")))})
    by_seed_method = {
        (int(_float(row.get("seed"))), str(row.get("method"))): _float(row.get("best_hf_objective_default"))
        for row in seed_rows
    }
    matrix = np.full((len(methods), len(seeds)), np.nan, dtype=float)
    objective_matrix = np.full_like(matrix, np.nan)
    for col, seed in enumerate(seeds):
        values = [(method, by_seed_method.get((seed, method), math.nan)) for method in methods]
        finite = sorted((item for item in values if math.isfinite(item[1])), key=lambda item: item[1])
        for rank, (method, value) in enumerate(finite, start=1):
            row = methods.index(method)
            matrix[row, col] = float(rank)
            objective_matrix[row, col] = value

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), dpi=PAPER_DPI)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu_r", vmin=1, vmax=max(len(methods), 1))
    ax.set_title("Per-seed method rank")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Method")
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([str(seed) for seed in seeds])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([_method_label(method).replace("\n", " ") for method in methods])
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if math.isfinite(matrix[y, x]):
                color = "white" if matrix[y, x] <= 2.0 else "#111111"
                ax.text(
                    x,
                    y,
                    f"#{int(matrix[y, x])}\n{objective_matrix[y, x]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color=color,
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Rank, lower is better")
    return _save_paper_figure(fig, path)


def _save_paper_objective_feasibility(
    path: Path,
    method_rows: list[dict[str, object]],
    seed_rows: list[dict[str, object]],
) -> Path:
    rows = _method_summary_rows(method_rows, seed_rows)
    fig, ax = plt.subplots(1, 1, figsize=(6.9, 4.6), dpi=PAPER_DPI)
    for row in rows:
        method = str(row.get("method"))
        mean_obj = _float(row.get("mean_best_hf_objective"))
        feasible_rate = _float(row.get("feasible_rate"))
        spread = max(_float(row.get("std_best_hf_objective")), 0.02)
        ax.scatter(
            [mean_obj],
            [feasible_rate],
            s=520 * spread + 80,
            color=_method_color(method),
            edgecolor="#222222",
            linewidth=0.7,
            alpha=0.86,
        )
        ax.annotate(
            _method_label(method).replace("\n", " "),
            xy=(mean_obj, feasible_rate),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8.7,
        )
    ax.set_title("Objective-feasibility trade-off")
    ax.set_xlabel("Mean best high-fidelity objective")
    ax.set_ylabel("Feasible seed rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.24)
    return _save_paper_figure(fig, path)


def _save_paper_best_terms(path: Path, best_rows: list[dict[str, object]]) -> Path:
    methods = [str(row.get("method")) for row in best_rows]
    labels = [_method_label(method) for method in methods]
    components = _objective_components(best_rows)
    x = np.arange(len(best_rows), dtype=float)
    bottoms = np.zeros(len(best_rows), dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8), dpi=PAPER_DPI)
    for term, values in components.items():
        values_arr = np.array(values, dtype=float)
        ax.bar(x, values_arr, bottom=bottoms, width=0.62, color=TERM_COLORS[term], label=term)
        bottoms += values_arr
    for xx, total, row in zip(x, bottoms, best_rows):
        ax.text(xx, total, f"{_float(row.get('best_hf_objective_default')):.3f}", ha="center", va="bottom", fontsize=8.0)
    ax.set_title("Best-case objective decomposition")
    ax.set_ylabel("Objective contribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False, ncols=3, fontsize=8.2)
    return _save_paper_figure(fig, path)


def _save_paper_convergence(path: Path, curve_rows: list[dict[str, object]]) -> Path:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in curve_rows:
        method = str(row.get("method"))
        seed_value = _float(row.get("seed"))
        if not math.isfinite(seed_value) or method == "baseline_prior_best":
            continue
        grouped[(method, int(seed_value))].append(row)
    methods = [method for method in _methods_present(curve_rows) if method != "baseline_prior_best"]
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.8), dpi=PAPER_DPI)
    for method in methods:
        seed_series = []
        max_eval = 0
        for (series_method, _seed), rows in grouped.items():
            if series_method != method:
                continue
            ordered = sorted(rows, key=lambda item: _float(item.get("evaluation_index")))
            pairs = [
                (int(_float(item.get("evaluation_index"))), _float(item.get("best_so_far")))
                for item in ordered
                if math.isfinite(_float(item.get("evaluation_index"))) and math.isfinite(_float(item.get("best_so_far")))
            ]
            if not pairs:
                continue
            max_eval = max(max_eval, pairs[-1][0])
            seed_series.append(pairs)
        if not seed_series:
            continue
        x_grid = np.arange(1, min(max_eval, 400) + 1, dtype=int)
        matrix = np.vstack([_forward_fill_series(series, x_grid) for series in seed_series])
        mean = np.nanmean(matrix, axis=0)
        q25 = np.nanpercentile(matrix, 25, axis=0)
        q75 = np.nanpercentile(matrix, 75, axis=0)
        color = _method_color(method)
        ax.plot(x_grid, mean, color=color, linewidth=1.9, label=_method_label(method).replace("\n", " "))
        ax.fill_between(x_grid, q25, q75, color=color, alpha=0.16, linewidth=0)
    ax.set_title("Best-so-far convergence")
    ax.set_xlabel("Optimization evaluations")
    ax.set_ylabel("Best objective so far")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, fontsize=8.4)
    return _save_paper_figure(fig, path)


def _save_paper_control_profiles(path: Path, best_rows: list[dict[str, object]], *, output_root: Path) -> Path:
    selected = [row for row in best_rows if str(row.get("method")) != "baseline_prior_best"]
    if not selected:
        return _save_empty_figure(path, "No non-baseline control profiles available")
    selected = _ordered_rows(selected)
    matrices: list[np.ndarray] = []
    titles: list[str] = []
    for row in selected:
        method = str(row.get("method"))
        seed = int(_float(row.get("seed")))
        control = _load_best_control(output_root, method, seed)
        matrices.append(_control_matrix(control))
        directions = control.get("directions", {}) if isinstance(control, dict) else {}
        if not isinstance(directions, dict):
            directions = {}
        direction_label = " ".join(f"{_channel_short(channel)}={directions.get(channel, '?')}" for channel in CHANNELS)
        titles.append(f"{_method_label(method).replace(chr(10), ' ')} | seed {seed} | {direction_label}")

    matrix = np.vstack(matrices)
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.nanpercentile(finite, 95)) if finite.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0

    ncols = 2 if len(selected) > 1 else 1
    nrows = int(math.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(9.6, max(4.2, 3.2 * nrows + 0.7)),
        dpi=PAPER_DPI,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.12, right=0.86, top=0.88, bottom=0.10, wspace=0.28, hspace=0.56)
    axes_array = axes.ravel()
    image = None
    gate_labels = [_gate_label(gate) for gate in GATE_ROWS]
    for index, (ax, title, method_matrix) in enumerate(zip(axes_array, titles, matrices)):
        image = ax.imshow(method_matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=9.8, loc="left")
        ax.set_yticks(np.arange(len(GATE_ROWS)))
        ax.set_yticklabels(gate_labels if index % ncols == 0 else [], fontsize=7.2)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["1", "2", "3", "4"])
        if index // ncols < nrows - 1:
            ax.set_xticklabels([])
        for y in range(method_matrix.shape[0]):
            for x in range(method_matrix.shape[1]):
                value = method_matrix[y, x]
                if math.isfinite(value) and value > 0.0:
                    color = "white" if value >= 0.62 * vmax else "#111111"
                    ax.text(x, y, f"{value:.2g}", ha="center", va="center", fontsize=7.0, color=color)
    for ax in axes_array[len(selected) :]:
        ax.set_axis_off()
    fig.suptitle("Best-solution capacity profiles", y=0.965, fontsize=12.0)
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Gate segment")
    if image is not None:
        cbar_ax = fig.add_axes([0.89, 0.17, 0.020, 0.64])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label("Capacity q")
    return _save_paper_figure(fig, path, tight=False)


def _save_paper_paired_delta(path: Path, seed_rows: list[dict[str, object]]) -> Path:
    hcmbo = {
        int(_float(row.get("seed"))): _float(row.get("best_hf_objective_default"))
        for row in seed_rows
        if str(row.get("method")) == "hcmbo_proposed" and math.isfinite(_float(row.get("seed")))
    }
    methods = [method for method in _methods_present(seed_rows) if method != "hcmbo_proposed"]
    if not hcmbo or not methods:
        return _save_empty_figure(path, "Paired delta requires HCMBO and comparator rows")
    data = []
    for method in methods:
        deltas = []
        for row in seed_rows:
            if str(row.get("method")) != method:
                continue
            seed = int(_float(row.get("seed")))
            if seed in hcmbo:
                deltas.append(_float(row.get("best_hf_objective_default")) - hcmbo[seed])
        data.append(deltas)

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.4), dpi=PAPER_DPI)
    positions = np.arange(len(methods), dtype=float)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    for pos, method, deltas in zip(positions, methods, data):
        values = np.array(deltas, dtype=float)
        if values.size == 0:
            continue
        jitter = np.linspace(-0.10, 0.10, values.size) if values.size > 1 else np.array([0.0])
        ax.scatter(np.full(values.size, pos) + jitter, values, s=34, color=_method_color(method), edgecolor="white", linewidth=0.4)
        mean = float(np.mean(values))
        ax.plot([pos - 0.22, pos + 0.22], [mean, mean], color="#111111", linewidth=1.5)
        ax.text(pos, mean, f" {mean:+.2f}", ha="left", va="center", fontsize=8.0)
    ax.set_title("Paired objective delta relative to HCMBO")
    ax.set_ylabel("Comparator objective - HCMBO objective")
    ax.set_xticks(positions)
    ax.set_xticklabels([_method_label(method) for method in methods])
    ax.grid(axis="y", alpha=0.24)
    return _save_paper_figure(fig, path)


def _save_paper_feasible_rate(
    path: Path,
    method_rows: list[dict[str, object]],
    seed_rows: list[dict[str, object]],
) -> Path:
    rows = _method_summary_rows(method_rows, seed_rows)
    labels = [_method_label(str(row.get("method"))) for row in rows]
    values = [_float(row.get("feasible_rate")) for row in rows]
    x = np.arange(len(rows), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(6.9, 4.0), dpi=PAPER_DPI)
    bars = ax.bar(x, values, color=[_method_color(str(row.get("method"))) for row in rows], width=0.62)
    ax.set_title("Feasible best-case rate")
    ax.set_ylabel("Feasible seeds / all seeds")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.24)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.0%}", ha="center", va="bottom", fontsize=8.4)
    return _save_paper_figure(fig, path)


def _save_top_candidates_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    fieldnames = [
        "rank",
        "method",
        "seed",
        "objective_value",
        "feasible",
        "case_id",
        "j1_eval",
        "j2_eval",
        "j5_eval",
        "jb_normalized",
        "jr_normalized",
        "gate_rejected",
    ] + [f"direction_{channel}" for channel in CHANNELS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow({field: index if field == "rank" else row.get(field, "") for field in fieldnames})
    return path


def _save_visual_summary(
    path: Path,
    *,
    seed_rows: list[dict[str, object]],
    method_rows: list[dict[str, object]],
    outputs: dict[str, str],
    excluded: tuple[str, ...],
    top_n: int,
) -> Path:
    summary_rows = _method_summary_rows(method_rows, seed_rows)
    best_mean = min(summary_rows, key=lambda row: _float(row.get("mean_best_hf_objective")))
    best_feasible = max(summary_rows, key=lambda row: _float(row.get("feasible_rate")))
    hcmbo = next((row for row in summary_rows if str(row.get("method")) == "hcmbo_proposed"), None)
    hcmbo_line = ""
    if hcmbo:
        hcmbo_line = (
            f"- HCMBO mean objective: {_float(hcmbo.get('mean_best_hf_objective')):.6f}; "
            f"feasible rate: {_float(hcmbo.get('feasible_rate')):.0%}."
        )
    lines = [
        "# G6 visual summary",
        "",
        f"- Result root: `{path.parent}`",
        f"- Excluded methods: `{', '.join(excluded) if excluded else 'none'}`",
        f"- Top candidate table size: {top_n}",
        hcmbo_line,
        f"- Lowest mean objective in this figure set: `{best_mean.get('method')}` "
        f"({_float(best_mean.get('mean_best_hf_objective')):.6f}).",
        f"- Highest feasible rate in this figure set: `{best_feasible.get('method')}` "
        f"({_float(best_feasible.get('feasible_rate')):.0%}).",
        "",
        "## Method summary",
        "",
        "| method | mean objective | std | best | feasible rate | best seed |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {mean:.6f} | {std:.6f} | {best:.6f} | {feasible:.0%} | {seed} |".format(
                method=row.get("method", ""),
                mean=_float(row.get("mean_best_hf_objective")),
                std=_float(row.get("std_best_hf_objective")),
                best=_float(row.get("best_hf_objective")),
                feasible=_float(row.get("feasible_rate")),
                seed=row.get("best_seed", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Figure purpose and conclusion",
            "",
            "| figure | purpose | conclusion supported by current G6 data |",
            "| --- | --- | --- |",
            "| `g6_paper_best_objective` | Compare final best objective across seeds. | With TPE excluded, HCMBO has the lowest mean objective among the retained methods. |",
            "| `g6_paper_seed_rank_heatmap` | Show seed-wise rank stability rather than only averages. | HCMBO is a stable top-tier method, while some seeds are competitive for pure SA and enum-DE. |",
            "| `g6_paper_objective_feasibility` | Jointly show scalar performance and feasibility. | HCMBO combines low objective with a high feasible-seed rate. |",
            "| `g6_paper_best_terms` | Explain which objective terms drive each method's best case. | Method differences are mainly visible through density and residual penalty terms, not only J1. |",
            "| `g6_paper_convergence` | Compare optimization progress under the same budget. | The curve separates final quality from search efficiency over the 400-evaluation budget. |",
            "| `g6_paper_control_profiles` | Show the actual best capacity-control structure. | HCMBO's best policy is interpretable as segment-wise gate capacity allocation plus channel directions. |",
            "| `g6_paper_paired_delta` | Use paired seeds to quantify improvement relative to HCMBO. | Positive deltas indicate seeds where HCMBO improves over the comparator. |",
            "| `g6_paper_feasible_rate` | Present feasibility as a standalone robustness metric. | HCMBO and the strongest retained baselines are separable by feasible best-case rate. |",
            "",
            "## Generated outputs",
            "",
        ]
    )
    for name, output_path in outputs.items():
        lines.append(f"- `{name}`: `{output_path}`")
    lines.append("")
    lines.append("Each paper figure is exported as same-name PNG and PDF.")
    path.write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            rows.append(row)
    return rows


def _filter_methods(rows: list[dict[str, object]], excluded: tuple[str, ...]) -> list[dict[str, object]]:
    if not excluded:
        return rows
    excluded_set = set(excluded)
    return [row for row in rows if str(row.get("method")) not in excluded_set]


def _method_summary_rows(method_rows: list[dict[str, object]], seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if method_rows:
        return _ordered_rows(method_rows)
    rows: list[dict[str, object]] = []
    for method in _methods_present(seed_rows):
        values = _seed_values(seed_rows, method)
        method_seed_rows = [row for row in seed_rows if str(row.get("method")) == method]
        best_row = min(method_seed_rows, key=lambda row: _float(row.get("best_hf_objective_default")))
        feasible = [str(row.get("feasible_best")).lower() == "true" for row in method_seed_rows]
        arr = np.array(values, dtype=float)
        rows.append(
            {
                "method": method,
                "seed_count": len(values),
                "mean_best_hf_objective": float(np.mean(arr)),
                "std_best_hf_objective": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "median_best_hf_objective": float(np.median(arr)),
                "iqr_best_hf_objective": float(np.percentile(arr, 75) - np.percentile(arr, 25)) if len(arr) > 1 else 0.0,
                "best_hf_objective": float(np.min(arr)),
                "worst_hf_objective": float(np.max(arr)),
                "feasible_rate": sum(1 for item in feasible if item) / max(len(feasible), 1),
                "best_seed": best_row.get("seed"),
                "best_case_id": best_row.get("best_case_id"),
            }
        )
    return rows


def _best_seed_rows_by_method(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in _methods_present(seed_rows):
        method_rows = [row for row in seed_rows if str(row.get("method")) == method]
        rows.append(min(method_rows, key=lambda row: _float(row.get("best_hf_objective_default"))))
    return rows


def _methods_present(rows: list[dict[str, object]]) -> list[str]:
    present = {str(row.get("method")) for row in rows if str(row.get("method"))}
    ordered = [method for method in METHOD_ORDER if method in present]
    return ordered + sorted(present.difference(ordered))


def _ordered_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_method = {str(row.get("method")): row for row in rows}
    ordered = [by_method[method] for method in METHOD_ORDER if method in by_method]
    extras = [row for row in rows if str(row.get("method")) not in METHOD_ORDER]
    return ordered + extras


def _seed_values(seed_rows: list[dict[str, object]], method: str) -> list[float]:
    return [
        _float(row.get("best_hf_objective_default"))
        for row in seed_rows
        if str(row.get("method")) == method and math.isfinite(_float(row.get("best_hf_objective_default")))
    ]


def _objective_components(rows: list[dict[str, object]]) -> dict[str, list[float]]:
    components: dict[str, list[float]] = {
        "J1": [],
        "J2": [],
        "J5": [],
        "JB": [],
        "0.1 JR": [],
        "Residual": [],
    }
    for row in rows:
        j1 = _float(row.get("J1_eval"))
        j2 = _float(row.get("J2_eval"))
        j5 = _float(row.get("J5_eval"))
        jb = _float(row.get("JB_normalized"))
        jr = 0.1 * _float(row.get("JR_normalized"))
        objective = _float(row.get("best_hf_objective_default"))
        subtotal = sum(value for value in (j1, j2, j5, jb, jr) if math.isfinite(value))
        residual = max(0.0, objective - subtotal) if math.isfinite(objective) else 0.0
        components["J1"].append(max(0.0, j1))
        components["J2"].append(max(0.0, j2))
        components["J5"].append(max(0.0, j5))
        components["JB"].append(max(0.0, jb))
        components["0.1 JR"].append(max(0.0, jr))
        components["Residual"].append(residual)
    if max(components["Residual"] or [0.0]) <= 1.0e-8:
        components.pop("Residual")
    return components


def _forward_fill_series(series: list[tuple[int, float]], x_grid: np.ndarray) -> np.ndarray:
    result = np.full(x_grid.shape, np.nan, dtype=float)
    cursor = 0
    current = math.nan
    for index, x_value in enumerate(x_grid):
        while cursor < len(series) and series[cursor][0] <= int(x_value):
            current = series[cursor][1]
            cursor += 1
        result[index] = current
    return result


def _load_best_control(output_root: Path, method: str, seed: int) -> dict[str, object]:
    path = output_root / method / f"seed_{seed}" / "G6_best_control.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _control_matrix(control: dict[str, object]) -> np.ndarray:
    q_by_gate = control.get("q_by_gate", {}) if isinstance(control, dict) else {}
    if not isinstance(q_by_gate, dict):
        q_by_gate = {}
    rows = []
    for gate in GATE_ROWS:
        raw_values = q_by_gate.get(gate, [])
        values = [_float(item) for item in raw_values] if isinstance(raw_values, list) else []
        if len(values) < 4:
            values = values + [math.nan] * (4 - len(values))
        rows.append(values[:4])
    return np.array(rows, dtype=float)


def _save_empty_figure(path: Path, message: str) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8), dpi=PAPER_DPI)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10.0)
    ax.set_axis_off()
    return _save_paper_figure(fig, path)


def _save_paper_figure(fig: object, path: Path, *, tight: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=PAPER_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#777777")


def _channel_short(channel: str) -> str:
    if channel == "lower_middle":
        return "LM"
    return channel[:1].upper()


def _gate_label(gate: str) -> str:
    return gate.replace("lower_middle", "lower-mid").replace(":", " ")


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return float(np.mean(np.array(finite, dtype=float)))


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
