from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "baseline": "#7F7F7F",
    "random_search": "#B279A2",
    "pure_sa": "#54A24B",
    "SA-HBO w/o proxy": "#E45756",
    "SA-HBO": "#4C78A8",
    "grid_search": "#F58518",
}
METHOD_ORDER = ("baseline", "random_search", "grid_search", "pure_sa", "SA-HBO w/o proxy", "SA-HBO")
METHOD_LABELS = {
    "baseline": "Baseline",
    "random_search": "Random\nsearch",
    "grid_search": "Grid\nsearch",
    "pure_sa": "Pure SA",
    "SA-HBO w/o proxy": "SA-HBO\nw/o proxy",
    "SA-HBO": "SA-HBO",
}
TERM_COLORS = {
    "j1": "#4C78A8",
    "j2": "#54A24B",
    "j5": "#E45756",
}
CHANNEL_COLORS = {
    "top": "#4C78A8",
    "middle": "#F58518",
    "lower_middle": "#54A24B",
    "bottom": "#B279A2",
}
CHANNELS = ("top", "middle", "lower_middle", "bottom")
TERMS = ("j1", "j2", "j5")
PAPER_DPI = 300


def build_g4_visual_report(output_root: Path, *, top_n: int = 12) -> dict[str, object]:
    output_root = output_root.resolve()
    rows = _read_evaluation_rows(output_root / "g4_evaluation_log.csv")
    method_rows = _read_method_rows(output_root / "g4_method_comparison.csv")
    summary = _read_summary(output_root / "g4_sahbo_grid_summary.json")
    if not rows:
        raise ValueError(f"No G4 evaluation rows found under {output_root}")

    top_rows = sorted(rows, key=lambda row: _float(row["objective_value"]))[: max(1, int(top_n))]
    paper_outputs = _save_paper_figures(output_root=output_root, rows=rows, method_rows=method_rows)
    outputs = {
        "objective_trace": str(_save_objective_trace(output_root / "g4_objective_trace.png", rows, method_rows)),
        "method_comparison": str(_save_method_comparison(output_root / "g4_method_comparison.png", method_rows)),
        "top_candidates_terms": str(
            _save_top_candidate_terms(output_root / "g4_top_candidates_objective_terms.png", top_rows)
        ),
        "pareto_j1_j2": str(_save_pareto_plot(output_root / "g4_pareto_j1_j2.png", rows, method_rows)),
        "best_channel_flux_share": str(
            _save_best_channel_flux(output_root / "g4_best_channel_flux_share.png", rows, method_rows)
        ),
        **paper_outputs,
        "top_candidates_csv": str(_save_top_candidates_csv(output_root / "g4_top_candidates.csv", top_rows)),
        "visual_summary": str(_save_markdown_summary(output_root / "g4_visual_summary.md", rows, method_rows, summary)),
    }
    return {
        "output_root": str(output_root),
        "evaluation_count": len(rows),
        "top_n": len(top_rows),
        "outputs": outputs,
    }


def _read_evaluation_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            numeric_keys = (
                "eval_id",
                "objective_value",
                "j1",
                "j2",
                "j5",
                "j1_raw",
                "j2_raw",
                "j5_raw",
            ) + tuple(
                item
                for channel in CHANNELS
                for item in (f"eta_{channel}", f"flux_{channel}", f"flux_share_{channel}")
            )
            for key in numeric_keys:
                if key in row:
                    row[key] = _number(row[key])
            row["method"] = _method_for_row(row)
            rows.append(row)
    return rows


def _read_method_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for key in ("objective_value", "evaluation_count"):
                if key in row:
                    row[key] = _number(row[key])
            rows.append(row)
    return rows


def _read_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_paper_figures(
    *,
    output_root: Path,
    rows: list[dict[str, object]],
    method_rows: list[dict[str, object]],
) -> dict[str, str]:
    paper_dir = output_root / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    best_rows = _best_rows_by_method(rows, method_rows)
    outputs = {
        "paper_best_objective": _save_paper_best_objective(paper_dir / "g4_paper_best_objective.png", method_rows),
        "paper_best_so_far": _save_paper_best_so_far(paper_dir / "g4_paper_best_so_far.png", rows),
        "paper_objective_distribution": _save_paper_objective_distribution(
            paper_dir / "g4_paper_objective_distribution.png", rows
        ),
        "paper_best_terms": _save_paper_best_terms(paper_dir / "g4_paper_best_terms.png", best_rows),
        "paper_direction_matrix": _save_paper_direction_matrix(
            paper_dir / "g4_paper_direction_matrix.png", best_rows
        ),
        "paper_eta_heatmap": _save_paper_eta_heatmap(paper_dir / "g4_paper_eta_heatmap.png", best_rows),
        "paper_flux_heatmap": _save_paper_flux_heatmap(paper_dir / "g4_paper_flux_heatmap.png", best_rows),
    }
    return {name: str(path) for name, path in outputs.items()}


def _save_paper_best_objective(path: Path, method_rows: list[dict[str, object]]) -> Path:
    ordered = _ordered_method_rows(method_rows)
    labels = [_method_label(str(row.get("method", ""))) for row in ordered]
    values = [_float(row.get("objective_value", math.nan)) for row in ordered]
    counts = [_float(row.get("evaluation_count", math.nan)) for row in ordered]
    colors = [METHOD_COLORS.get(str(row.get("method", "")), "#777777") for row in ordered]

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.2), dpi=PAPER_DPI)
    y = np.arange(len(ordered), dtype=float)
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Best scalar objective")
    ax.set_title("G4 best objective by method")
    ax.grid(axis="x", alpha=0.25)
    for yy, value, count in zip(y, values, counts):
        ax.text(value, yy, f"  {value:.4f} ({count:.0f})", va="center", fontsize=8.5)
    ax.set_xlim(0.0, max(values) * 1.22 if values else 1.0)
    return _save_paper_figure(fig, path)


def _save_paper_best_so_far(path: Path, rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8), dpi=PAPER_DPI)
    for method in METHOD_ORDER:
        method_rows = _rows_for_method(rows, method)
        if not method_rows:
            continue
        xs = np.arange(1, len(method_rows) + 1, dtype=float)
        values = np.array([_float(row["objective_value"]) for row in method_rows], dtype=float)
        best_so_far = np.minimum.accumulate(values)
        ax.step(
            xs,
            best_so_far,
            where="post",
            linewidth=1.8,
            color=METHOD_COLORS.get(method, "#777777"),
            label=_method_label(method).replace("\n", " "),
        )
        ax.scatter([xs[-1]], [best_so_far[-1]], s=24, color=METHOD_COLORS.get(method, "#777777"), zorder=3)
    ax.set_title("G4 best-so-far convergence")
    ax.set_xlabel("Evaluations within method")
    ax.set_ylabel("Best objective so far")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncols=2, fontsize=8.5)
    return _save_paper_figure(fig, path)


def _save_paper_objective_distribution(path: Path, rows: list[dict[str, object]]) -> Path:
    methods = [method for method in METHOD_ORDER if _rows_for_method(rows, method)]
    data = [
        [_float(row["objective_value"]) for row in _rows_for_method(rows, method)]
        for method in methods
    ]
    labels = [_method_label(method) for method in methods]
    colors = [METHOD_COLORS.get(method, "#777777") for method in methods]

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.8), dpi=PAPER_DPI)
    box = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.54, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.4)
    for index, values in enumerate(data, start=1):
        jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), index, dtype=float) + jitter,
            values,
            s=18,
            color=colors[index - 1],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.35,
        )
    ax.set_title("G4 objective distribution")
    ax.set_ylabel("Scalar objective")
    ax.grid(axis="y", alpha=0.25)
    return _save_paper_figure(fig, path)


def _save_paper_best_terms(path: Path, best_rows: list[dict[str, object]]) -> Path:
    labels = [_method_label(str(row.get("method", ""))) for row in best_rows]
    x = np.arange(len(best_rows), dtype=float)
    bottoms = np.zeros(len(best_rows), dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.7), dpi=PAPER_DPI)
    for term in TERMS:
        values = np.array([_float(row.get(term, 0.0)) for row in best_rows], dtype=float)
        ax.bar(x, values, bottom=bottoms, color=TERM_COLORS[term], width=0.62, label=term.upper())
        bottoms += values
    for xx, total in zip(x, bottoms):
        ax.text(xx, total, f"{total:.3f}", ha="center", va="bottom", fontsize=8.0)
    ax.set_title("Best-solution objective decomposition")
    ax.set_ylabel("Normalized objective terms")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    return _save_paper_figure(fig, path)


def _save_paper_direction_matrix(path: Path, best_rows: list[dict[str, object]]) -> Path:
    state_to_value = {"CLOSED": 0, "W": 1, "FREE": 2, "E": 3}
    matrix = np.array(
        [
            [state_to_value.get(str(row.get(f"direction_{channel}", "")).upper(), np.nan) for channel in CHANNELS]
            for row in best_rows
        ],
        dtype=float,
    )
    colors = ["#BFBFBF", "#5B8FF9", "#A0A0A0", "#F6BD16"]
    cmap = matplotlib.colors.ListedColormap(colors)
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.9), dpi=PAPER_DPI)
    im = ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_title("Best-solution channel direction states")
    ax.set_xticks(np.arange(len(CHANNELS)))
    ax.set_xticklabels(_channel_labels())
    ax.set_yticks(np.arange(len(best_rows)))
    ax.set_yticklabels([_method_label(str(row.get("method", ""))).replace("\n", " ") for row in best_rows])
    for y, row in enumerate(best_rows):
        for x, channel in enumerate(CHANNELS):
            state = str(row.get(f"direction_{channel}", ""))
            ax.text(x, y, _state_symbol(state), ha="center", va="center", color="#111111", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["Closed", "W", "Free", "E"])
    return _save_paper_figure(fig, path)


def _save_paper_eta_heatmap(path: Path, best_rows: list[dict[str, object]]) -> Path:
    matrix = np.array(
        [
            [_float(row.get(f"eta_{channel}", math.nan)) for channel in CHANNELS]
            for row in best_rows
        ],
        dtype=float,
    )
    return _save_annotated_heatmap(
        path=path,
        matrix=matrix,
        row_labels=[_method_label(str(row.get("method", ""))).replace("\n", " ") for row in best_rows],
        col_labels=_channel_labels(),
        title="Best-solution guidance intensity eta",
        colorbar_label="eta",
        fmt="{:.2f}",
        cmap="YlGnBu",
    )


def _save_paper_flux_heatmap(path: Path, best_rows: list[dict[str, object]]) -> Path:
    matrix = np.array(
        [
            [_float(row.get(f"flux_share_{channel}", math.nan)) for channel in CHANNELS]
            for row in best_rows
        ],
        dtype=float,
    )
    return _save_annotated_heatmap(
        path=path,
        matrix=matrix,
        row_labels=[_method_label(str(row.get("method", ""))).replace("\n", " ") for row in best_rows],
        col_labels=_channel_labels(),
        title="Best-solution channel flux share",
        colorbar_label="flux share",
        fmt="{:.1%}",
        cmap="PuBuGn",
    )


def _save_annotated_heatmap(
    *,
    path: Path,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    fmt: str,
    cmap: str,
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.9), dpi=PAPER_DPI)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    finite = matrix[np.isfinite(matrix)]
    threshold = float(np.nanmean(finite)) if finite.size else 0.0
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            if not np.isfinite(value):
                label = ""
            else:
                label = fmt.format(value)
            color = "white" if np.isfinite(value) and value > threshold else "#111111"
            ax.text(x, y, label, ha="center", va="center", color=color, fontsize=8.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(colorbar_label)
    return _save_paper_figure(fig, path)


def _save_objective_trace(path: Path, rows: list[dict[str, object]], method_rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.4), dpi=160)
    rows_by_method = {
        method: [row for row in rows if row.get("method") == method]
        for method in METHOD_ORDER
    }
    for method, method_data in rows_by_method.items():
        if not method_data:
            continue
        xs = [_float(row["eval_id"]) for row in method_data]
        ys = [_float(row["objective_value"]) for row in method_data]
        ax.plot(xs, ys, "o-", linewidth=1.4, markersize=4.2, color=METHOD_COLORS[method], label=method)

    for item in method_rows:
        best = _find_row_by_case(rows, str(item.get("case_id", "")))
        if best is None:
            continue
        method = str(item.get("method", ""))
        ax.scatter(
            [_float(best["eval_id"])],
            [_float(best["objective_value"])],
            s=95,
            marker="*",
            color=METHOD_COLORS.get(method, "#333333"),
            edgecolor="#222222",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"{method} best",
            xy=(_float(best["eval_id"]), _float(best["objective_value"])),
            xytext=(6, -16 if method == "grid_search" else 10),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.set_title("G4 objective trace")
    ax.set_xlabel("Evaluation id")
    ax.set_ylabel("Objective value")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    return _save_paper_figure(fig, path)


def _save_method_comparison(path: Path, method_rows: list[dict[str, object]]) -> Path:
    if not method_rows:
        raise ValueError("Missing g4_method_comparison.csv rows")
    labels = [str(row["method"]) for row in method_rows]
    values = [_float(row["objective_value"]) for row in method_rows]
    counts = [_float(row.get("evaluation_count", math.nan)) for row in method_rows]
    colors = [METHOD_COLORS.get(label, "#777777") for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=160)
    bars = axes[0].bar(labels, values, color=colors, width=0.55)
    axes[0].set_title("Best objective")
    axes[0].set_ylabel("Objective value")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    bars = axes[1].bar(labels, counts, color=colors, width=0.55)
    axes[1].set_title("Evaluation budget used")
    axes[1].set_ylabel("Evaluations")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_top_candidate_terms(path: Path, rows: list[dict[str, object]]) -> Path:
    labels = [_short_candidate_label(row) for row in rows]
    x = np.arange(len(rows), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(1, 1, figsize=(max(9.5, 0.65 * len(rows)), 5.6), dpi=160)
    for offset, term in zip((-width, 0.0, width), TERMS):
        values = [_float(row.get(term, math.nan)) for row in rows]
        ax.bar(x + offset, values, width=width, color=TERM_COLORS[term], label=term)
    ax.plot(x, [_float(row["objective_value"]) for row in rows], "k.-", linewidth=1.0, markersize=4.5, label="objective")
    ax.set_title(f"Top {len(rows)} candidate objective terms")
    ax.set_ylabel("Normalized value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_pareto_plot(path: Path, rows: list[dict[str, object]], method_rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.8), dpi=160)
    for method in METHOD_ORDER:
        method_data = [row for row in rows if row.get("method") == method]
        if not method_data:
            continue
        xs = [_float(row["j1"]) for row in method_data]
        ys = [_float(row["j2"]) for row in method_data]
        sizes = [34 + 420 * max(_float(row.get("j5", 0.0)), 0.0) for row in method_data]
        ax.scatter(xs, ys, s=sizes, alpha=0.72, color=METHOD_COLORS[method], label=method, edgecolor="white", linewidth=0.5)

    for item in method_rows:
        best = _find_row_by_case(rows, str(item.get("case_id", "")))
        if best is None:
            continue
        method = str(item.get("method", ""))
        ax.scatter(
            [_float(best["j1"])],
            [_float(best["j2"])],
            s=145,
            marker="*",
            color=METHOD_COLORS.get(method, "#333333"),
            edgecolor="#222222",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(method, xy=(_float(best["j1"]), _float(best["j2"])), xytext=(6, 6), textcoords="offset points", fontsize=8.5)

    ax.set_title("J1-J2 tradeoff, marker size by J5")
    ax.set_xlabel("J1 normalized travel time")
    ax.set_ylabel("J2 normalized density exposure")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_best_channel_flux(path: Path, rows: list[dict[str, object]], method_rows: list[dict[str, object]]) -> Path:
    selected = []
    for item in method_rows:
        best = _find_row_by_case(rows, str(item.get("case_id", "")))
        if best is not None:
            selected.append((str(item.get("method", "")), best))
    if not selected:
        raise ValueError("No best cases can be matched to evaluation rows")

    x = np.arange(len(selected), dtype=float)
    width = 0.8 / max(1, len(CHANNELS))
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.8), dpi=160)
    for idx, channel in enumerate(CHANNELS):
        offset = (idx - (len(CHANNELS) - 1) / 2.0) * width
        values = [_float(row.get(f"flux_share_{channel}", math.nan)) for _, row in selected]
        ax.bar(x + offset, values, width=width, color=CHANNEL_COLORS[channel], label=channel)
    ax.set_title("Best-case channel flux share")
    ax.set_ylabel("Flux share")
    ax.set_xticks(x)
    ax.set_xticklabels([method for method, _ in selected])
    ax.set_ylim(0.0, max(0.62, max(_float(row.get(f"flux_share_{channel}", 0.0)) for _, row in selected for channel in CHANNELS) * 1.16))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=len(CHANNELS))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_top_candidates_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    fields = [
        "rank",
        "method",
        "eval_id",
        "source",
        "case_id",
        "objective_value",
        "j1",
        "j2",
        "j5",
    ] + [f"direction_{channel}" for channel in CHANNELS]
    fields += [f"eta_{channel}" for channel in CHANNELS]
    fields += [f"flux_share_{channel}" for channel in CHANNELS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow({field: index if field == "rank" else row.get(field, "") for field in fields})
    return path


def _save_markdown_summary(
    path: Path,
    rows: list[dict[str, object]],
    method_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> Path:
    best_overall = min(rows, key=lambda row: _float(row["objective_value"]))
    lines = [
        "# G4 visual summary",
        "",
        f"- Result root: `{path.parent}`",
        f"- Total evaluations: {len(rows)}",
        f"- Design version: `{summary.get('design_version', 'unknown')}`",
        "",
        "## Method comparison",
        "",
        "| method | best objective | evals | case | directions | eta |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in method_rows:
        lines.append(
            "| {method} | {objective:.6f} | {evals:.0f} | `{case}` | {directions} | {eta} |".format(
                method=row.get("method", ""),
                objective=_float(row.get("objective_value", math.nan)),
                evals=_float(row.get("evaluation_count", math.nan)),
                case=row.get("case_id", ""),
                directions=row.get("directions", ""),
                eta=row.get("eta", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Best overall",
            "",
            f"- Case: `{best_overall.get('case_id', '')}`",
            f"- Method: {best_overall.get('method', '')}",
            f"- Objective: {_float(best_overall.get('objective_value', math.nan)):.6f}",
            f"- Terms: J1={_float(best_overall.get('j1', math.nan)):.6f}, "
            f"J2={_float(best_overall.get('j2', math.nan)):.6f}, "
            f"J5={_float(best_overall.get('j5', math.nan)):.6f}",
            "- Directions: "
            + ", ".join(f"{channel}={best_overall.get(f'direction_{channel}', '')}" for channel in CHANNELS),
            "- Eta: "
            + ", ".join(f"{channel}={_float(best_overall.get(f'eta_{channel}', math.nan)):.6g}" for channel in CHANNELS),
            "",
            "## Generated figures",
            "",
            "- `g4_objective_trace.png`",
            "- `g4_method_comparison.png`",
            "- `g4_top_candidates_objective_terms.png`",
            "- `g4_pareto_j1_j2.png`",
            "- `g4_best_channel_flux_share.png`",
            "- `paper_figures/g4_paper_best_objective.png`",
            "- `paper_figures/g4_paper_best_so_far.png`",
            "- `paper_figures/g4_paper_objective_distribution.png`",
            "- `paper_figures/g4_paper_best_terms.png`",
            "- `paper_figures/g4_paper_direction_matrix.png`",
            "- `paper_figures/g4_paper_eta_heatmap.png`",
            "- `paper_figures/g4_paper_flux_heatmap.png`",
            "- `g4_top_candidates.csv`",
            "",
            "Paper figures are also exported as same-name PDF files under `paper_figures/`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _save_paper_figure(fig: object, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=PAPER_DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def _ordered_method_rows(method_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_method = {str(row.get("method", "")): row for row in method_rows}
    ordered = [by_method[method] for method in METHOD_ORDER if method in by_method]
    extras = [row for row in method_rows if str(row.get("method", "")) not in METHOD_ORDER]
    return ordered + extras


def _best_rows_by_method(
    rows: list[dict[str, object]],
    method_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    best_rows: list[dict[str, object]] = []
    for method_row in _ordered_method_rows(method_rows):
        case_id = str(method_row.get("case_id", ""))
        best = _find_row_by_case(rows, case_id)
        if best is not None:
            best_rows.append(best)
    return best_rows


def _rows_for_method(rows: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    method_rows = [row for row in rows if str(row.get("method", "")) == method]
    return sorted(method_rows, key=lambda row: _float(row.get("method_eval_id", row.get("eval_id", math.inf))))


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _channel_labels() -> list[str]:
    return ["Top", "Middle", "Lower-mid", "Bottom"]


def _state_symbol(state: str) -> str:
    normalized = state.upper()
    if normalized == "CLOSED":
        return "X"
    if normalized == "FREE":
        return "F"
    if normalized == "E":
        return "E"
    if normalized == "W":
        return "W"
    return "?"


def _find_row_by_case(rows: list[dict[str, object]], case_id: str) -> dict[str, object] | None:
    for row in rows:
        if str(row.get("case_id", "")) == case_id:
            return row
    return None


def _short_candidate_label(row: dict[str, object]) -> str:
    directions = "".join(str(row.get(f"direction_{channel}", "?"))[:1] for channel in CHANNELS)
    eta_values = [_float(row.get(f"eta_{channel}", math.nan)) for channel in CHANNELS]
    if max(eta_values) - min(eta_values) <= 1.0e-8:
        eta_label = f"eta={eta_values[0]:.3g}"
    else:
        eta_label = "/".join(f"{value:.2g}" for value in eta_values)
    return f"{int(_float(row['eval_id']))}:{row.get('method', '')}\n{directions} {eta_label}"


def _method_for_row(row: dict[str, object]) -> str:
    method_key = str(row.get("method_key", "")).strip().lower()
    if method_key == "baseline":
        return "baseline"
    if method_key == "random_search":
        return "random_search"
    if method_key == "grid":
        return "grid_search"
    if method_key == "pure_sa":
        return "pure_sa"
    if method_key == "sahbo_no_proxy":
        return "SA-HBO w/o proxy"
    if method_key == "sahbo":
        return "SA-HBO"
    return _method_for_source(str(row.get("source", "")))


def _method_for_source(source: str) -> str:
    normalized = source.lower()
    if normalized == "baseline":
        return "baseline"
    if normalized.startswith("random_search"):
        return "random_search"
    if normalized == "grid":
        return "grid_search"
    if normalized.startswith("pure_sa"):
        return "pure_sa"
    if normalized.startswith("sahbo_no_proxy"):
        return "SA-HBO w/o proxy"
    return "SA-HBO"


def _number(value: object) -> object:
    text = str(value).strip()
    if text == "":
        return math.nan
    try:
        numeric = float(text)
    except ValueError:
        return value
    if numeric.is_integer() and not any(marker in text.lower() for marker in (".", "e")):
        return int(numeric)
    return numeric


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
