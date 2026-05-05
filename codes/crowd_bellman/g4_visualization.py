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
    "SA-HBO": "#4C78A8",
    "grid_search": "#F58518",
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


def build_g4_visual_report(output_root: Path, *, top_n: int = 12) -> dict[str, object]:
    output_root = output_root.resolve()
    rows = _read_evaluation_rows(output_root / "g4_evaluation_log.csv")
    method_rows = _read_method_rows(output_root / "g4_method_comparison.csv")
    summary = _read_summary(output_root / "g4_sahbo_grid_summary.json")
    if not rows:
        raise ValueError(f"No G4 evaluation rows found under {output_root}")

    top_rows = sorted(rows, key=lambda row: _float(row["objective_value"]))[: max(1, int(top_n))]
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
            row["method"] = _method_for_source(str(row.get("source", "")))
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


def _save_objective_trace(path: Path, rows: list[dict[str, object]], method_rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.4), dpi=160)
    rows_by_method = {
        method: [row for row in rows if row.get("method") == method]
        for method in ("SA-HBO", "grid_search")
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
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


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
    for method in ("SA-HBO", "grid_search"):
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
            "- `g4_top_candidates.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


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


def _method_for_source(source: str) -> str:
    return "grid_search" if source == "grid" else "SA-HBO"


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
