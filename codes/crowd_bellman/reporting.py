from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_timeseries(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {field: [] for field in reader.fieldnames or []}
        for row in reader:
            for field, value in row.items():
                columns[field].append(float(value))
    return columns


def _summary_rows(comparison: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in comparison["cases"]:
        channel_density = case["channel_time_mean_density"]
        channel_share = case["channel_flux_share"]
        rows.append(
            {
                "case_id": case["case_id"],
                "title": case["title"],
                "final_time": case["final_time"],
                "final_sink_cumulative": case["final_sink_cumulative"],
                "mean_density_avg": case["mean_density_avg"],
                "peak_density_max": case["peak_density_max"],
                "channel_mean_density_top": channel_density["top"],
                "channel_mean_density_middle": channel_density["middle"],
                "channel_mean_density_bottom": channel_density["bottom"],
                "channel_flux_share_top": channel_share["top"],
                "channel_flux_share_middle": channel_share["middle"],
                "channel_flux_share_bottom": channel_share["bottom"],
                "velocity_discontinuity_avg": case["velocity_discontinuity_avg"],
                "density_gradient_avg": case["density_gradient_avg"],
            }
        )
    return rows


def save_section_5_1_tables(output_root: Path, comparison: dict[str, object]) -> None:
    rows = _summary_rows(comparison)
    csv_path = output_root / "section_5_1_summary.csv"
    md_path = output_root / "section_5_1_summary.md"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "Case",
        "Final Time",
        "Sink Throughput",
        "Mean Density",
        "Peak Density",
        "Top Density",
        "Middle Density",
        "Bottom Density",
        "Top Share",
        "Middle Share",
        "Bottom Share",
        "Vel. Disc.",
        "Density Grad.",
    ]
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Section 5.1 Summary\n\n")
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row["case_id"]),
                        f'{row["final_time"]:.3f}',
                        f'{row["final_sink_cumulative"]:.3f}',
                        f'{row["mean_density_avg"]:.3f}',
                        f'{row["peak_density_max"]:.3f}',
                        f'{row["channel_mean_density_top"]:.3f}',
                        f'{row["channel_mean_density_middle"]:.3f}',
                        f'{row["channel_mean_density_bottom"]:.3f}',
                        f'{row["channel_flux_share_top"]:.3%}',
                        f'{row["channel_flux_share_middle"]:.3%}',
                        f'{row["channel_flux_share_bottom"]:.3%}',
                        f'{row["velocity_discontinuity_avg"]:.3f}',
                        f'{row["density_gradient_avg"]:.3f}',
                    ]
                )
                + " |\n"
            )


def save_section_5_1_timeseries_plot(output_root: Path, comparison: dict[str, object]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.suptitle("Section 5.1 Time-Series Comparison")

    for case in comparison["cases"]:
        case_id = str(case["case_id"])
        timeseries = _load_timeseries(output_root / case_id / "timeseries.csv")
        time = np.array(timeseries["time"])

        axes[0, 0].plot(time, timeseries["mean_density"], label=case_id)
        axes[0, 1].plot(time, timeseries["peak_density"], label=case_id)
        axes[1, 0].plot(time, timeseries["sink_cumulative"], label=case_id)
        axes[1, 1].plot(time, timeseries["velocity_discontinuity"], label=f"{case_id}: vel")
        axes[1, 1].plot(time, timeseries["density_gradient_intensity"], "--", label=f"{case_id}: grad")

    axes[0, 0].set_title("Global mean density")
    axes[0, 1].set_title("Peak density")
    axes[1, 0].set_title("Cumulative sink throughput")
    axes[1, 1].set_title("Stability metrics")

    for ax in axes.flat:
        ax.set_xlabel("time")
        ax.grid(alpha=0.2)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_root / "section_5_1_timeseries.png")
    plt.close(fig)


def save_section_5_1_channel_plot(output_root: Path, comparison: dict[str, object]) -> None:
    cases = [str(case["case_id"]) for case in comparison["cases"]]
    channels = ["top", "middle", "bottom"]
    x = np.arange(len(cases))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle("Section 5.1 Channel Statistics")

    for idx, channel in enumerate(channels):
        offset = (idx - 1) * width
        density_values = [
            float(case["channel_time_mean_density"][channel])
            for case in comparison["cases"]
        ]
        share_values = [
            float(case["channel_flux_share"][channel])
            for case in comparison["cases"]
        ]
        axes[0].bar(x + offset, density_values, width=width, label=channel)
        axes[1].bar(x + offset, share_values, width=width, label=channel)

    axes[0].set_title("Channel time-mean density")
    axes[1].set_title("Channel throughput share")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(cases)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_root / "section_5_1_channels.png")
    plt.close(fig)


def generate_section_5_1_report(output_root: Path) -> None:
    comparison = _load_json(output_root / "comparison_summary.json")
    save_section_5_1_tables(output_root, comparison)
    save_section_5_1_timeseries_plot(output_root, comparison)
    save_section_5_1_channel_plot(output_root, comparison)
