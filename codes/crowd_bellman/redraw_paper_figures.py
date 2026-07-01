from __future__ import annotations

import json
import shutil
from pathlib import Path

from .g1_mechanism import (
    _save_m_attraction_flow_comparison,
    _save_u_direction_field_comparison,
    _save_um_configuration_flow_comparison,
)
from .g1_u_bidirectional import _save_bidirectional_line_plot
from .g2_strategy import build_g2_strategy_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "codes" / "results"
WRITING_ARCHIVE = PROJECT_ROOT / "writing" / "images" / "experiment_result_archive"


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def redraw_g1_figures() -> list[Path]:
    result_dir = RESULTS_ROOT / "g1_mechanism_reframed"
    mechanism = _load_json(result_dir / "g1_mechanism_summary.json")
    bidirectional = _load_json(result_dir / "g1_u_bidirectional_summary.json")

    behavior_cases = list(mechanism["behavior_cases"])  # type: ignore[index]
    validation_cases = list(bidirectional["validation_cases"])  # type: ignore[index]

    outputs = [
        result_dir / "g1_m_attraction_flow_comparison.png",
        result_dir / "g1_u_direction_field_comparison.png",
        result_dir / "g1_um_configuration_flow_comparison.png",
        result_dir / "g1_bidirectional_dynamics_lines.png",
    ]

    _save_m_attraction_flow_comparison(outputs[0], behavior_cases)
    _save_u_direction_field_comparison(outputs[1], behavior_cases)
    _save_um_configuration_flow_comparison(outputs[2], behavior_cases)
    _save_bidirectional_line_plot(outputs[3], validation_cases)

    archive_dir = WRITING_ARCHIVE / "g1"
    for output in outputs:
        _copy(output, archive_dir / output.name)
    return [archive_dir / output.name for output in outputs]


def redraw_g2_figures() -> list[Path]:
    result_dir = RESULTS_ROOT / "g2_multistage_direction_scan"
    report = _load_json(result_dir / "g2_strategy_summary.json")

    build_g2_strategy_report(
        output_root=result_dir,
        case_summaries=list(report["cases"]),  # type: ignore[index]
        behavior_summaries=list(report["behavior_cases"]),  # type: ignore[index]
    )

    outputs = [
        result_dir / "g2_control_tradeoff_summary.png",
        result_dir / "g2_control_tradeoff_summary_ab.png",
        result_dir / "g2_spatial_mechanism_summary.png",
    ]
    for output in outputs:
        _copy(output, WRITING_ARCHIVE / output.name)
    return [WRITING_ARCHIVE / output.name for output in outputs]


def main() -> None:
    outputs = redraw_g1_figures() + redraw_g2_figures()
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
