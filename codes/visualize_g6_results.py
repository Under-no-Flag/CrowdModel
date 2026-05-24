from __future__ import annotations

import argparse
from pathlib import Path

from crowd_bellman.g6_visualization import build_g6_visual_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures for completed G6 horizontal-comparison results.")
    parser.add_argument(
        "--output-root",
        default="codes/results/g6_horizontal_comparison",
        help="Directory containing G6_seed_summary.csv and related completed G6 outputs.",
    )
    parser.add_argument(
        "--exclude-methods",
        default="tpe_mixed_bo",
        help="Comma-separated method names to omit from the generated figures.",
    )
    parser.add_argument("--top-n", type=int, default=12, help="Number of best HF candidates to export.")
    parser.add_argument(
        "--figure-dir-name",
        default="paper_figures_no_tpe",
        help="Subdirectory under the output root for PNG/PDF paper figures.",
    )
    args = parser.parse_args()

    excluded = tuple(item.strip() for item in args.exclude_methods.split(",") if item.strip())
    report = build_g6_visual_report(
        Path(args.output_root),
        exclude_methods=excluded,
        top_n=args.top_n,
        figure_dir_name=args.figure_dir_name,
    )
    print(f"Wrote G6 visual report to {report['output_root']}")
    for name, path in report["outputs"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
