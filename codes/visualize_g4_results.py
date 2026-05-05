from __future__ import annotations

import argparse
from pathlib import Path

from crowd_bellman.g4_visualization import build_g4_visual_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual summaries for completed G4 SA-HBO/grid results.")
    parser.add_argument(
        "--output-root",
        default="codes/results/g4_sahbo_vs_grid",
        help="Directory containing g4_sahbo_grid_summary.json and G4 CSV outputs.",
    )
    parser.add_argument("--top-n", type=int, default=12, help="Number of best candidates to include in the top-case plot.")
    args = parser.parse_args()

    report = build_g4_visual_report(Path(args.output_root), top_n=args.top_n)
    print(f"Wrote G4 visual report to {report['output_root']}")
    for name, path in report["outputs"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
