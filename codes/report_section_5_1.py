from __future__ import annotations

import argparse
from pathlib import Path

from crowd_bellman.reporting import generate_section_5_1_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Section 5.1 tables and figures from experiment outputs.")
    parser.add_argument("--output-root", default="codes/results")
    args = parser.parse_args()
    generate_section_5_1_report(Path(args.output_root))


if __name__ == "__main__":
    main()
