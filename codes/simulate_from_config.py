from __future__ import annotations

import argparse
from pathlib import Path

from crowd_bellman.config_workflow import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a config-driven crowd simulation from TOML files.")
    parser.add_argument("--config", required=True, help="Path to the run TOML file")
    args = parser.parse_args()
    run_from_config(Path(args.config))


if __name__ == "__main__":
    main()
