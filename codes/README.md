# Unified Bellman Crowd Simulator

`codes` has been refactored around a single discrete Bellman solver that supports:

- isotropic Hughes baseline;
- geometry guidance via a spatially varying SPD tensor `M(x)`;
- direction constraints via a local admissible control set `U(x)`;
- density evolution, quantitative metrics, and one-way HJB validation.

## Main entry points

- `python codes/simulate_from_config.py --config <run.toml>`
  Runs a config-driven simulation from TOML files. This is now the primary simulation entrypoint.
- `python codes/g1_runner.py`
  Runs the G1 mechanism-validation batch for `C0/C1/C2/C3` and writes one shared `comparison_summary.json` plus `comparison.png`.
- `python codes/verify_unidirectional_hjb.py`
  Verifies the strict one-way reduction `tau · grad(phi) = -1 / f(rho)`.
- `python codes/report_section_5_1.py --output-root <results_dir>`
  Generates Section 5.1 summary tables and paper-ready figures from completed runs.
- `python codes/evaluate_objectives.py --input <summary.json|comparison_summary.json> --weights <weights.toml>`
  Re-evaluates `J1/J2/J5/J` for one or more weight sets directly from saved results, without rerunning the PDE solver.

## Config workflow

The old hardcoded case-id workflow has been removed. Scene geometry, initial populations, and route/stage logic are now expected to come from TOML files.

Phase 1 examples:

- `codes/scenes/examples/bund_simplified/`
  Simplified Nanjing Road to Bund scene with channels `5/6` entry, southward touring on the platform, and split departure via channels `9/10` or return via `5/6`.
- `codes/scenes/examples/single_stage/`
  Minimal single-stage config example.
- `codes/scenes/examples/multi_stage/`
  Minimal multi-stage / split-routing config example.
- `codes/scenes/examples/three_channel_hardcoded/`
  Migrated TOML version of the old three-channel hardcoded scene family, including `baseline`, `M-only`, `U-only`, and `M+U-middle`.
- `codes/scenes/examples/tour_hardcoded/`
  Migrated TOML version of the old multi-stage sightseeing scene.
- `codes/scenes/README.md`
  Field-by-field guide for writing `run.toml`, `scene.toml`, `population.toml`, and `routes.toml`.

Example commands:

- `python codes/simulate_from_config.py --config codes/scenes/examples/single_stage/run.toml`
- `python codes/simulate_from_config.py --config codes/scenes/examples/multi_stage/run.toml`
- `python codes/simulate_from_config.py --config codes/scenes/examples/three_channel_hardcoded/run_m_only.toml`
- `python codes/simulate_from_config.py --config codes/scenes/examples/three_channel_hardcoded/run_u_only.toml`
- `python codes/simulate_from_config.py --config codes/scenes/examples/three_channel_hardcoded/run_middle_guided.toml`
- `python codes/simulate_from_config.py --config codes/scenes/examples/tour_hardcoded/run.toml`
- `python codes/g1_runner.py --steps 600 --time-horizon 40`

## Output

Results are written to the `--output-root` directory. The default and recommended location is `codes/results/`.

Typical contents:

- per-case time-series csv;
- per-case summary json;
- per-case config snapshot copied from the resolved TOML inputs;
- config-driven run summary json;
- section 5.1 csv and markdown tables;
- section 5.1 comparison figures;
- field snapshots and time-series figures;
- one-way HJB validation report.
