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
  Runs the reframed G1 mechanism-validation batch, covering the three-channel `baseline / M-only / U-only / middle / top / bottom` cases, one multi-stage bridge case, and one bidirectional `U(x)` counterflow-validation pair. It writes per-case behavior plots, `g1_behavior_metrics.csv`, `g1_u_bidirectional_metrics.csv`, branch comparison figures, and one shared `g1_mechanism_summary.json`.
- `python codes/g2_runner.py`
  Runs the G2 multistage directional-setting scan. It keeps one unrestricted `baseline` with middle-channel route preference, then scans six channel-direction configurations and writes Pareto, objective-comparison, channel-load, hotspot-migration, and summary outputs.
- `python codes/g3_runner.py`
  Runs the G3 behavior-layer batch for `single-stage approximation / multi-stage uniform preference / full multi-stage preference`, then writes behavior-layer comparison tables and plots.
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
- `codes/scenes/examples/three_channel_bidirectional/`
  Three-channel counterflow variant with west-to-east and east-to-west groups, used to validate whether `U(x)` suppresses reverse motion and head-on conflicts inside the middle channel.
- `codes/scenes/examples/tour_hardcoded/`
  Migrated TOML version of the old multi-stage sightseeing scene.
- `codes/scenes/examples/g2_multistage_directional/`
  G2-specific multistage browsing scene: middle-preference entry to the right platform, southward tour, and return-to-left baseline used for channel-direction scans.
- `codes/scenes/examples/tour_hardcoded/run_single_stage_approx.toml`
  G3 single-stage approximation baseline.
- `codes/scenes/examples/tour_hardcoded/run_uniform_preference.toml`
  G3 multi-stage / uniform-preference comparison case.
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
- `python codes/g2_runner.py --steps 600 --time-horizon 40`
- `python codes/g3_runner.py --steps 600 --time-horizon 40`

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
