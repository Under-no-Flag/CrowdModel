# Unified Bellman Crowd Simulator

`codes` has been refactored around a single discrete Bellman solver that supports:

- isotropic Hughes baseline;
- geometry guidance via a spatially varying SPD tensor `M(x)`;
- direction constraints via a local admissible control set `U(x)`;
- density evolution, quantitative metrics, and one-way HJB validation.

## Main entry points

- `python codes/run_experiments.py`
  Runs the three-channel scene experiments, the one-way validation, and the Section 5.1 report under one `codes/results/` folder by default.
- `python codes/verify_unidirectional_hjb.py`
  Verifies the strict one-way reduction `tau · grad(phi) = -1 / f(rho)`.
- `python codes/report_section_5_1.py --output-root <results_dir>`
  Generates Section 5.1 summary tables and paper-ready figures from completed runs.
- `python codes/simulate_scene1.py`
  Thin wrapper for `Case 1`.
- `python codes/simulate_method2_scene2.py`
  Thin wrapper for `Case 2`.
- `python codes/simulate_method3_scene1.py`
  Thin wrapper that runs both experiment cases together.

## Available Cases

- `case1_baseline`: no geometry guidance, all three channels remain available.
- `case2_middle_guided`: soft guidance toward the middle channel, with one-way motion inside the middle lane.
- `case3_top_guided`: soft guidance toward the top channel, with one-way motion inside the top lane.
- `case4_bottom_guided`: soft guidance toward the bottom channel, with one-way motion inside the bottom lane.

## Output

Results are written to the `--output-root` directory. The default and recommended location is `codes/results/`.

Typical contents:

- per-case time-series csv;
- per-case summary json;
- section 5.1 csv and markdown tables;
- section 5.1 comparison figures;
- field snapshots and time-series figures;
- cross-case comparison figure;
- one-way HJB validation report.
