# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a doctoral research project on macroscopic crowd dynamics modeling for large passenger flow management (e.g., Shanghai Bund tourist area). The research combines:
- **Hughes' continuous model**: Classical macroscopic crowd model with density evolution via continuity equation
- **Anisotropic geometric guidance**: Metric tensor M(x) for channel guidance (soft steering toward targets)
- **HJB control constraints**: Discrete Bellman solver for directional constraints (one-way channels via allowed_directions)

The unified discrete Bellman solver supports:
- Isotropic Hughes baseline
- Geometry guidance via spatially varying SPD tensor `M(x)`
- Direction constraints via local admissible control set `U(x)`
- Multi-stage routes with probabilistic splitting

## Common Commands

### Run Simulations

```bash
# Primary entry point - config-driven simulation
python codes/simulate_from_config.py --config codes/scenes/examples/single_stage/run.toml

# G1 mechanism validation batch (compares C0/C1/C2/C3 configurations)
python codes/g1_runner.py --steps 600 --time-horizon 40

# Verify strict one-way HJB reduction: τ · ∇φ = -1/f(ρ)
python codes/verify_unidirectional_hjb.py
```

### Run Tests

```bash
# Run all tests
cd codes && python -m unittest discover -s tests -v

# Run specific test file
cd codes && python -m unittest tests.test_core_backends -v

# Run single test method
cd codes && python -m unittest tests.test_core_backends.CoreBackendTests.test_solve_bellman_backends_match -v
```

### Generate Reports

```bash
# Generate Section 5.1 summary tables and figures from completed runs
python codes/report_section_5_1.py --output-root codes/results

# Re-evaluate objectives with different weights (no PDE re-run)
python codes/evaluate_objectives.py \
  --input codes/results/g1_formal/comparison_summary.json \
  --weights codes/scenes/examples/objective_sets/section_5_1.toml
```

### Benchmarking

```bash
# Run performance benchmark for core algorithms
python codes/benchmark_macro_model.py
```

## High-Level Architecture

### Config-Driven Workflow (Phase 1 Architecture)

The codebase has migrated from hardcoded cases to a config-driven architecture using 4 TOML files:

1. **`run.toml`**: Entry point that links other files and defines numerical parameters
2. **`scene.toml`**: Geometry (regions, obstacles, exits, channels for statistics)
3. **`population.toml`**: Initial crowd distribution (groups mapped to stages)
4. **`routes.toml`**: Case identifier, stages, control strategies, and stage transitions

Key directories:
- `codes/crowd_bellman/loaders/`: TOML parsing (`config_loader.py`)
- `codes/crowd_bellman/compilers/`: Config → runtime objects (`config_compiler.py`)
- `codes/crowd_bellman/spec/`: Dataclasses for config structures
- `codes/scenes/examples/`: Example configurations (bund_simplified, three_channel_hardcoded, etc.)

See `codes/scenes/README.md` for detailed config field documentation.

### Core Algorithm Structure

**`codes/crowd_bellman/core.py`** - Numerical kernels:
- `solve_bellman()`: Dijkstra-like HJB solver using 8-direction discretization (E/W/N/S/NE/NW/SE/SW)
- `update_density()`: Continuity equation with upwind flux, multigroup support
- `compute_cfl_dt()`: Adaptive time step based on CFL condition
- `precompute_step_factors()`: Precompute metric-dependent step factors for Bellman updates
- `recover_optimal_direction()`: Recover optimal control directions from value function
- `build_transition_out_rate_maps()`: Stage transition rate maps for multi-group simulations

**`codes/crowd_bellman/runner.py`** - Simulation loop:
- `simulate_case()`: Main loop (Bellman solve → direction recovery → density update → stage transitions)
- Outputs: timeseries CSV, summary JSON, config snapshots, field snapshots PNGs

**`codes/crowd_bellman/scenes.py`** - Runtime data structures:
- `SimulationConfig`: Grid (nx, ny, dx), time steps, vmax, rho_max, numerical params
- `BaseScene`: Walkable mask, initial density, exit/channel masks
- `CaseModel`: Groups, transition rules, metric tensor fields (m11, m12, m22)
- `GroupModel`: Per-group goal/sink/allowed masks and metric tensors

**`codes/crowd_bellman/config_workflow.py`** - Config execution:
- `run_from_config()`: Load → Compile → Simulate → Write outputs workflow
- `run_g1_workflow()`: Specialized workflow for G1 ablation experiments

### Multi-Group and Multi-Stage Support

The solver supports heterogeneous crowds via:
- **GroupKey = (stage_id, route_variant)**: Unique identifier for each sub-population
- **Per-group metric tensors**: Each group can have different M(x) for anisotropic guidance
- **Stage transitions**: Groups can transition between stages at decision regions with rate κ
- **Probabilistic splitting**: At decision points, groups can split by fixed probabilities to different target stages

### Output Structure

Results written to `codes/results/<case_id>/`:
- `summary.json`: Aggregated metrics (J1, J2, J5, total time, etc.)
- `timeseries.csv`: Time-varying statistics (density, flow, stage populations)
- `config_snapshot/`: Copied TOML files and manifest
- `snapshot_*.png`: Density field visualizations
- `timeseries.png`: Time series plots

For batch runs (like G1), `comparison_summary.json` contains all cases for comparison.

## Key Configuration Concepts

### Control Modes (in routes.toml)

Controls are applied in order, later controls override earlier ones:
- `identity`: No modification
- `isotropic`: Reset to isotropic tensor (needs `value`)
- `fixed_direction`: Fixed preferred direction (needs `direction`, `alpha`, `beta`)
- `target_region`: Steer toward region center (needs `target_region`, `alpha`, `beta`)
- `target_point`: Steer toward point (needs `target_point = [x, y]`, `alpha`, `beta`)

Alpha controls anisotropy strength, beta controls smoothing width.

### Direction Constraints

`allowed_directions = ["E", "NE", "SE"]` restricts available controls to subset of 8 directions. Used for one-way channels.

### Objective Function

`J = lambda_j1 * J1 + lambda_j2 * J2 + lambda_j5 * J5` where:
- J1: Total evacuation time
- J2: Congestion exposure (time above rho_safe)
- J5: Flow efficiency metric

## Development Workflow

### After Completing Any Task

Create a daily log in `records/YYYYMMMDD.md` with:
- Task objective description
- Completed work and artifacts (code files, documents, results)

### Writing Academic Content (Chinese)

When writing paper sections in `writing/期刊论文章节/`:
1. **Read methodology first**: Before writing methods, read `methodology/model.md` and `methodology/实现/` documents
2. **Paragraph style**: Write continuous prose, not bullet points
3. **Punctuation**: Avoid double quotes (") and dashes (—)
4. **Consistency**: Maintain consistent terminology and titles throughout


### Reports Writing

When writing reports (e.g., `2026-04-14汇报.md`):
- 包含最新的模型描述（公式与符号说明）
- 理论模式的更新内容与理由（对比上一次汇报，如没有则忽略）
- 当前已完成的理论分析、推导与数值公式推导、实现
- 当前已完成的实验与对应的结论分析
- 未来的理论研究安排和实验安排，需根据最新的plan和实验设计`experiments/实验设计.md`进行更新

## Dependencies

Standard scientific Python stack:
- numpy
- matplotlib

No requirements.txt or setup.py; ensure these are available.

## Key References

- `methodology/model.md`: Theoretical model (anisotropic eikonal, HJB constraints, M(x) tensor construction)
- `methodology/实现/`: Implementation details (CFL stability, Godunov scheme, discrete HJB algorithm)
- `codes/scenes/README.md`: Config file field reference
- `codes/README.md`: Entry points and config workflow overview
