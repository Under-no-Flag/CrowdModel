# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a doctoral research project on macroscopic crowd dynamics modeling for large passenger flow management (e.g., Shanghai Bund tourist area). The research combines:
- **Hughes' continuous model**: Classical macroscopic crowd model with density evolution
- **Anisotropic geometric guidance**: Metric tensor M(x) for channel guidance
- **HJB control constraints**: Discrete Bellman solver for directional constraints (one-way channels)

## Directory Structure

```
codes/              # Python simulation code
crowd_bellman/    # Main simulation package
results/          # Default output directory (generated)
methodology/      # Methodology documentation (Chinese)
实现/             # Implementation details (CFL, Godunov scheme, etc.)
writing/          # Academic paper writing
期刊论文章节/      # Journal paper sections (01-摘要.md to 07-附件.md)
大论文对应章节/    # Thesis chapter mappings
records/          # Daily work logs (YYYYMMMDD.md format)
plans/            # Planning documents
references/       # Reference materials
images/           # Images and figures
```

## Code Architecture

The simulation is built around a unified discrete Bellman solver:

### Core Modules (codes/crowd_bellman/)

- **core.py**: Core algorithms
  - `solve_bellman()`: Dijkstra-like HJB solver using 8-direction discretization
  - `update_density()`: Continuity equation with upwind flux
  - `compute_cfl_dt()`: Adaptive time step based on CFL condition
  - `tensor_from_tau()`: Build metric tensor M from channel tangent/normal

- **scenes.py**: Scene configuration
  - `SimulationConfig`: Simulation parameters (nx, ny, dx, steps, vmax, etc.)
  - `build_three_channel_scene()`: Three-channel corridor with obstacles
  - `build_case_model()`: Case configurations (baseline, middle/top/bottom guided)

- **runner.py**: Simulation execution
  - `simulate_case()`: Main simulation loop (Bellman solve → direction recovery → density update)
  - `run_cases()`: Batch run multiple cases

- **validation.py**: One-way HJB validation
  - `run_validation()`: Verifies τ·∇φ = -1/f(ρ) for strict one-way reduction

- **metrics.py** / **plotting.py** / **reporting.py**: Metrics, visualization, and reporting

### Entry Points

```bash
# Run full experiments (default: case1 + case2)
python codes/run_experiments.py

# Run specific cases
python codes/run_experiments.py --cases case1_baseline case2_middle_guided

# Run one-way HJB validation
python codes/verify_unidirectional_hjb.py

# Generate Section 5.1 report from completed runs
python codes/report_section_5_1.py --output-root codes/results

# Thin wrappers for individual cases
python codes/simulate_scene1.py          # Case 1 only
python codes/simulate_method2_scene2.py  # Case 2 only
python codes/simulate_method3_scene1.py  # Both cases
```

### Available Cases

- `case1_baseline`: No geometry guidance, all three channels available
- `case2_middle_guided`: Soft guidance toward middle channel, one-way motion inside
- `case3_top_guided`: Soft guidance toward top channel, one-way motion inside
- `case4_bottom_guided`: Soft guidance toward bottom channel, one-way motion inside

### Command-Line Options

```bash
python codes/run_experiments.py \
  --cases case1_baseline case2_middle_guided \
  --output-root codes/results \
  --steps 600 \
  --save-every 40 \
  --time-horizon 40.0 \
  --skip-validation \
  --skip-report
```

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

### Key References

- `methodology/model.md`: Theoretical model (anisotropic eikonal, HJB constraints)
- `methodology/实现/`: Implementation details (CFL stability, Godunov scheme, discrete HJB)
- `codes/README.md`: Code-specific documentation
- `writing/期刊论文架构.md`: Paper outline and methodology summary

## Dependencies

Standard scientific Python stack:
- numpy
- matplotlib

No requirements.txt or setup.py present; ensure these are available in the environment.
