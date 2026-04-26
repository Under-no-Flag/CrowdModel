# Run G2 Experiment

Run and configure the G2 (direction-setting scan) experiment for channel flow control analysis.

## Usage

```
/run_g2 [options]
```

Or use the runner directly:

```bash
# Run full G2 scan (baseline + 6 direction configurations)
python codes/g2_runner.py --output-root codes/results/g2_multistage_direction_scan

# Run with custom simulation steps and time horizon
python codes/g2_runner.py --steps 600 --time-horizon 40
```

## What G2 Experiment Does

G2 experiment tests **7 channel direction configurations** to compare their efficiency (J1), safety (J2), and balance (J5) metrics:

1. **baseline** - No direction restrictions (all channels bidirectional)
2. **case2-4** (single_entry family) - One eastbound entry + two westbound return channels
3. **case5-7** (single_return family) - One westbound return + two eastbound entry channels

## Generated Outputs

Results saved to `codes/results/g2_multistage_direction_scan/`:

| File | Description |
|------|-------------|
| `comparison_summary.json` | Full experiment results with all metrics |
| `g2_strategy_summary.json` | G2-specific analysis data |
| `g2_direction_metrics.csv` | CSV table of all cases and metrics |
| `g2_direction_pareto.png` | J1-J2 Pareto scatter plot with non-dominated solutions |
| `g2_direction_objectives.png` | J1/J2/J5 bar comparison chart |
| `g2_direction_channel_loads.png` | Channel flux share and peak timing |
| `g2_direction_hotspot_migration.png` | High-density hotspot centroid migration |
| `case1_baseline/` ... `case7_*/` | Individual case outputs (timeseries.csv, snapshots/) |

## Configuration Parameters

### 1. Modify Simulation Parameters (run_baseline.toml)

```toml
[simulation]
steps = 2000           # Total simulation steps (reduce for faster testing)
time_horizon = 140.0   # Max simulation time
save_every = 40        # Snapshot interval
nx = 128               # Grid resolution X
ny = 96                # Grid resolution Y
dx = 0.5               # Grid spacing
vmax = 1.5             # Free-flow speed
rho_max = 5.0          # Maximum density
rho_init = 2.2         # Initial crowd density
```

**Effect of changes:**
- Lower `steps`/`time_horizon` → Faster execution, may not complete evacuation
- Higher `rho_init` → More crowded, higher J1/J2 values
- Higher `vmax` → Faster evacuation, lower J1
- Finer grid (`nx`, `ny`) → Higher accuracy, slower computation

### 2. Modify Objective Weights (run_baseline.toml)

```toml
[objective]
lambda_j1 = 1.0        # Weight for total travel time (efficiency)
lambda_j2 = 1.0        # Weight for high-density exposure (safety)
lambda_j5 = 1.0        # Weight for channel flux variance (balance)
rho_safe = 3.5         # Safety density threshold
```

**Effect of changes:**
- `lambda_j1` higher → Prioritize faster evacuation
- `lambda_j2` higher → Prioritize crowd safety, avoid congestion
- `lambda_j5` higher → Prioritize even channel utilization
- `rho_safe` higher → Tolerate more congestion before counting J2

### 3. Modify Stage Behavior (routes_baseline.toml)

```toml
[[stages]]
stage_id = "enter_platform"
kappa = 2.0            # Transition rate to next stage

[[stages.controls]]
mode = "target_region"
alpha = 7.0            # Anisotropy strength
beta = 0.35            # Smoothing width
```

**Effect of changes:**
- `kappa` higher → Faster stage transitions, quicker flow through
- `alpha` higher → Stronger directional guidance toward target
- `beta` smaller → More localized guidance (narrower influence)

## Direction Setting Reference

The 6 direction configurations are defined in `codes/g2_runner.py`:

| Case | Family | Top | Middle | Bottom | Entry Channel(s) |
|------|--------|-----|--------|--------|------------------|
| case2 | single_entry | E | W | W | top |
| case3 | single_entry | W | E | W | middle |
| case4 | single_entry | W | W | E | bottom |
| case5 | single_return | W | E | E | middle, bottom |
| case6 | single_return | E | W | E | top, bottom |
| case7 | single_return | E | E | W | top, middle |

**To customize direction settings**, edit `DIRECTION_SETTINGS` in `codes/g2_runner.py`:

```python
DirectionSetting(
    case_id="case8_custom",
    title="Custom configuration",
    family="custom_family",
    directions={"top": "E", "middle": "E", "bottom": "E"},  # All eastbound
)
```

## Running Individual Cases

To run a single case instead of the full scan:

```bash
# Run baseline only
python codes/simulate_from_config.py \
  --config codes/scenes/examples/g2_multistage_directional/run_baseline.toml

# Run a specific generated case
python codes/simulate_from_config.py \
  --config codes/results/g2_multistage_direction_scan/_generated_configs/run_case3_topW_middleE_bottomW.toml
```

## Interpreting Results

### Key Metrics in comparison_summary.json

```json
{
  "cases": [
    {
      "case_id": "case1_baseline",
      "j1_total_travel_time": 1234.5,
      "j2_high_density_exposure": 567.8,
      "j5_channel_flux_variance": 0.15,
      "peak_density_max": 4.2,
      "final_sink_cumulative": 1000.0
    }
  ]
}
```

- **j1_total_travel_time** - Lower is better (efficiency)
- **j2_high_density_exposure** - Lower is better (safety)
- **j5_channel_flux_variance** - Lower is better (balance)
- **peak_density_max** - Maximum density reached (safety indicator)
- **final_sink_cumulative** - Total evacuated mass (throughput)

### Non-Dominated Solutions

Check `g2_strategy_summary.json` for `non_dominated_cases` - these are configurations that cannot be improved in all three objectives simultaneously.

## Common Workflows

### Quick Test Run
```bash
python codes/g2_runner.py --steps 300 --time-horizon 30
```

### Full Production Run
```bash
python codes/g2_runner.py --steps 2000 --time-horizon 140
```

### Compare Different Weights
1. Edit `run_baseline.toml` objective weights
2. Run G2 again to a different output directory:
   ```bash
   python codes/g2_runner.py --output-root codes/results/g2_weight_comparison_1
   ```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Simulation too slow | Reduce `steps` or grid resolution |
| J2 always zero | Lower `rho_safe` threshold |
| No evacuation | Check `time_horizon` is sufficient |
| Memory error | Reduce grid size (`nx`, `ny`) |
