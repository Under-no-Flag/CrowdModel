from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .core import (
    compute_cfl_dt,
    greenshields_speed,
    precompute_step_factors,
    recover_optimal_direction,
    solve_bellman,
    update_density,
)
from .metrics import build_summary, init_case_stats, record_step, save_case_timeseries, save_json
from .plotting import save_case_snapshot, save_comparison_plot, save_timeseries_plot
from .scenes import BaseScene, CaseModel, SimulationConfig, build_case_model, build_three_channel_scene


def simulate_case(
    cfg: SimulationConfig,
    scene: BaseScene,
    case: CaseModel,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rho = scene.initial_rho.copy()
    rho[~case.walkable] = 0.0

    step_factor = precompute_step_factors(
        walkable=case.walkable,
        dx=cfg.dx,
        m11=case.m11,
        m12=case.m12,
        m22=case.m22,
    )

    stats = init_case_stats(list(scene.channel_masks))
    sink_total = 0.0
    time_value = 0.0
    phi = np.full_like(rho, np.inf)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)

    step = 0
    while step < cfg.steps and time_value < cfg.time_horizon - 1.0e-12:
        if step % cfg.bellman_every == 0:
            speed = greenshields_speed(rho, cfg.vmax, cfg.rho_max)
            phi = solve_bellman(
                walkable=case.walkable,
                exit_mask=case.exit_mask,
                allowed_mask=case.allowed_mask,
                speed=speed,
                step_factor=step_factor,
                f_eps=cfg.bellman_f_eps,
            )
            ux, uy = recover_optimal_direction(
                walkable=case.walkable,
                exit_mask=case.exit_mask,
                allowed_mask=case.allowed_mask,
                speed=speed,
                step_factor=step_factor,
                phi=phi,
                f_eps=cfg.bellman_f_eps,
            )

        speed = greenshields_speed(rho, cfg.vmax, cfg.rho_max)
        vx = speed * ux
        vy = speed * uy
        vx[~case.walkable] = 0.0
        vy[~case.walkable] = 0.0

        dt = compute_cfl_dt(
            speed=speed,
            m11=case.m11,
            m22=case.m22,
            dx=cfg.dx,
            cfl=cfg.cfl,
            dt_cap=cfg.dt_cap,
        )

        rho, fx, _, sink_increment = update_density(
            rho=rho,
            walkable=case.walkable,
            exit_mask=case.exit_mask,
            vx=vx,
            vy=vy,
            dx=cfg.dx,
            dt=dt,
        )
        rho = np.clip(rho, 0.0, cfg.rho_max)

        sink_total += sink_increment
        time_value += dt

        record_step(
            stats=stats,
            time_value=time_value,
            rho=rho,
            walkable=case.walkable,
            vx=vx,
            vy=vy,
            fx=fx,
            sink_total=sink_total,
            dt=dt,
            dx=cfg.dx,
            channel_masks=case.channel_masks,
            probe_x=case.probe_x,
        )

        if (step % cfg.save_every) == 0 or step == cfg.steps - 1 or time_value >= cfg.time_horizon - 1.0e-12:
            save_case_snapshot(
                path=output_dir / f"snapshot_{step:04d}.png",
                title=f"{case.title} | step={step} | t={time_value:.2f}",
                rho=rho,
                phi=phi,
                ux=ux,
                uy=uy,
                walkable=case.walkable,
                rho_max=cfg.rho_max,
            )
        step += 1

    save_case_timeseries(output_dir / "timeseries.csv", stats)
    save_timeseries_plot(output_dir / "timeseries.png", case.title, stats)

    summary = build_summary(case.case_id, case.title, stats)
    summary["config"] = asdict(cfg)
    save_json(output_dir / "summary.json", summary)
    return summary


def run_cases(
    cfg: SimulationConfig,
    cases: tuple[str, ...],
    output_root: Path,
) -> list[dict[str, object]]:
    scene = build_three_channel_scene(cfg)
    summaries: list[dict[str, object]] = []
    for case_id in cases:
        case = build_case_model(case_id, scene)
        summaries.append(simulate_case(cfg, scene, case, output_root / case_id))

    save_json(output_root / "comparison_summary.json", {"cases": summaries})
    save_comparison_plot(output_root / "comparison.png", summaries)
    return summaries


def run_cli(default_cases: tuple[str, ...]) -> None:
    parser = argparse.ArgumentParser(description="Run unified Bellman crowd experiments.")
    parser.add_argument("--cases", nargs="+", default=list(default_cases))
    parser.add_argument("--output-root", default="codes/results")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--time-horizon", type=float, default=None)
    args = parser.parse_args()

    cfg = SimulationConfig()
    if args.steps is not None:
        cfg = SimulationConfig(**{**asdict(cfg), "steps": args.steps})
    if args.save_every is not None:
        cfg = SimulationConfig(**{**asdict(cfg), "save_every": args.save_every})
    if args.time_horizon is not None:
        cfg = SimulationConfig(**{**asdict(cfg), "time_horizon": args.time_horizon})

    run_cases(cfg=cfg, cases=tuple(args.cases), output_root=Path(args.output_root))
