from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .config import ObjectiveConfig
from .core import (
    GroupKey,
    apply_fixed_probability_splitting,
    build_transition_out_rate_maps,
    compute_cfl_dt,
    compute_cfl_dt_multigroup,
    compute_total_density,
    greenshields_speed,
    precompute_step_factors,
    recover_optimal_direction,
    solve_bellman,
    update_density,
)
from .metrics import build_summary, init_case_stats, record_step, save_case_timeseries, save_json
from .plotting import save_case_snapshot, save_timeseries_plot
from .scenes import (
    BaseScene,
    CaseModel,
    GroupModel,
    SimulationConfig,
)


StepObserver = Callable[[dict[str, object]], None]


def _build_group_models(case: CaseModel) -> dict[GroupKey, GroupModel]:
    """Return group map; fallback to legacy single-group case."""

    if case.groups:
        return dict(case.groups)

    single = GroupModel(
        key=(0, 0),
        name="legacy_single_group",
        goal_mask=case.exit_mask,
        sink_mask=case.exit_mask,
        allowed_mask=case.allowed_mask,
        m11=case.m11,
        m12=case.m12,
        m22=case.m22,
    )
    return {(0, 0): single}


def _build_initial_group_density(
    scene: BaseScene,
    case: CaseModel,
    groups: dict[GroupKey, GroupModel],
) -> dict[GroupKey, np.ndarray]:
    """Initialize per-group density fields with backward compatibility."""

    rho_by_group = {key: np.zeros_like(scene.initial_rho) for key in groups}

    if case.initial_group_density:
        for key, rho in case.initial_group_density.items():
            if key not in rho_by_group:
                continue
            rho_by_group[key] = np.array(rho, copy=True)
    else:
        first_key = next(iter(groups.keys()))
        rho_by_group[first_key] = scene.initial_rho.copy()

    for key in groups:
        rho = rho_by_group[key]
        rho[~case.walkable] = 0.0
        rho_by_group[key] = np.clip(rho, 0.0, None)

    return rho_by_group


def simulate_case(
    cfg: SimulationConfig,
    scene: BaseScene,
    case: CaseModel,
    output_dir: Path,
    objective_cfg: ObjectiveConfig | None = None,
    step_observer: StepObserver | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if objective_cfg is None:
        objective_cfg = ObjectiveConfig()

    groups = _build_group_models(case)
    rho_by_group = _build_initial_group_density(scene, case, groups)

    step_factors = {
        key: precompute_step_factors(
            walkable=case.walkable,
            dx=cfg.dx,
            m11=group.m11,
            m12=group.m12,
            m22=group.m22,
        )
        for key, group in groups.items()
    }

    transitions = tuple(case.transitions)
    transition_rates = build_transition_out_rate_maps(transitions=transitions, shape=case.walkable.shape)

    stats = init_case_stats(list(scene.channel_masks))
    sink_total = 0.0
    time_value = 0.0

    phi_by_group = {key: np.full_like(scene.initial_rho, np.inf) for key in groups}
    ux_by_group = {key: np.zeros_like(scene.initial_rho) for key in groups}
    uy_by_group = {key: np.zeros_like(scene.initial_rho) for key in groups}

    step = 0
    while step < cfg.steps and time_value < cfg.time_horizon - 1.0e-12:
        rho_tot = compute_total_density(rho_by_group)
        speed = greenshields_speed(rho_tot, cfg.vmax, cfg.rho_max)

        if step % cfg.bellman_every == 0:
            for key, group in groups.items():
                phi_by_group[key] = solve_bellman(
                    walkable=case.walkable,
                    exit_mask=group.goal_mask,
                    allowed_mask=group.allowed_mask,
                    speed=speed,
                    step_factor=step_factors[key],
                    f_eps=cfg.bellman_f_eps,
                    backend=cfg.bellman_backend,
                )
                ux_by_group[key], uy_by_group[key] = recover_optimal_direction(
                    walkable=case.walkable,
                    exit_mask=group.goal_mask,
                    allowed_mask=group.allowed_mask,
                    speed=speed,
                    step_factor=step_factors[key],
                    phi=phi_by_group[key],
                    f_eps=cfg.bellman_f_eps,
                    backend=cfg.direction_recovery_backend,
                )

        vx_by_group: dict[GroupKey, np.ndarray] = {}
        vy_by_group: dict[GroupKey, np.ndarray] = {}
        for key in groups:
            # u is the Bellman-optimal heading (toward lower potential / goal); speed is non-negative.
            vx = speed * ux_by_group[key]
            vy = speed * uy_by_group[key]
            vx[~case.walkable] = 0.0
            vy[~case.walkable] = 0.0
            vx_by_group[key] = vx
            vy_by_group[key] = vy

        if len(groups) == 1 and not transitions:
            key0 = next(iter(groups.keys()))
            g0 = groups[key0]
            dt = compute_cfl_dt(
                speed=speed,
                m11=g0.m11,
                m22=g0.m22,
                dx=cfg.dx,
                cfl=cfg.cfl,
                dt_cap=cfg.dt_cap,
            )
        else:
            dt = compute_cfl_dt_multigroup(
                speed=speed,
                m11_by_group={key: group.m11 for key, group in groups.items()},
                m22_by_group={key: group.m22 for key, group in groups.items()},
                dx=cfg.dx,
                cfl=cfg.cfl,
                dt_cap=cfg.dt_cap,
                transition_out_rate_by_group=transition_rates,
            )

        fx_sum = np.zeros((scene.initial_rho.shape[0], scene.initial_rho.shape[1] - 1), dtype=float)
        vy_weighted = np.zeros_like(scene.initial_rho)
        vx_weighted = np.zeros_like(scene.initial_rho)

        for key, group in groups.items():
            rho_next, fx, _, sink_increment = update_density(
                rho=rho_by_group[key],
                walkable=case.walkable,
                exit_mask=group.sink_mask,
                vx=vx_by_group[key],
                vy=vy_by_group[key],
                dx=cfg.dx,
                dt=dt,
            )
            rho_by_group[key] = np.clip(rho_next, 0.0, cfg.rho_max)
            fx_sum += fx
            sink_total += sink_increment
            vx_weighted += rho_by_group[key] * vx_by_group[key]
            vy_weighted += rho_by_group[key] * vy_by_group[key]

        if transitions:
            rho_by_group = apply_fixed_probability_splitting(
                rho_by_group=rho_by_group,
                transitions=transitions,
                dt=dt,
                walkable=case.walkable,
            )
            for key in groups:
                rho_by_group[key] = np.clip(rho_by_group[key], 0.0, cfg.rho_max)

        rho_tot = compute_total_density(rho_by_group)
        rho_safe = np.maximum(rho_tot, 1.0e-8)
        vx_total = vx_weighted / rho_safe
        vy_total = vy_weighted / rho_safe
        vx_total[rho_tot <= 1.0e-8] = 0.0
        vy_total[rho_tot <= 1.0e-8] = 0.0

        time_value += dt

        record_step(
            stats=stats,
            time_value=time_value,
            rho=rho_tot,
            walkable=case.walkable,
            vx=vx_total,
            vy=vy_total,
            fx=fx_sum,
            sink_total=sink_total,
            dt=dt,
            dx=cfg.dx,
            rho_safe=objective_cfg.rho_safe,
            channel_masks=case.channel_masks,
            probe_x=case.probe_x,
        )

        if step_observer is not None:
            vis_key = (0, 0) if (0, 0) in groups else next(iter(groups.keys()))
            step_observer(
                {
                    "step": step,
                    "time": time_value,
                    "dt": dt,
                    "rho": rho_tot,
                    "speed": speed,
                    "vx": vx_total,
                    "vy": vy_total,
                    "phi": phi_by_group[vis_key],
                    "ux": ux_by_group[vis_key],
                    "uy": uy_by_group[vis_key],
                    "rho_by_group": rho_by_group,
                    "vx_by_group": vx_by_group,
                    "vy_by_group": vy_by_group,
                }
            )

        if (step % cfg.save_every) == 0 or step == cfg.steps - 1 or time_value >= cfg.time_horizon - 1.0e-12:
            vis_key = (0, 0) if (0, 0) in groups else next(iter(groups.keys()))
            save_case_snapshot(
                path=output_dir / f"snapshot_{step:04d}.png",
                title=f"{case.title} | step={step} | t={time_value:.2f}",
                rho=rho_tot,
                phi=phi_by_group[vis_key],
                ux=ux_by_group[vis_key],
                uy=uy_by_group[vis_key],
                walkable=case.walkable,
                rho_max=cfg.rho_max,
            )
        step += 1

    save_case_timeseries(output_dir / "timeseries.csv", stats)
    save_timeseries_plot(output_dir / "timeseries.png", case.title, stats)

    summary = build_summary(case.case_id, case.title, stats, objective_cfg=objective_cfg)
    summary["config"] = asdict(cfg)
    summary["objective_config"] = asdict(objective_cfg)
    summary["group_count"] = len(groups)
    summary["transition_count"] = len(transitions)
    save_json(output_dir / "summary.json", summary)
    return summary
