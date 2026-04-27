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
    enforce_total_density_cap,
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
    InflowModel,
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


def _apply_inflows(
    *,
    rho_by_group: dict[GroupKey, np.ndarray],
    inflows: tuple[InflowModel, ...],
    time_value: float,
    dt: float,
    dx: float,
    rho_max: float,
) -> float:
    if not inflows or dt <= 0.0:
        return 0.0

    cell_area = dx * dx
    window_start = time_value
    window_end = time_value + dt
    added_mass_total = 0.0

    for inflow in inflows:
        if inflow.rate <= 0.0:
            continue
        active_start = max(window_start, inflow.time_start)
        active_end = min(window_end, inflow.time_end) if inflow.time_end is not None else window_end
        active_dt = active_end - active_start
        if active_dt <= 1.0e-12:
            continue

        mask = inflow.region_mask
        cell_count = int(np.count_nonzero(mask))
        if cell_count == 0:
            continue

        area = float(cell_count) * cell_area
        rho_increment = float(inflow.rate * active_dt / area)
        if rho_increment <= 0.0:
            continue

        rho_limit = rho_max if inflow.rho_cap is None else min(float(inflow.rho_cap), rho_max)
        rho_total = compute_total_density(rho_by_group)
        remaining_capacity = np.maximum(rho_limit - rho_total[mask], 0.0)
        applied_increment = np.minimum(remaining_capacity, rho_increment)
        if not np.any(applied_increment > 0.0):
            continue

        rho_by_group[inflow.key][mask] += applied_increment
        added_mass_total += float(np.sum(applied_increment) * cell_area)

    return added_mass_total


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
    inflow_total = 0.0
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
            for handoff_rule in case.handoff_rules:
                source_ux = ux_by_group.get(handoff_rule.source)
                source_uy = uy_by_group.get(handoff_rule.source)
                target_ux = ux_by_group.get(handoff_rule.target)
                target_uy = uy_by_group.get(handoff_rule.target)
                if source_ux is None or source_uy is None or target_ux is None or target_uy is None:
                    continue
                mask = handoff_rule.handoff_mask & case.walkable
                source_ux[mask] = target_ux[mask]
                source_uy[mask] = target_uy[mask]

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

        if transitions:
            rho_by_group = apply_fixed_probability_splitting(
                rho_by_group=rho_by_group,
                transitions=transitions,
                dt=dt,
                walkable=case.walkable,
            )
            for key in groups:
                rho_by_group[key] = np.clip(rho_by_group[key], 0.0, cfg.rho_max)

        inflow_total += _apply_inflows(
            rho_by_group=rho_by_group,
            inflows=case.inflows,
            time_value=time_value,
            dt=dt,
            dx=cfg.dx,
            rho_max=cfg.rho_max,
        )

        rho_by_group = enforce_total_density_cap(
            rho_by_group=rho_by_group,
            rho_max=cfg.rho_max,
            walkable=case.walkable,
        )

        rho_tot = compute_total_density(rho_by_group)
        vx_weighted = np.zeros_like(scene.initial_rho)
        vy_weighted = np.zeros_like(scene.initial_rho)
        for key in groups:
            vx_weighted += rho_by_group[key] * vx_by_group[key]
            vy_weighted += rho_by_group[key] * vy_by_group[key]
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
            inflow_total=inflow_total,
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
                panel_title=f"{groups[vis_key].name} potential and direction",
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
