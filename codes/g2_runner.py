from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.g2_strategy import G2StrategyCollector, build_g2_strategy_report
from crowd_bellman.metrics import save_json


BASELINE_CONFIG = Path("codes/scenes/examples/three_channel_hardcoded/run_baseline.toml")
GUIDED_CONFIGS = {
    "middle": Path("codes/scenes/examples/three_channel_hardcoded/run_middle_guided.toml"),
    "top": Path("codes/scenes/examples/three_channel_hardcoded/run_top_guided.toml"),
    "bottom": Path("codes/scenes/examples/three_channel_hardcoded/run_bottom_guided.toml"),
}
DEFAULT_ETAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _eta_slug(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML scalar value: {value!r}")


def _dump_table(table_name: str, payload: dict[str, object]) -> list[str]:
    lines = [f"[{table_name}]"]
    for key, value in payload.items():
        lines.append(f"{key} = {_format_scalar(value)}")
    return lines


def _dump_run_toml(payload: dict[str, object]) -> str:
    lines: list[str] = []
    for table_name in ("simulation", "objective", "scene", "population", "routes", "outputs"):
        table = payload.get(table_name, {})
        if not isinstance(table, dict):
            continue
        if lines:
            lines.append("")
        lines.extend(_dump_table(table_name, table))
    return "\n".join(lines) + "\n"


def _dump_routes_toml(payload: dict[str, object]) -> str:
    lines: list[str] = []
    case_table = payload.get("case", {})
    if isinstance(case_table, dict):
        lines.extend(_dump_table("case", case_table))

    stages = payload.get("stages", [])
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            lines.append("")
            lines.append("[[stages]]")
            for key, value in stage.items():
                if key in {"controls", "targets"}:
                    continue
                lines.append(f"{key} = {_format_scalar(value)}")
            controls = stage.get("controls", [])
            if isinstance(controls, list):
                for control in controls:
                    if not isinstance(control, dict):
                        continue
                    lines.append("")
                    lines.append("[[stages.controls]]")
                    for key, value in control.items():
                        lines.append(f"{key} = {_format_scalar(value)}")
            targets = stage.get("targets", [])
            if isinstance(targets, list):
                for target in targets:
                    if not isinstance(target, dict):
                        continue
                    lines.append("")
                    lines.append("[[stages.targets]]")
                    for key, value in target.items():
                        lines.append(f"{key} = {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _eta_applied_routes(base_routes: dict[str, object], eta: float, strategy: str) -> dict[str, object]:
    routes = {
        "case": dict(base_routes["case"]),
        "stages": [],
    }
    case_id = f"g2_{strategy}_eta_{_eta_slug(eta)}"
    routes["case"]["case_id"] = case_id
    routes["case"]["title"] = f"G2 {strategy} guided eta={eta:g}"

    for stage in base_routes.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_copy = dict(stage)
        controls_out: list[dict[str, object]] = []
        for control in stage.get("controls", []):
            if not isinstance(control, dict):
                continue
            control_copy = dict(control)
            beta = control_copy.get("beta")
            alpha = control_copy.get("alpha")
            if isinstance(beta, int | float) and isinstance(alpha, int | float):
                control_copy["alpha"] = float(beta) * float(eta)
            controls_out.append(control_copy)
        if controls_out:
            stage_copy["controls"] = controls_out
        routes["stages"].append(stage_copy)
    return routes


def _generated_config_paths(*, output_root: Path, strategy: str, eta: float) -> tuple[Path, Path]:
    generated_dir = output_root / "_generated_configs" / strategy
    generated_dir.mkdir(parents=True, exist_ok=True)
    slug = _eta_slug(eta)
    return (
        generated_dir / f"run_{strategy}_eta_{slug}.toml",
        generated_dir / f"routes_{strategy}_eta_{slug}.toml",
    )


def _write_eta_config(
    *,
    base_run_path: Path,
    strategy: str,
    eta: float,
    output_root: Path,
) -> Path:
    base_run = _load_toml(base_run_path)
    routes_table = base_run.get("routes", {})
    if not isinstance(routes_table, dict):
        raise ValueError(f"{base_run_path} does not contain [routes]")
    routes_path = (base_run_path.parent / str(routes_table["file"])).resolve()
    base_routes = _load_toml(routes_path)

    run_path, generated_routes_path = _generated_config_paths(output_root=output_root, strategy=strategy, eta=eta)
    generated_routes = _eta_applied_routes(base_routes, eta=eta, strategy=strategy)

    generated_run = {
        "simulation": dict(base_run["simulation"]),
        "objective": dict(base_run["objective"]),
        "scene": {"file": str((base_run_path.parent / str(base_run["scene"]["file"])).resolve())},
        "population": {"file": str((base_run_path.parent / str(base_run["population"]["file"])).resolve())},
        "routes": {"file": str(generated_routes_path.resolve())},
        "outputs": {"output_root": str(output_root.resolve())},
    }
    generated_run["objective"]["name"] = generated_routes["case"]["case_id"]

    generated_routes_path.write_text(_dump_routes_toml(generated_routes), encoding="utf-8")
    run_path.write_text(_dump_run_toml(generated_run), encoding="utf-8")
    return run_path


def _attach_scan_metadata(
    *,
    summary: dict[str, object],
    output_root: Path,
    strategy: str,
    eta: float | None,
    template_config: Path,
) -> None:
    scan = {
        "strategy": strategy,
        "eta": eta,
        "template_config": str(template_config.resolve()),
        "is_baseline": eta is None,
    }
    summary["g2_scan"] = scan
    metadata = summary.get("metadata")
    if isinstance(metadata, dict):
        metadata["g2_scan"] = scan
    case_output_dir = output_root / str(summary["case_id"])
    save_json(case_output_dir / "summary.json", summary)


def _parse_etas(raw: str | None) -> tuple[float, ...]:
    if raw is None or not raw.strip():
        return DEFAULT_ETAS
    values: list[float] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("No eta values parsed from --etas")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the G2 strategy-response eta sweep.")
    parser.add_argument("--output-root", default="codes/results/g2_eta_scan")
    parser.add_argument("--etas", default=None, help="Comma-separated eta values, e.g. 1,2,4,8,16,32")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--time-horizon", type=float, default=None)
    args = parser.parse_args()

    simulation_overrides: dict[str, object] = {}
    if args.steps is not None:
        simulation_overrides["steps"] = args.steps
    if args.save_every is not None:
        simulation_overrides["save_every"] = args.save_every
    if args.time_horizon is not None:
        simulation_overrides["time_horizon"] = args.time_horizon

    eta_values = _parse_etas(args.etas)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    collectors: dict[str, tuple[G2StrategyCollector, Path]] = {}

    def observer_factory(**kwargs: object):
        case = kwargs["case"]
        run_spec = kwargs["run_spec"]
        simulation = kwargs["simulation"]
        case_output_dir = Path(str(kwargs["case_output_dir"]))
        collector = G2StrategyCollector(
            case_id=str(case.case_id),
            title=str(case.title),
            walkable=case.walkable,
            channel_masks=case.channel_masks,
            rho_safe=float(run_spec.objective.rho_safe),
            dx=float(simulation.dx),
        )
        collectors[str(case.case_id)] = (collector, case_output_dir)
        return collector.observe

    summaries: list[dict[str, object]] = []

    baseline_summary = run_from_config(
        config_path=BASELINE_CONFIG,
        output_root=output_root,
        simulation_overrides=simulation_overrides or None,
        write_root_summary=False,
        step_observer_factory=observer_factory,
    )
    _attach_scan_metadata(
        summary=baseline_summary,
        output_root=output_root,
        strategy="baseline",
        eta=None,
        template_config=BASELINE_CONFIG,
    )
    summaries.append(baseline_summary)

    generated_config_paths: list[str] = []
    for strategy, base_run_path in GUIDED_CONFIGS.items():
        for eta in eta_values:
            generated_run_path = _write_eta_config(
                base_run_path=base_run_path,
                strategy=strategy,
                eta=eta,
                output_root=output_root,
            )
            generated_config_paths.append(str(generated_run_path.resolve()))
            summary = run_from_config(
                config_path=generated_run_path,
                output_root=output_root,
                simulation_overrides=simulation_overrides or None,
                write_root_summary=False,
                step_observer_factory=observer_factory,
            )
            _attach_scan_metadata(
                summary=summary,
                output_root=output_root,
                strategy=strategy,
                eta=eta,
                template_config=base_run_path,
            )
            summaries.append(summary)

    behavior_summaries: list[dict[str, object]] = []
    for summary in summaries:
        collector, case_output_dir = collectors[str(summary["case_id"])]
        behavior_summaries.append(collector.save_case_outputs(case_output_dir))

    payload = {
        "experiment_group": "G2",
        "design_version": "eta_scan",
        "baseline_config": str(BASELINE_CONFIG.resolve()),
        "guided_template_configs": {key: str(path.resolve()) for key, path in GUIDED_CONFIGS.items()},
        "generated_config_paths": generated_config_paths,
        "eta_values": [float(value) for value in eta_values],
        "cases": summaries,
        "behavior_cases": behavior_summaries,
    }
    save_json(output_root / "comparison_summary.json", payload)
    build_g2_strategy_report(
        output_root=output_root,
        case_summaries=summaries,
        behavior_summaries=behavior_summaries,
    )


if __name__ == "__main__":
    main()
