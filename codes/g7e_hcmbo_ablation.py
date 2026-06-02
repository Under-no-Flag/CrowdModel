from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "codes" / "scenes" / "examples" / "g7e_hcmbo_ablation" / "g7e.toml"

COMPONENT_LABELS = {
    "main_hcmbo_full": "HCMBO with LF hard shortlist",
    "no_lf_selection": "HCMBO",
    "no_dfo": "no DFO",
    "no_jb": "no J_B",
    "only_s_high": "only s with high capacity",
    "only_q_prior": "only q with prior directions",
    "random_search": "random search",
}

SEARCH_LABELS = {
    "hcmbo_structured_only": "HCMBO",
    "hcmbo_queue_aware_lcb": "queue-aware LCB",
    "hcmbo_rf_constrained_bo": "RF-style constrained BO",
    "hcmbo_adaptive_racing": "adaptive racing",
    "hcmbo_adaptive_racing_queue_aware": "adaptive racing + queue-aware",
    "hcmbo_current": "HCMBO with internal random search",
    "hcmbo_diverse_hf_topk": "diverse HF top-k",
    "hcmbo_trust_region": "trust-region search",
}


@dataclass(frozen=True)
class G7EAblationConfig:
    profile: str
    output_root: Path
    control_config: Path
    search_config: Path
    workers: int
    control_workers: int
    search_workers: int
    force: bool
    fail_fast: bool


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unified G7-E HCMBO ablation experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--profile", choices=("full", "smoke"), default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--control-workers", type=int, default=None)
    parser.add_argument("--search-workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    loaded = load_config(Path(args.config))
    profile = args.profile or loaded.profile
    workers = args.workers if args.workers is not None else loaded.workers
    output_root = Path(args.output_root).resolve() if args.output_root else loaded.output_root
    config = G7EAblationConfig(
        profile=profile,
        output_root=output_root,
        control_config=loaded.control_config,
        search_config=loaded.search_config,
        workers=workers,
        control_workers=args.control_workers if args.control_workers is not None else loaded.control_workers,
        search_workers=args.search_workers if args.search_workers is not None else loaded.search_workers,
        force=bool(args.force or loaded.force),
        fail_fast=bool(args.fail_fast or loaded.fail_fast),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "experiment_group": "G7-E",
        "design_version": "hcmbo_unified_ablation",
        "profile": config.profile,
        "config_path": str(Path(args.config).resolve()),
        "output_root": str(output_root),
        "control_config": str(config.control_config),
        "search_config": str(config.search_config),
        "workers": config.workers,
        "control_workers": config.control_workers,
        "search_workers": config.search_workers,
        "force": config.force,
        "argv": sys.argv,
        "runs": [],
    }
    write_json(output_root / "G7E_manifest.json", manifest)

    failures: list[str] = []
    failures.extend(run_control_ablation(config, manifest))
    if failures and config.fail_fast:
        write_outputs(output_root, manifest)
        raise RuntimeError(f"G7-E control ablation failed: {', '.join(failures)}")

    failures.extend(run_search_ablation(config, manifest))
    write_outputs(output_root, manifest)
    if failures:
        raise RuntimeError(f"G7-E failed blocks: {', '.join(failures)}")
    print(f"G7-E ablation summary: {output_root / 'G7E_ablation_summary.csv'}")


def load_config(path: Path) -> G7EAblationConfig:
    path = path.resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    table = dict(raw.get("g7e", {}))
    base_dir = path.parent
    workers = int(table.get("workers", 4))
    return G7EAblationConfig(
        profile=str(table.get("profile", "full")),
        output_root=resolve_config_path(base_dir, str(table.get("output_root", "../../../results/g7e_hcmbo_ablation"))),
        control_config=resolve_config_path(base_dir, str(table.get("control_config", "../g5_hcmbo_v2_small/g5.toml"))),
        search_config=resolve_config_path(base_dir, str(table.get("search_config", "../g7_hcmbo_variant_ablation/g7.toml"))),
        workers=workers,
        control_workers=int(table.get("control_workers", min(workers, 3))),
        search_workers=int(table.get("search_workers", min(workers, 4))),
        force=bool(table.get("force", False)),
        fail_fast=bool(table.get("fail_fast", False)),
    )


def resolve_config_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def run_control_ablation(config: G7EAblationConfig, manifest: dict[str, Any]) -> list[str]:
    output_dir = config.output_root / "control_component_ablation"
    command = [
        sys.executable,
        str(REPO_ROOT / "codes" / "g5_experiment_matrix.py"),
        "--config",
        str(config.control_config),
        "--profile",
        config.profile,
        "--output-root",
        str(output_dir),
        "--workers",
        str(config.control_workers),
    ]
    if config.force:
        command.append("--force")
    if config.fail_fast:
        command.append("--fail-fast")
    return run_block("control_component_ablation", command, output_dir, manifest)


def run_search_ablation(config: G7EAblationConfig, manifest: dict[str, Any]) -> list[str]:
    output_dir = config.output_root / "search_mechanism_ablation"
    command = [
        sys.executable,
        str(REPO_ROOT / "codes" / "g7_hcmbo_variant_ablation.py"),
        "--config",
        str(config.search_config),
        "--profile",
        config.profile,
        "--output-root",
        str(output_dir),
        "--workers",
        str(config.search_workers),
    ]
    if config.force:
        command.append("--force")
    if config.fail_fast:
        command.append("--fail-fast")
    return run_block("search_mechanism_ablation", command, output_dir, manifest)


def run_block(name: str, command: list[str], output_dir: Path, manifest: dict[str, Any]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / f"{name}_stdout.log"
    stderr_path = output_dir / f"{name}_stderr.log"
    entry: dict[str, Any] = {
        "name": name,
        "command": command,
        "output_dir": str(output_dir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "status": "running",
    }
    manifest.setdefault("runs", []).append(entry)
    write_json(Path(manifest["output_root"]) / "G7E_manifest.json", manifest)

    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, text=True)
    entry["runtime_seconds"] = time.perf_counter() - start
    entry["returncode"] = completed.returncode
    entry["status"] = "completed" if completed.returncode == 0 else "failed"
    write_json(Path(manifest["output_root"]) / "G7E_manifest.json", manifest)
    return [] if completed.returncode == 0 else [name]


def write_outputs(output_root: Path, manifest: dict[str, Any]) -> None:
    component_rows = build_component_rows(output_root / "control_component_ablation" / "G5_matrix_summary.csv")
    search_rows = build_search_rows(output_root / "search_mechanism_ablation" / "G7B_method_summary.csv")
    write_csv(output_root / "G7E_component_ablation.csv", component_rows)
    write_csv(output_root / "G7E_search_ablation.csv", search_rows)
    write_csv(output_root / "G7E_ablation_summary.csv", component_rows + search_rows)
    write_report(output_root, manifest, component_rows, search_rows)
    write_json(output_root / "G7E_manifest.json", manifest)


def build_component_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv(path):
        name = str(raw.get("experiment", ""))
        rows.append(
            {
                "panel": "control_component",
                "variant": COMPONENT_LABELS.get(name, name),
                "source_name": name,
                "best_hf_objective": raw.get("best_objective", ""),
                "feasible": raw.get("feasible", ""),
                "feasible_rate": "",
                "j1": raw.get("j1_eval") or raw.get("j1", ""),
                "j2": raw.get("j2_eval", ""),
                "j5": raw.get("j5_eval") or raw.get("j5", ""),
                "jb": raw.get("jb_normalized", ""),
                "jr": raw.get("jr_normalized", ""),
                "gate_rejected": raw.get("gate_rejected", ""),
                "note": "Control variable and objective-term ablation.",
            }
        )
    return sort_by_objective(rows)


def build_search_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv(path):
        name = str(raw.get("variant", ""))
        rows.append(
            {
                "panel": "search_mechanism",
                "variant": SEARCH_LABELS.get(name, name),
                "source_name": name,
                "best_hf_objective": raw.get("best_hf_objective", ""),
                "feasible": "",
                "feasible_rate": raw.get("feasible_rate", ""),
                "j1": "",
                "j2": raw.get("mean_j2_eval", ""),
                "j5": "",
                "jb": "",
                "jr": "",
                "gate_rejected": raw.get("mean_gate_rejected", ""),
                "note": "HCMBO search-mechanism ablation.",
            }
        )
    return sort_by_objective(rows)


def sort_by_objective(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> float:
        try:
            return float(row.get("best_hf_objective", "inf"))
        except (TypeError, ValueError):
            return float("inf")

    return sorted(rows, key=key)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "panel",
        "variant",
        "source_name",
        "best_hf_objective",
        "feasible",
        "feasible_rate",
        "j1",
        "j2",
        "j5",
        "jb",
        "jr",
        "gate_rejected",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_report(
    output_root: Path,
    manifest: dict[str, Any],
    component_rows: list[dict[str, Any]],
    search_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = [
        "# G7-E HCMBO Ablation Report",
        "",
        "## Scope",
        "",
        "- Unified HCMBO ablation over control components and search mechanisms.",
        "- Final ranking in each panel uses high-fidelity recheck objective values.",
        f"- Profile: `{manifest.get('profile')}`.",
        "",
        "## Best Rows",
        "",
    ]
    if component_rows:
        best = component_rows[0]
        lines.append(
            f"- Control component panel: `{best['variant']}` objective `{best['best_hf_objective']}`, "
            f"feasible `{best['feasible']}`."
        )
    if search_rows:
        best = search_rows[0]
        lines.append(
            f"- Search mechanism panel: `{best['variant']}` objective `{best['best_hf_objective']}`, "
            f"feasible rate `{best['feasible_rate']}`."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `G7E_manifest.json`: `{output_root / 'G7E_manifest.json'}`",
            f"- `G7E_component_ablation.csv`: `{output_root / 'G7E_component_ablation.csv'}`",
            f"- `G7E_search_ablation.csv`: `{output_root / 'G7E_search_ablation.csv'}`",
            f"- `G7E_ablation_summary.csv`: `{output_root / 'G7E_ablation_summary.csv'}`",
        ]
    )
    (output_root / "G7E_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
