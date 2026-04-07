from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import ObjectiveConfig
from ..scenes import SimulationConfig


@dataclass(frozen=True)
class RunConfigSpec:
    config_path: Path
    simulation: SimulationConfig
    objective: ObjectiveConfig
    scene_path: Path
    population_path: Path
    routes_path: Path
    output_root: Path
