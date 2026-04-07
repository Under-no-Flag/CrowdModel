from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InitialGroupSpec:
    group_id: str
    stage_id: str
    region: str
    density: float


@dataclass(frozen=True)
class PopulationSpec:
    initial_groups: tuple[InitialGroupSpec, ...] = ()
