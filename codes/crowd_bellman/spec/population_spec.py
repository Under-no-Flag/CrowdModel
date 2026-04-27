from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InitialGroupSpec:
    group_id: str
    stage_id: str
    region: str
    density: float


@dataclass(frozen=True)
class InflowGroupSpec:
    group_id: str
    stage_id: str
    region: str
    rate: float
    time_start: float = 0.0
    time_end: float | None = None
    rho_cap: float | None = None


@dataclass(frozen=True)
class PopulationSpec:
    initial_groups: tuple[InitialGroupSpec, ...] = ()
    inflow_groups: tuple[InflowGroupSpec, ...] = ()
