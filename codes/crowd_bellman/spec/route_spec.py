from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlSpec:
    mode: str
    region: str | None = None
    alpha: float | None = None
    beta: float | None = None
    value: float | None = None
    direction: str | None = None
    target_region: str | None = None
    target_point: tuple[int, int] | None = None
    allowed_directions: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TransitionTargetSpec:
    stage_id: str
    probability: float


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    group_key: tuple[int, int]
    goal_regions: tuple[str, ...]
    sink_regions: tuple[str, ...] | None = None
    allowed_directions: tuple[str, ...] | None = None
    controls: tuple[ControlSpec, ...] = ()
    decision_regions: tuple[str, ...] | None = None
    next_stage: str | None = None
    kappa: float = 1.0
    targets: tuple[TransitionTargetSpec, ...] = ()
    transition_direction: str = "stop"


@dataclass(frozen=True)
class CaseRouteSpec:
    case_id: str
    title: str
    stages: tuple[StageSpec, ...]
