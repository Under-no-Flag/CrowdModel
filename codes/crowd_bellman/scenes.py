from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .core import GroupKey, TransitionRule


@dataclass(frozen=True)
class SimulationConfig:
    nx: int = 96
    ny: int = 72
    dx: float = 0.5
    steps: int = 600
    time_horizon: float = 40.0
    vmax: float = 1.5
    rho_max: float = 5.0
    rho_init: float = 2.2
    bellman_every: int = 1
    bellman_f_eps: float = 0.08
    cfl: float = 0.9
    dt_cap: float = 0.18
    save_every: int = 40
    bellman_backend: str = "optimized"
    direction_recovery_backend: str = "vectorized"


@dataclass(frozen=True)
class BaseScene:
    """Geometric scene and baseline initialization."""

    walkable: np.ndarray
    initial_rho: np.ndarray
    exit_mask: np.ndarray
    channel_masks: dict[str, np.ndarray]
    probe_x: dict[str, int]
    wall_x0: int
    wall_x1: int
    tooth_x1: int
    centers_y: tuple[int, int, int]
    middle_entry: tuple[int, int]


@dataclass(frozen=True)
class GroupModel:
    """Per-(stage, route) model fields."""

    key: GroupKey
    name: str
    goal_mask: np.ndarray
    sink_mask: np.ndarray
    allowed_mask: np.ndarray
    m11: np.ndarray
    m12: np.ndarray
    m22: np.ndarray


@dataclass(frozen=True)
class CaseModel:
    """Simulation-ready case configuration.

    Legacy single-group fields are kept for backward compatibility.
    """

    case_id: str
    title: str
    walkable: np.ndarray
    exit_mask: np.ndarray
    channel_masks: dict[str, np.ndarray]
    probe_x: dict[str, int]
    m11: np.ndarray
    m12: np.ndarray
    m22: np.ndarray
    allowed_mask: np.ndarray
    groups: dict[GroupKey, GroupModel] | None = None
    transitions: tuple[TransitionRule, ...] = ()
    initial_group_density: dict[GroupKey, np.ndarray] | None = None
