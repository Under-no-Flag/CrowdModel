from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionSpec:
    name: str
    x0: int | None = None
    x1: int | None = None
    y0: int | None = None
    y1: int | None = None
    shape: str = "rect"
    points: tuple[tuple[float, float], ...] = ()
    axis: tuple[float, float] | None = None


RectRegionSpec = RegionSpec


@dataclass(frozen=True)
class WallSpec:
    name: str
    points: tuple[tuple[float, float], ...]
    width: float = 1.0


@dataclass(frozen=True)
class NamedRegionSelectionSpec:
    name: str
    regions: tuple[str, ...]


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    regions: tuple[str, ...]
    probe_x: int | None = None


@dataclass(frozen=True)
class SceneSpec:
    block_boundaries: bool = True
    regions: tuple[RegionSpec, ...] = ()
    walls: tuple[WallSpec, ...] = ()
    obstacles: tuple[str, ...] = ()
    exits: tuple[NamedRegionSelectionSpec, ...] = ()
    channels: tuple[ChannelSpec, ...] = ()
