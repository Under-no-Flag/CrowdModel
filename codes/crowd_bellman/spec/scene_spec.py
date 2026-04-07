from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RectRegionSpec:
    name: str
    x0: int
    x1: int
    y0: int
    y1: int


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
    regions: tuple[RectRegionSpec, ...] = ()
    obstacles: tuple[str, ...] = ()
    exits: tuple[NamedRegionSelectionSpec, ...] = ()
    channels: tuple[ChannelSpec, ...] = ()
