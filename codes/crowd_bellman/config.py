from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveConfig:
    """Objective-function settings for J1/J2/J5 aggregation."""

    lambda_j1: float = 1.0
    lambda_j2: float = 1.0
    lambda_j5: float = 1.0
    rho_safe: float = 3.5
    use_normalized_terms: bool = False
    j1_scale: float = 1.0
    j2_scale: float = 1.0
    j5_scale: float = 1.0
    name: str = "default"

    def __post_init__(self) -> None:
        if self.rho_safe <= 0.0:
            raise ValueError("rho_safe must be positive")
        if self.j1_scale <= 0.0 or self.j2_scale <= 0.0 or self.j5_scale <= 0.0:
            raise ValueError("objective normalization scales must be positive")
