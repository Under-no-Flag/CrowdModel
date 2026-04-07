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


@dataclass(frozen=True)
class CaseOverrides:
    """Per-case parameter overrides used by search/evaluation workers."""

    guidance_eta: float | None = None
    split_probabilities: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.guidance_eta is not None and self.guidance_eta < 1.0:
            raise ValueError("guidance_eta must be >= 1 when provided")
        if self.split_probabilities is not None:
            if len(self.split_probabilities) != 3:
                raise ValueError("split_probabilities must contain exactly three values")
            if any(value < 0.0 for value in self.split_probabilities):
                raise ValueError("split probabilities must be non-negative")
            if sum(self.split_probabilities) <= 1.0e-12:
                raise ValueError("split probabilities must have positive sum")


@dataclass(frozen=True)
class SearchConfig:
    """Candidate sets for the first-stage parameter search."""

    strategy_case_ids: tuple[str, ...] = (
        "case1_baseline",
        "case2_middle_guided",
        "case3_top_guided",
        "case4_bottom_guided",
    )
    eta_case_ids: tuple[str, ...] = (
        "case2_middle_guided",
        "case3_top_guided",
        "case4_bottom_guided",
    )
    eta_values: tuple[float, ...] = (2.0, 4.0, 8.0, 12.0)
    split_case_id: str = "case5_multistage_tour"
    split_probability_candidates: tuple[tuple[float, float, float], ...] = (
        (0.2, 0.3, 0.5),
        (0.3, 0.3, 0.4),
        (0.4, 0.2, 0.4),
    )
    top_k: int = 10

    def __post_init__(self) -> None:
        if not self.strategy_case_ids:
            raise ValueError("strategy_case_ids cannot be empty")
        if not self.eta_case_ids:
            raise ValueError("eta_case_ids cannot be empty")
        if not self.eta_values:
            raise ValueError("eta_values cannot be empty")
        if any(value < 1.0 for value in self.eta_values):
            raise ValueError("all eta_values must be >= 1")
        if not self.split_probability_candidates:
            raise ValueError("split_probability_candidates cannot be empty")
        for probs in self.split_probability_candidates:
            if len(probs) != 3:
                raise ValueError("each split probability candidate must contain exactly three values")
            if any(value < 0.0 for value in probs):
                raise ValueError("split probability candidates must be non-negative")
            if sum(probs) <= 1.0e-12:
                raise ValueError("each split probability candidate must have positive sum")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
