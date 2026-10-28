"""Common representation of uncertainties that do not need varied histograms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


ProcessSelector = Callable[[str, object], bool]


@dataclass(frozen=True)
class NormalizationUncertainty:
    """One fully correlated normalization nuisance.

    ``down`` and ``up`` are positive fractional yield changes.  For example,
    ``down=0.016`` means a multiplicative Down factor of 0.984.
    """

    name: str
    down: float
    up: float
    applies_to: ProcessSelector

    @property
    def factors(self) -> tuple[float, float]:
        return 1.0 - self.down, 1.0 + self.up

    def processes(self, cfg: object) -> list[str]:
        return [
            name for name, process in cfg.processes.items()
            if self.applies_to(name, process)
        ]


def is_mc(_name: str, process: object) -> bool:
    return getattr(process, "kind", None) == "mc"


def name_or_sample_matches(name: str, process: object, prefixes: tuple[str, ...]) -> bool:
    candidates = [name, *getattr(process, "samples", [])]
    return any(
        candidate.lower().replace("+", "").startswith(prefixes)
        for candidate in candidates
    )
