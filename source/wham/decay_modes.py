"""Decay-mode plot categories derived from the main analysis configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wham.config import AnalysisConfig


@dataclass(frozen=True)
class DecayModeCategory:
    name: str
    cut: str
    output_subdir: Path


def _dm_cuts(column: str, leg: int) -> tuple[tuple[str, str], ...]:
    branch = f"{column}_{leg}"
    # PNet distinguishes rho/a1 submodes which are intentionally combined in
    # the physics-facing pi/rho/a1 categories.  The legacy branch has no DM 2.
    rho = f"({branch} == 1 | {branch} == 2)" if column == "decayModePNet" else f"{branch} == 1"
    return (
        ("pi", f"{branch} == 0"),
        ("rho", rho),
        ("a1", f"({branch} == 10 | {branch} == 11)"),
    )


def decay_mode_categories(cfg: AnalysisConfig) -> tuple[DecayModeCategory, ...]:
    """Return the extra categories requested by levels 2 and 3.

    Level 1 is represented by the ordinary, unsplit pipeline and therefore
    returns no extra categories.  Level 3 includes all level-2 categories.
    """
    split = cfg.decay_mode_split
    if split is None or split.level == 1:
        return ()

    out: list[DecayModeCategory] = []
    if split.channel in ("mt", "et"):
        lepton = "mu" if split.channel == "mt" else "e"
        for dm, cut in _dm_cuts(split.column, 2):
            name = f"{lepton}-{dm}"
            out.append(DecayModeCategory(name, cut, Path(name)))
        return tuple(out)  # level 3 is identical for one-tau channels

    leg1 = _dm_cuts(split.column, 1)
    leg2 = _dm_cuts(split.column, 2)
    for dm, cut in leg1:
        name = f"{dm}-anything"
        out.append(DecayModeCategory(name, cut, Path(name)))
    for dm, cut in leg2:
        name = f"anything-{dm}"
        out.append(DecayModeCategory(name, cut, Path(name)))

    if split.level == 3:
        for dm1, cut1 in leg1:
            for dm2, cut2 in leg2:
                name = f"{dm1}-{dm2}"
                out.append(DecayModeCategory(
                    name,
                    f"({cut1}) & ({cut2})",
                    Path("split_by_decay_modes") / name,
                ))
    return tuple(out)


def category_config(cfg: AnalysisConfig, category: DecayModeCategory) -> AnalysisConfig:
    """Analysis variant with a category cut and category-specific sidecars."""
    return cfg.model_copy(update={
        # Keep the inclusive histogram index intact and isolate categories
        # from one another while still reusing the already-created skims.
        "name": f"{cfg.name}__dm__{category.name}",
        "selection": f"({cfg.selection}) & ({category.cut})",
        "output_dir": cfg.resolved_output_dir() / category.output_subdir,
    })
