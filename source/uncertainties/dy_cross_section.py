"""Drell--Yan cross-section normalization uncertainty."""

from uncertainties.base import NormalizationUncertainty, name_or_sample_matches

# Configurable fractional variations from the TIDAL reference setup.
DY_XS_UP = 0.013
DY_XS_DOWN = 0.016


def _is_dy(name: str, process: object) -> bool:
    return name_or_sample_matches(name, process, ("dy", "ztt", "zl"))


DY_CROSS_SECTION = NormalizationUncertainty(
    name="dy_xs",
    down=DY_XS_DOWN,
    up=DY_XS_UP,
    applies_to=_is_dy,
)
