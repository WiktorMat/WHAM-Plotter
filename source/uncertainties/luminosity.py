"""Integrated-luminosity normalization uncertainty."""

from uncertainties.base import NormalizationUncertainty, is_mc

# Configurable fractional variations (1.4% in the TIDAL reference setup).
LUMINOSITY_UP = 0.014
LUMINOSITY_DOWN = 0.014

LUMINOSITY = NormalizationUncertainty(
    name="lumi",
    down=LUMINOSITY_DOWN,
    up=LUMINOSITY_UP,
    applies_to=is_mc,
)
