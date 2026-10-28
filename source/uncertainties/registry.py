"""Default normalization uncertainties enabled by WHAM."""

from uncertainties.dy_cross_section import DY_CROSS_SECTION
from uncertainties.luminosity import LUMINOSITY
from uncertainties.top_cross_section import TOP_CROSS_SECTION

DEFAULT_NORMALIZATION_UNCERTAINTIES = (
    LUMINOSITY,
    DY_CROSS_SECTION,
    TOP_CROSS_SECTION,
)
