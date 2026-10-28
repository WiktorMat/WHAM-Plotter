"""Top-production cross-section normalization uncertainty."""

from uncertainties.base import NormalizationUncertainty, name_or_sample_matches

# Configurable fractional variations (5% in the TIDAL reference setup).
TOP_XS_UP = 0.05
TOP_XS_DOWN = 0.05


def _is_top(name: str, process: object) -> bool:
    return name_or_sample_matches(
        name, process, ("tt", "top", "st_", "st-", "singletop")
    )


TOP_CROSS_SECTION = NormalizationUncertainty(
    name="top_xs",
    down=TOP_XS_DOWN,
    up=TOP_XS_UP,
    applies_to=_is_top,
)
