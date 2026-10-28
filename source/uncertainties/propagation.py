"""Propagation helpers for normalization-only uncertainties."""

from __future__ import annotations

import numpy as np

from uncertainties.registry import DEFAULT_NORMALIZATION_UNCERTAINTIES


def normalization_variance(h, cfg, region: str) -> tuple[np.ndarray, np.ndarray]:
    """Return squared Down/Up shifts of the total prediction, bin by bin.

    Processes affected by one nuisance move together (full correlation), while
    distinct nuisances are combined in quadrature.
    """

    nbins = h.axes[-1].size
    down2 = np.zeros(nbins)
    up2 = np.zeros(nbins)
    available = set(h.axes["process"])
    stack = set(cfg.stack_order())
    for uncertainty in DEFAULT_NORMALIZATION_UNCERTAINTIES:
        shifted = np.zeros(nbins)
        for process in uncertainty.processes(cfg):
            if process not in available or process not in stack:
                continue
            shifted += h[{
                "process": process,
                "region": region,
                "variation": "nominal",
            }].view()["value"]
        down2 += (uncertainty.down * shifted) ** 2
        up2 += (uncertainty.up * shifted) ** 2
    return down2, up2
