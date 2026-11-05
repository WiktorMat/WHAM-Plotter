from __future__ import annotations

import matplotlib.axes
import pytest

pytest.importorskip("mplhep")

from wham.config import discover_samples, load_config
from wham.fill import fill_all
from wham.render.datamc import render_datamc
from wham.skim import ensure_skims


def test_blinded_render_keeps_data_driven_qcd_but_draws_no_data(
    workspace: dict, monkeypatch,
) -> None:
    cfg = load_config(workspace["yaml"])
    plots = cfg.plots.model_copy(update={
        "datamc_blind": True,
        "datamc_metrics": ["normalization"],
    })
    cfg = cfg.model_copy(update={"plots": plots})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    hists = fill_all(
        cfg, samples, skims, families=["datamc"], only_vars=("m_vis",),
        workers=1, use_cache=False, write_cache=False, sidecars=False,
    )

    h = hists[("datamc", "m_vis")]
    qcd = h[{"process": "QCD", "region": "OS_iso", "variation": "nominal"}]
    assert qcd.view()["value"].sum() > 0

    def fail_if_data_are_drawn(*args, **kwargs):
        raise AssertionError("blinded rendering must not draw data points")

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", fail_if_data_are_drawn)
    render_datamc(cfg, hists, workspace["tmp"] / "blinded")

    assert (workspace["tmp"] / "blinded" / "datamc" / "m_vis.png").is_file()
    assert (workspace["tmp"] / "blinded" / "datamc" / "m_vis.pdf").is_file()
