from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wham.config import discover_samples, load_config, sample_scale
from wham.fill import (
    build_fill_spec,
    fill_all,
    fill_sample,
    merge_hists,
    resolution_name,
)
from wham.skim import ensure_skims

FAMILIES = ["resolution", "datamc", "cp"]


@pytest.fixture
def filled(workspace: dict) -> dict:
    cfg = load_config(workspace["yaml"])
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=FAMILIES)

    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))
    return {"cfg": cfg, "samples": samples, "skims": skims, "spec": spec, "hists": hists}


def _reference_frame(sample, cfg):
    df = pq.read_table(sample.path).to_pandas()
    sel = (
        (df.pt_1 > 25) & (df.eta_1.abs() < 2.4)
    )
    df = df[sel].copy()
    scale = sample_scale(sample, cfg.lumi)
    df["w"] = df.weight * scale
    return df


def test_datamc_regions_partition(filled: dict) -> None:
    cfg = filled["cfg"]
    h = filled["hists"][("datamc", "m_vis")]
    sample = next(s for s in filled["samples"] if s.kind == "data")
    df = _reference_frame(sample, cfg)
    df = df[(df.trg == 1) & (df.m_vis >= 0) & (df.m_vis < 250)]

    expectations = {
        "OS_iso": (df.os == 1) & (df.id_2 >= 5),
        "SS_iso": (df.os == 0) & (df.id_2 >= 5),
        "OS_antiiso": (df.os == 1) & (df.id_2 > 1) & (df.id_2 < 5),
        "SS_antiiso": (df.os == 0) & (df.id_2 > 1) & (df.id_2 < 5),
    }
    for region, mask in expectations.items():
        got = float(
            h[{"process": "data", "region": region, "variation": "nominal"}]
            .view()["value"].sum()
        )
        assert got == pytest.approx(float(df.loc[mask, "w"].sum())), region

        variance = float(
            h[{"process": "data", "region": region, "variation": "nominal"}]
            .view()["variance"].sum()
        )
        assert variance == pytest.approx(float((df.loc[mask, "w"] ** 2).sum())), region


def test_resolution_relative_formula(filled: dict) -> None:
    cfg = filled["cfg"]
    name = resolution_name("pt_1", "pt_2")
    h = filled["hists"][("resolution", name)]
    sample = next(s for s in filled["samples"] if s.name == "TT_test")
    df = _reference_frame(sample, cfg)
    res = (df.pt_1 - df.pt_2) / df.pt_2
    in_range = (res >= -2) & (res < 2) & (df.pt_2 != 0)
    expected = df.w[in_range].sum()
    got = float(
        h[{"process": "TT", "region": "nominal", "variation": "nominal"}].view()["value"].sum()
    )
    assert got == pytest.approx(expected, rel=1e-9)


def test_cp_even_odd_weights(filled: dict) -> None:
    cfg = filled["cfg"]
    h = filled["hists"][("cp", "met_phi")]
    sample = next(s for s in filled["samples"] if s.name == "DY_test")
    df = _reference_frame(sample, cfg)
    in_range = (df.met_phi >= -3.2) & (df.met_phi < 3.2)
    for region, col in [("even", "wt_cp_sm"), ("odd", "wt_cp_ps")]:
        expected = (df.w * df[col])[in_range].sum()
        got = float(
            h[{"process": "DY", "region": region, "variation": "nominal"}].view()["value"].sum()
        )
        assert got == pytest.approx(expected, rel=1e-9), region
    # data sample contributes nothing to cp
    got_data = float(
        h[{"process": "data", "region": "even", "variation": "nominal"}].view()["value"].sum()
    )
    assert got_data == 0.0


def test_fitcp_signal_region_templates(workspace: dict) -> None:
    from wham.config import ComponentFillCfg

    cfg = load_config(workspace["yaml"])
    plots = cfg.plots.model_copy(
        update={"fitcp": [ComponentFillCfg(
            var="met_phi", process="TT",
            components={"even": "wt_cp_sm", "odd": "wt_cp_ps"},
        )]}
    )
    cfg = cfg.model_copy(update={"plots": plots})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["fitcp"])

    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))

    h = hists[("fitcp", "met_phi")]
    assert sorted(h.axes["region"]) == ["even", "odd"]
    sample = next(s for s in samples if s.name == "TT_test")
    df = _reference_frame(sample, cfg)
    # SR = trigger & os & iso (abcd) on top of base selection
    sr = (df.trg == 1) & (df.os == 1) & (df.id_2 >= 5)
    in_range = (df.met_phi >= -3.2) & (df.met_phi < 3.2)
    for region, col in [("even", "wt_cp_sm"), ("odd", "wt_cp_ps")]:
        expected = (df.w * df[col])[sr & in_range].sum()
        got = float(
            h[{"process": "TT", "region": region, "variation": "nominal"}].view()["value"].sum()
        )
        assert got == pytest.approx(expected, rel=1e-9), region
    # the fills are per-process: only TT was requested; data/QCD stay empty too
    for proc in ("DY", "data", "QCD"):
        assert float(
            h[{"process": proc, "region": "even", "variation": "nominal"}].view()["value"].sum()
        ) == 0.0


def test_fitcp_drops_nonfinite_component_weights(workspace: dict) -> None:
    from wham.config import ComponentFillCfg

    source = workspace["data_dir"] / "TT_test" / "nominal" / "merged.parquet"
    table = pq.read_table(source)
    idx = table.schema.get_field_index("wt_cp_sm")
    table = table.set_column(
        idx, "wt_cp_sm", pa.array(np.full(table.num_rows, np.nan))
    )
    pq.write_table(table, source)

    cfg = load_config(workspace["yaml"])
    plots = cfg.plots.model_copy(
        update={"fitcp": [ComponentFillCfg(
            var="met_phi", process="TT",
            components={"even": "wt_cp_sm", "odd": "wt_cp_ps"},
        )]}
    )
    cfg = cfg.model_copy(update={"plots": plots})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["fitcp"])
    sample = next(s for s in samples if s.name == "TT_test")

    with pytest.warns(RuntimeWarning, match="non-finite.*region=even"):
        hists = fill_sample(sample, skims[sample.name], spec)

    h = hists[("fitcp", "met_phi")]
    assert np.isfinite(h.view()["value"]).all()
    assert np.isfinite(h.view()["variance"]).all()
    assert h[{"process": "TT", "region": "even"}].view()["value"].sum() == 0.0
    assert h[{"process": "TT", "region": "odd"}].view()["value"].sum() > 0.0


def test_ffcheck_raw_vs_weighted(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    qcd = cfg.qcd.model_copy(update={"method": "ff", "ff_weight": "pt_2 / 100"})
    plots = cfg.plots.model_copy(update={"ffcheck": ["m_vis"]})
    cfg = cfg.model_copy(update={"qcd": qcd, "plots": plots})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["ffcheck"])

    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))

    h = hists[("ffcheck", "m_vis")]
    assert list(h.axes["region"]) == ["OS_antiiso_raw", "OS_antiiso_ff"]
    sample = next(s for s in samples if s.kind == "data")
    df = _reference_frame(sample, cfg)
    anti = (
        (df.trg == 1) & (df.os == 1) & (df.id_2 > 1) & (df.id_2 < 5)
        & (df.m_vis >= 0) & (df.m_vis < 250)
    )
    expectations = {
        "OS_antiiso_raw": df.w[anti].sum(),
        "OS_antiiso_ff": (df.w * df.pt_2 / 100)[anti].sum(),
    }
    for region, expected in expectations.items():
        got = float(
            h[{"process": "data", "region": region, "variation": "nominal"}]
            .view()["value"].sum()
        )
        assert got == pytest.approx(float(expected), rel=1e-9), region


def test_unrolled_fill_matches_2d_reference(workspace: dict) -> None:
    from wham.config import VariableCfg

    cfg = load_config(workspace["yaml"])
    variables = {**cfg.variables, "unr": VariableCfg(unroll=("m_vis", "pt_1"))}
    plots = cfg.plots.model_copy(update={"datamc": ["unr"]})
    cfg = cfg.model_copy(update={"variables": variables, "plots": plots})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["datamc"])

    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))

    h = hists[("datamc", "unr")]
    assert h.axes[-1].size == 25 * 20  # m_vis(25) x pt_1(20)

    sample = next(s for s in samples if s.name == "TT_test")
    df = _reference_frame(sample, cfg)
    sel = (df.trg == 1) & (df.os == 1) & (df.id_2 >= 5)  # OS_iso (abcd)
    ref2d, _, _ = np.histogram2d(
        df.m_vis[sel], df.pt_1[sel], bins=[25, 20],
        range=[[0, 250], [0, 100]], weights=df.w[sel],
    )
    got = h[{"process": "TT", "region": "OS_iso", "variation": "nominal"}].view()["value"]
    # index = ix + nx*iy (y-major blocks) -> transpose then flatten
    assert np.allclose(got, ref2d.T.reshape(-1))


def test_process_cut_filters_events(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    # an extra per-event mask on the DY process only (a genmatch cut stand-in)
    procs = dict(cfg.processes)
    procs["DY"] = procs["DY"].model_copy(update={"cut": "id_2 >= 4"})
    cfg = cfg.model_copy(update={"processes": procs})
    assert "id_2" in cfg.required_columns()  # the cut's column gets skimmed

    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["datamc"], only_vars=("m_vis",))
    assert ("DY", "id_2 >= 4") in spec.process_cuts

    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))
    h = hists[("datamc", "m_vis")]

    # DY: only id_2 >= 4 events survive, in every region it fills
    sample = next(s for s in samples if s.name == "DY_test")
    df = _reference_frame(sample, cfg)
    df = df[(df.trg == 1) & (df.m_vis >= 0) & (df.m_vis < 250)]
    expect = df[(df.os == 1) & (df.id_2 >= 5) & (df.id_2 >= 4)].w.sum()
    got = float(h[{"process": "DY", "region": "OS_iso",
                   "variation": "nominal"}].view()["value"].sum())
    assert got == pytest.approx(expect, rel=1e-9)

    # data carries no cut -> unaffected
    data = next(s for s in samples if s.kind == "data")
    ddf = _reference_frame(data, cfg)
    ddf = ddf[(ddf.trg == 1) & (ddf.m_vis >= 0) & (ddf.m_vis < 250)]
    d_expect = float(ddf.loc[(ddf.os == 1) & (ddf.id_2 >= 5), "w"].sum())
    d_got = float(h[{"process": "data", "region": "OS_iso",
                     "variation": "nominal"}].view()["value"].sum())
    assert d_got == pytest.approx(d_expect)


def test_weight_variation_fills_and_completion(workspace: dict) -> None:
    from wham.config import VariationCfg
    from wham.fill import complete_variation_slices
    from wham.qcd import estimate_qcd

    cfg = load_config(workspace["yaml"])
    cfg = cfg.model_copy(update={"variations": [VariationCfg(
        name="tt_sys", processes=["TT"],
        weight_up="weight * 2", weight_down="weight * 0.5",
    )]})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["datamc"], only_vars=("m_vis",))

    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))
    complete_variation_slices(spec, hists)

    h = hists[("datamc", "m_vis")]
    assert sorted(h.axes["variation"]) == ["nominal", "tt_sys_down", "tt_sys_up"]

    def total(proc: str, variation: str) -> float:
        return float(h[{"process": proc, "region": "OS_iso",
                        "variation": variation}].view()["value"].sum())

    assert total("TT", "tt_sys_up") == pytest.approx(2 * total("TT", "nominal"), rel=1e-9)
    assert total("TT", "tt_sys_down") == pytest.approx(0.5 * total("TT", "nominal"), rel=1e-9)
    # unmatched processes get the nominal content copied in
    assert total("DY", "tt_sys_up") == total("DY", "nominal")
    assert total("data", "tt_sys_up") == total("data", "nominal")

    # the varied TT subtraction propagates into that variation's QCD estimate
    estimate_qcd(cfg, hists)
    assert total("QCD", "tt_sys_up") <= total("QCD", "nominal")


def test_qcd_ff_variation(workspace: dict) -> None:
    from wham.config import VariationCfg
    from wham.fill import complete_variation_slices

    cfg = load_config(workspace["yaml"])
    qcd = cfg.qcd.model_copy(update={"method": "ff", "ff_weight": "pt_2 / 100"})
    cfg = cfg.model_copy(update={"qcd": qcd, "variations": [VariationCfg(
        name="ffv", target="qcd_ff",
        weight_up="pt_2 / 50", weight_down="pt_2 / 200",
    )]})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["datamc"], only_vars=("m_vis",))

    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))
    complete_variation_slices(spec, hists)

    h = hists[("datamc", "m_vis")]
    sample = next(s for s in samples if s.kind == "data")
    df = _reference_frame(sample, cfg)
    anti = (
        (df.trg == 1) & (df.os == 1) & (df.id_2 > 1) & (df.id_2 < 5)
        & (df.m_vis >= 0) & (df.m_vis < 250)
    )

    def total(region: str, variation: str) -> float:
        return float(h[{"process": "data", "region": region,
                        "variation": variation}].view()["value"].sum())

    # anti-iso entries carry the varied FF weight; iso entries are copied
    assert total("OS_antiiso", "ffv_up") == pytest.approx(
        float((df.w * df.pt_2 / 50)[anti].sum()), rel=1e-9)
    assert total("OS_antiiso", "ffv_down") == pytest.approx(
        float((df.w * df.pt_2 / 200)[anti].sum()), rel=1e-9)
    assert total("OS_iso", "ffv_up") == total("OS_iso", "nominal")


def test_columns_variation_scales_observable(workspace: dict) -> None:
    from wham.config import VariationCfg
    from wham.fill import complete_variation_slices

    cfg = load_config(workspace["yaml"])
    cfg = cfg.model_copy(update={"variations": [VariationCfg(
        name="tes", target="columns", processes=["DY"], factors={"m_vis": 0.9},
    )]})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["datamc"], only_vars=("m_vis",))

    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))
    complete_variation_slices(spec, hists)

    h = hists[("datamc", "m_vis")]
    # one slice named by the variation itself (not _up/_down)
    assert "tes" in list(h.axes["variation"])
    assert "tes_up" not in list(h.axes["variation"])

    # DY OS_iso 'tes' slice = histogram of (m_vis * 0.9); the selection has no
    # m_vis cut, so the varied slice sees the same events with shifted values
    sample = next(s for s in samples if s.name == "DY_test")
    df = _reference_frame(sample, cfg)
    sel = (df.trg == 1) & (df.os == 1) & (df.id_2 >= 5)
    ref, _ = np.histogram(df.m_vis[sel] * 0.9, bins=25, range=(0, 250), weights=df.w[sel])
    got = h[{"process": "DY", "region": "OS_iso", "variation": "tes"}].view()["value"]
    assert np.allclose(got, ref)

    def tot(proc: str, var: str) -> float:
        return float(h[{"process": proc, "region": "OS_iso",
                        "variation": var}].view()["value"].sum())

    # x-axis rescale within range preserves the DY yield; shape moves
    assert tot("DY", "tes") == pytest.approx(tot("DY", "nominal"), rel=1e-9)
    assert not np.allclose(got, h[{"process": "DY", "region": "OS_iso",
                                   "variation": "nominal"}].view()["value"])
    # unmatched process completed to nominal
    assert tot("TT", "tes") == tot("TT", "nominal")


def test_columns_variation_migrates_selection(workspace: dict) -> None:
    """Scaling a SELECTION column re-applies the cut on the shifted values:
    events just below the pt_1 threshold enter the varied slice (migration),
    while the nominal slice stays byte-identical to a variation-free fill."""
    from wham.config import VariationCfg
    from wham.fill import complete_variation_slices

    cfg0 = load_config(workspace["yaml"])
    cfg = cfg0.model_copy(update={"variations": [VariationCfg(
        name="tes", target="columns", processes=["DY"],
        factors={"pt_1": 1.1, "m_vis": 0.9},
    )]})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["datamc"], only_vars=("m_vis",))

    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))
    complete_variation_slices(spec, hists)
    h = hists[("datamc", "m_vis")]

    # reference from the RAW parquet (no pre-selection): the widened read must
    # recover events failing the nominal pt_1 > 25 but passing the shifted cut
    sample = next(s for s in samples if s.name == "DY_test")
    raw = pq.read_table(sample.path).to_pandas()
    w = raw.weight * sample_scale(sample, cfg.lumi)
    base = ((raw.eta_1.abs() < 2.4) & (raw.trg == 1) & (raw.os == 1)
            & (raw.id_2 >= 5))
    nom_sel = base & (raw.pt_1 > 25) & (raw.m_vis >= 0) & (raw.m_vis < 250)
    var_sel = base & (raw.pt_1 * 1.1 > 25)

    def tot(variation: str) -> float:
        return float(h[{"process": "DY", "region": "OS_iso",
                        "variation": variation}].view()["value"].sum())

    assert tot("nominal") == pytest.approx(float(w[nom_sel].sum()), rel=1e-9)
    ref, _ = np.histogram(raw.m_vis[var_sel] * 0.9, bins=25, range=(0, 250),
                          weights=w[var_sel])
    got = h[{"process": "DY", "region": "OS_iso", "variation": "tes"}].view()["value"]
    assert np.allclose(got, ref)
    assert tot("tes") > tot("nominal")  # migration in: pt_1 in (25/1.1, 25]

    # samples not targeted by the variation keep the exact read path: their
    # nominal contents match a variation-free fill exactly
    spec0 = build_fill_spec(cfg0, families=["datamc"], only_vars=("m_vis",))
    hists0: dict = {}
    for s in samples:
        merge_hists(hists0, fill_sample(s, skims[s.name], spec0))
    h0 = hists0[("datamc", "m_vis")]
    for proc in ("DY", "TT", "data"):
        a = h[{"process": proc, "region": "OS_iso", "variation": "nominal"}].view()
        b = h0[{"process": proc, "region": "OS_iso", "variation": "nominal"}].view()
        assert np.array_equal(a["value"], b["value"]), proc


def test_columns_variation_cache_key(workspace: dict) -> None:
    """The per-column factors are part of the histogram cache key: changing
    the grid point must never serve a stale histogram."""
    from wham.config import VariationCfg
    from wham.fill import spec_extras
    from wham.histcache import cache_key

    cfg = load_config(workspace["yaml"])
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    vcfg = cfg.variables["m_vis"].model_dump()

    def key(factors: dict) -> str:
        c = cfg.model_copy(update={"variations": [VariationCfg(
            name="tes", target="columns", processes=["DY"], factors=factors,
        )]})
        spec = build_fill_spec(c, families=["datamc"], only_vars=("m_vis",))
        extras = spec_extras(spec)
        return cache_key(c, samples, skims, spec, "datamc", "m_vis", vcfg,
                         extra=extras[("datamc", "m_vis")])

    assert key({"m_vis": 0.9}) != key({"m_vis": 0.95})
    assert key({"m_vis": 0.9}) != key({"m_vis": 0.9, "pt_1": 0.9})


def test_fitcp_cache_key_includes_weight_columns(workspace: dict) -> None:
    from wham.config import ComponentFillCfg
    from wham.fill import spec_extras
    from wham.histcache import cache_key

    cfg = load_config(workspace["yaml"])
    plots = cfg.plots.model_copy(
        update={"fitcp": [ComponentFillCfg(
            var="met_phi", process="TT",
            components={"even": "wt_cp_sm", "odd": "wt_cp_ps"},
        )]}
    )
    cfg = cfg.model_copy(update={"plots": plots})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["fitcp"])
    vcfg = cfg.variables["met_phi"].model_dump()

    extras = spec_extras(spec)
    k1 = cache_key(cfg, samples, skims, spec, "fitcp", "met_phi", vcfg,
                   extra=extras[("fitcp", "met_phi")])
    k2 = cache_key(cfg, samples, skims, spec, "fitcp", "met_phi", vcfg,
                   extra={"components": {"TT": {"even": "other_col", "odd": "wt_cp_ps"}}})
    assert k1 != k2
    # no extra -> matches legacy keying (datamc etc. unaffected)
    k3 = cache_key(cfg, samples, skims, spec, "datamc", "met_phi", vcfg)
    k4 = cache_key(cfg, samples, skims, spec, "datamc", "met_phi", vcfg, extra=None)
    assert k3 == k4


def test_column_alias_and_explicit_edges(workspace: dict) -> None:
    from wham.config import VariableCfg
    from wham.histcache import cache_key

    cfg = load_config(workspace["yaml"])
    variables = dict(cfg.variables)
    variables["m_vis_coarse"] = VariableCfg(column="m_vis", bins=[0.0, 50.0, 100.0, 250.0])
    plots = cfg.plots.model_copy(update={"datamc": [*cfg.plots.datamc, "m_vis_coarse"]})
    cfg = cfg.model_copy(update={"variables": variables, "plots": plots})

    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["datamc"])
    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))

    coarse = hists[("datamc", "m_vis_coarse")]
    assert list(coarse.axes[-1].edges) == [0.0, 50.0, 100.0, 250.0]

    # contents must be a rebin of the uniform m_vis hist (25 bins over [0, 250])
    base = hists[("datamc", "m_vis")]
    sl = {"process": "data", "region": "OS_iso", "variation": "nominal"}
    base_vals = base[sl].view()["value"]
    coarse_vals = coarse[sl].view()["value"]
    assert coarse_vals[0] == pytest.approx(base_vals[:5].sum())    # [0, 50)
    assert coarse_vals[1] == pytest.approx(base_vals[5:10].sum())  # [50, 100)
    assert coarse_vals[2] == pytest.approx(base_vals[10:].sum())   # [100, 250)

    # the alias's source column is part of the cache key
    vcfg = variables["m_vis_coarse"].model_dump()
    k1 = cache_key(cfg, samples, skims, spec, "datamc", "m_vis_coarse", vcfg)
    k2 = cache_key(cfg, samples, skims, spec, "datamc", "m_vis_coarse",
                   dict(vcfg, column="pt_1"))
    assert k1 != k2


def test_fill_all_uses_cache(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)

    h1 = fill_all(cfg, samples, skims, families=FAMILIES, workers=1, use_cache=True)
    # second call must come purely from cache (cache_only raises otherwise)
    h2 = fill_all(cfg, samples, skims, families=FAMILIES, workers=1,
                  use_cache=True, cache_only=True)
    k = ("datamc", "pt_1")
    assert np.allclose(h1[k].view()["value"], h2[k].view()["value"])


def test_cache_key_sensitivity(workspace: dict) -> None:
    from wham.histcache import cache_key

    cfg = load_config(workspace["yaml"])
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=FAMILIES)
    vcfg = cfg.variables["pt_1"].model_dump()

    base = cache_key(cfg, samples, skims, spec, "datamc", "pt_1", vcfg)

    cut = cfg.model_copy(update={"selection": "pt_1 > 30"})
    assert cache_key(cut, samples, skims, spec, "datamc", "pt_1", vcfg) != base

    relabeled = dict(vcfg, label="something new")
    assert cache_key(cfg, samples, skims, spec, "datamc", "pt_1", relabeled) == base

    rebinned = dict(vcfg, bins=99)
    assert cache_key(cfg, samples, skims, spec, "datamc", "pt_1", rebinned) != base
