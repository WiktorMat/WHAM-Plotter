from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import uproot

from wham.combine import (
    CONTROL_CARD_FILE,
    DATACARD_FILE,
    HARVEST_CARD_FILE,
    dc_name,
    export_key,
    fit_templates,
    harvest_source,
    inject_toy_asymmetry,
    merge_command,
    model_source,
    run_fit,
    scan_command,
    scan_file,
    write_datacard,
    write_shapes,
)
from wham.config import discover_samples
from wham.fill import fill_all
from wham.fitconfig import (
    FitConfig,
    SystematicCfg,
    category_analysis,
    fit_families,
    load_fit_config,
    resolve_fit_config,
    template_parents,
    translate_formula,
    union_analysis,
)
from wham.skim import ensure_skims

COMBINE_IMAGE = Path(
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/combine-standalone:latest"
)


def _cmssw_dir() -> str | None:
    """A CMSSW release with combine + CombineHarvester, for the morph backend."""
    import os

    for cand in (os.environ.get("CMSSW_BASE"),
                 "/afs/cern.ch/user/h/haawedik/CMSSW_14_1_0_pre4"):
        if cand and (Path(cand) / "src").is_dir():
            return cand
    return None


CMSSW_DIR = _cmssw_dir()

_SCALE_MODEL = """
model:
  pois:
    r: {init: 1, range: [0, 3]}
  processes:
    DY: {scale: "r"}
"""

_COMPONENTS_MODEL = """
model:
  pois:
    mu:    {init: 1, range: [0, 5]}
    alpha: {init: 0.7854, range: [0, 1.5708]}
  processes:
    DY:
      components:
        even: {weight: wt_cp_sm, scale: "mu * cos(alpha)^2"}
        odd:  {weight: wt_cp_ps, scale: "mu * sin(alpha)^2"}
"""

_ONE_CATEGORY = """
categories:
  - {name: SR, variable: met_phi}
"""

_TWO_CATEGORIES = """
categories:
  - {name: lowj,  variable: met_phi, cut: "n_jets < 2"}
  - {name: highj, variable: met_phi, cut: "n_jets >= 2"}
"""


@pytest.fixture
def fit_setup(workspace: dict):
    """New-schema fit YAML against the synthetic workspace + filled hists."""

    def _make(model: str = "scale", *, toy: float = 0.0, two_cats: bool = False,
              extra: str = "") -> dict:
        model_block = _SCALE_MODEL if model == "scale" else _COMPONENTS_MODEL
        cats_block = _TWO_CATEGORIES if two_cats else _ONE_CATEGORY
        fit_yaml = workspace["tmp"] / f"fit_{model}_{toy}_{two_cats}.yaml"
        fit_yaml.write_text(
            f"""
name: testfit_{model}{"_2cat" if two_cats else ""}
analysis: {workspace["yaml"]}
{cats_block}
{model_block}
asimov: {{enabled: true}}
toy: {{asymmetry: {toy}}}
scans:
  - {{pois: [{"r" if model == "scale" else "alpha"}], points: 10}}
systematics:
  - {{name: lumi, effect: lnN, processes: [TT, DY], scaleFactor: 1.025}}
  - {{name: xsec_tt, effect: lnN, processes: [TT], scaleFactor: 1.05}}
  - {{name: norm_qcd, effect: lnN, processes: [QCD], scaleFactor: 1.3}}
{extra}""",
            encoding="utf-8",
        )
        fit, cfg, cat_cfgs = load_fit_config(fit_yaml)
        samples, _ = discover_samples(cfg)
        skims = ensure_skims(union_analysis(fit, cfg), samples, workers=1)
        hists = {}
        for cat in fit.categories:
            ccfg = category_analysis(fit, cfg, cat)
            hists[cat.name] = fill_all(
                ccfg, samples, skims, families=fit_families(fit),
                only_vars=(cat.variable,), workers=1, sidecars=False,
            )
        return {"fit": fit, "cfg": cfg, "cat_cfgs": cat_cfgs, "hists": hists,
                "samples": samples, "skims": skims, "tmp": workspace["tmp"],
                "yaml": fit_yaml}

    return _make


# ------------------------------------------------------------- formulas


def test_translate_formula() -> None:
    f, deps = translate_formula("mu * cos(alpha)^2", ["mu", "alpha"])
    assert f == "(@0*pow(cos(@1),2))"
    assert deps == ["mu", "alpha"]

    f, deps = translate_formula("r", ["r"])
    assert (f, deps) == ("@0", ["r"])

    f, deps = translate_formula("2*sin(a)*cos(a)", ["a"])
    assert f == "((2*sin(@0))*cos(@0))"
    assert deps == ["a"]

    # ** and ^ are equivalent
    assert translate_formula("x**2", ["x"]) == translate_formula("x^2", ["x"])


@pytest.mark.parametrize("bad", [
    "nope * 2",          # unknown name
    "max(r, 1)",         # disallowed function
    "r > 1",             # comparison
    "r.real",            # attribute access
    "pow(r, exponent=2)",  # keyword args
    "__import__('os')",  # nice try
    "r +",               # syntax error
])
def test_translate_formula_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid formula"):
        translate_formula(bad, ["r"])


# ------------------------------------------------------------- schema


def _fit_dict(**over) -> dict:
    base = {
        "name": "t",
        "analysis": "a",
        "categories": [{"name": "SR", "variable": "x"}],
        "model": {
            "pois": {"r": {"range": [0, 3]}},
            "processes": {"DY": {"scale": "r"}},
        },
    }
    base.update(over)
    return base


@pytest.mark.parametrize("over, msg", [
    ({"model": {"pois": {"r": {"range": [0, 3]}},
                "processes": {"DY": {"scale": "r", "components": {
                    "a": {"weight": "w", "scale": "r"}}}}}},
     "exactly one of"),
    ({"model": {"pois": {}, "processes": {"DY": {"scale": "r"}}}},
     "at least one POI"),
    ({"model": {"pois": {"r": {"range": [0, 3]}, "unused": {"range": [0, 1]}},
                "processes": {"DY": {"scale": "r"}}}},
     "not used by any scale"),
    ({"model": {"pois": {"r": {"range": [3, 0]}},
                "processes": {"DY": {"scale": "r"}}}},
     "must be increasing"),
    ({"scans": [{"pois": ["nope"], "points": 10}]}, "undeclared POIs"),
    ({"scans": [{"pois": ["r", "r"], "points": 10}]}, "duplicate POI"),
    ({"scans": [{"pois": ["r", "r2"], "points": 10, "range": [0, 1]}]},
     "1D scans only"),
    ({"scans": [{"pois": ["r"], "points": 10, "ranges": [[0, 1], [0, 1]]}]},
     "2D scans only"),
    ({"categories": [{"name": "SR", "variable": "x"},
                     {"name": "SR", "variable": "y"}]}, "duplicate category"),
    ({"categories": [{"name": "no spaces", "variable": "x"}]}, "datacard-safe"),
    ({"asimov": {"enabled": True, "parameters": {"nope": 1.0}}},
     "undeclared POIs"),
    ({"toy": {"asymmetry": 0.3}}, "exactly two components"),
    ({"systematics": [{"name": "s", "effect": "lnN", "processes": ["DY"],
                       "scaleFactor": 1.1, "categories": ["nope"]}]},
     "unknown"),
])
def test_fit_config_rejections(over: dict, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        FitConfig.model_validate(_fit_dict(**over))


def test_systematic_cfg_validation() -> None:
    with pytest.raises(ValueError, match="needs a scaleFactor"):
        SystematicCfg(name="x", effect="lnN", processes=["A"])
    with pytest.raises(ValueError, match="no scaleFactor"):
        SystematicCfg(name="x", effect="rateParam", processes=["A"], scaleFactor=1.1)
    with pytest.raises(ValueError, match="rateParam-only"):
        SystematicCfg(name="x", effect="lnN", processes=["A"], scaleFactor=1.1,
                      range=(0.1, 5.0))
    s = SystematicCfg(name="x", effect="rateParam", processes=["A"], range=(0.1, 5.0))
    assert s.init == 1.0

    with pytest.raises(ValueError, match="either weight_up/weight_down or scales"):
        SystematicCfg(name="x", effect="shape", processes=["A"], weight_up="w")
    with pytest.raises(ValueError, match="do not apply to shape"):
        SystematicCfg(name="x", effect="shape", processes=["A"],
                      weight_up="w", weight_down="w2", scaleFactor=1.1)
    with pytest.raises(ValueError, match="shape-only"):
        SystematicCfg(name="x", effect="lnN", processes=["A"], scaleFactor=1.1,
                      weight_up="w")


def test_dc_name_sanitization() -> None:
    assert dc_name("W+jets") == "W_jets"
    assert dc_name("DY_2Tau") == "DY_2Tau"


def test_resolve_fit_config_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_fit_config(str(tmp_path / "nope.yaml"))


# ------------------------------------------------------------- model source


def test_scan_window_defaults(fit_setup) -> None:
    fit = fit_setup("scale")["fit"]
    scan = fit.scans[0]  # 1D over r, no explicit range

    # no explicit window -> the full POI range (no auto-windowing)
    assert "--setParameterRanges r=0,3" in scan_command(fit, scan)

    # explicit range wins
    fixed = scan.model_copy(update={"range": (0.5, 1.5)})
    assert "--setParameterRanges r=0.5,1.5" in scan_command(fit, fixed)


def test_scan_points_per_axis() -> None:
    from wham.fitconfig import ScanCfg

    # per-POI grid counts: 2D only, both > 1
    ok = ScanCfg(pois=["a", "b"], points=(15, 51))
    assert ok.points == (15, 51)
    with pytest.raises(ValueError, match="2D scans only"):
        ScanCfg(pois=["a"], points=(15, 51))
    with pytest.raises(ValueError, match="at least 2 points per axis"):
        ScanCfg(pois=["a", "b"], points=(15, 1))
    with pytest.raises(ValueError, match="at least 2 points"):
        ScanCfg(pois=["a"], points=1)


def test_scan_command_grid_points(fit_setup) -> None:
    from wham.fitconfig import ScanCfg

    fit = fit_setup("components")["fit"]  # POIs: mu, alpha
    sym = ScanCfg(pois=["mu", "alpha"], points=8)
    cmd = scan_command(fit, sym)
    assert "--points 64" in cmd and "--gridPoints" not in cmd

    asym = ScanCfg(pois=["mu", "alpha"], points=(15, 51))
    cmd = scan_command(fit, asym)
    # combine ignores --points once --gridPoints is set (verified in the help)
    assert "--gridPoints 15,51" in cmd and "--points 765" in cmd


def test_fitdiag_seed(fit_setup) -> None:
    from wham.combine import fitdiag_command

    fit = fit_setup("scale")["fit"]  # fixture enables asimov
    # Asimov: --setParameters defines the generated truth — seed must be ignored
    assert "r=0.7" not in fitdiag_command(fit, seed={"r": 0.7})

    # data fit: exactly one --setParameters flag, carrying the seed
    data_fit = fit.model_copy(update={
        "asimov": fit.asimov.model_copy(update={"enabled": False})})
    cmd = fitdiag_command(data_fit, seed={"r": 0.7})
    assert cmd.count("--setParameters ") == 1 and "r=0.7" in cmd
    # no seed -> no flag at all (container backend, no asimov)
    assert "--setParameters" not in fitdiag_command(data_fit)

    # CH/morph backend: seed merges with the frozen r=1 into ONE flag
    morph = FitConfig.model_validate(_morph_fit())
    morph = morph.model_copy(update={
        "asimov": morph.asimov.model_copy(update={"enabled": False})})
    cmd = fitdiag_command(morph, seed={"tes_dm0": 0.978})
    assert cmd.count("--setParameters ") == 1
    assert "r=1" in cmd and "tes_dm0=0.978" in cmd


def test_scan_seed(fit_setup, tmp_path: Path) -> None:
    import numpy as np

    from wham.combine import scan_seed

    fit = fit_setup("scale")["fit"]
    scan = fit.resolved_scans()[0]  # 1D over r

    # a grid row deeper than the row-0 free fit -> its POI values are the seed
    with uproot.recreate(tmp_path / scan_file(fit, scan)) as f:
        f["limit"] = {"r": np.array([1.0, 0.6, 0.8, 1.2]),
                      "deltaNLL": np.array([0.0, -3.0, 1.0, np.nan])}
    assert scan_seed(tmp_path, fit) == {"r": pytest.approx(0.6)}

    # nothing (meaningfully) below the free fit -> no seed
    with uproot.recreate(tmp_path / scan_file(fit, scan)) as f:
        f["limit"] = {"r": np.array([1.0, 0.6, 0.8]),
                      "deltaNLL": np.array([0.0, 0.5, 1.0])}
    assert scan_seed(tmp_path, fit) == {}

    # missing scan file -> no seed
    (tmp_path / scan_file(fit, scan)).unlink()
    assert scan_seed(tmp_path, fit) == {}


def test_model_source_scale(fit_setup) -> None:
    src = model_source(fit_setup("scale")["fit"])
    assert 'self.modelBuilder.doVar("r[1,0,3]")' in src
    assert "SCALES = {'DY': 'r'}" in src  # bare POI: no expr:: needed
    assert 'self.modelBuilder.doSet("POI", "r")' in src


def test_per_category_scale() -> None:
    from wham.fitconfig import scale_entries

    fit = FitConfig.model_validate({
        "name": "t", "analysis": "a",
        "categories": [{"name": "c1", "variable": "x"},
                       {"name": "c2", "variable": "x"}],
        "model": {
            "pois": {"sf1": {"range": [0, 3]}, "sf2": {"range": [0, 3]}},
            "processes": {"DY": {"scale": {"c1": "sf1", "c2": "sf2"}}},
        },
    })
    # one entry per (category, template), carrying its own expression
    assert set(scale_entries(fit)) == {("c1", "DY", "sf1"), ("c2", "DY", "sf2")}

    src = model_source(fit)
    assert 'self.modelBuilder.doVar("sf1[1,0,3]")' in src
    assert 'self.modelBuilder.doVar("sf2[1,0,3]")' in src
    # SCALES keyed by (bin, process); getYieldScale falls back per process
    assert "('c1', 'DY'): 'sf1'" in src
    assert "('c2', 'DY'): 'sf2'" in src
    assert "SCALES.get((bin, process)) or SCALES.get(process, 1)" in src
    assert 'self.modelBuilder.doSet("POI", "sf1,sf2")' in src


def test_per_category_scale_unknown_category() -> None:
    with pytest.raises(ValueError, match="scale maps unknown categories"):
        FitConfig.model_validate({
            "name": "t", "analysis": "a",
            "categories": [{"name": "c1", "variable": "x"}],
            "model": {
                "pois": {"sf1": {"range": [0, 3]}, "sf2": {"range": [0, 3]}},
                "processes": {"DY": {"scale": {"c1": "sf1", "nope": "sf2"}}},
            },
        })


def _morph_fit(**morph_over) -> dict:
    morph = {"process": "DY", "grid": {"from": 0.95, "to": 1.05, "step": 0.005},
             "categories": ["dm0_pt1"], "range": [0.95, 1.05],
             "scales": {"x": "sqrt", "pt_2": "linear"}}
    morph.update(morph_over)
    return {
        "name": "t", "analysis": "a",
        "categories": [{"name": "dm0_pt1", "variable": "x"},
                       {"name": "dm1_pt1", "variable": "x"}],
        "model": {
            "pois": {"sf0": {"range": [0, 3]}, "sf1": {"range": [0, 3]}},
            "processes": {"DY": {"scale": {"dm0_pt1": "sf0", "dm1_pt1": "sf1"}}},
            "morphs": {
                "tes_dm0": morph,
                "tes_dm1": {"process": "DY", "grid": {"from": 0.99, "to": 1.01, "step": 0.005},
                            "categories": ["dm1_pt1"], "range": [0.95, 1.05],
                            "scales": {"x": "sqrt"}},
            },
        },
    }


def test_morph_schema_and_variations() -> None:
    import math

    from wham.fitconfig import all_pois, grid_points, morph_point_label, morph_variations

    fit = FitConfig.model_validate(_morph_fit())
    assert all_pois(fit) == ["sf0", "sf1", "tes_dm0", "tes_dm1"]

    pts = grid_points(fit.model.morphs["tes_dm0"].grid)  # 'from' alias accepted
    assert len(pts) == 21 and pts[0] == 0.95 and pts[-1] == 1.05

    # per-category: only that category's morph, nominal (f=1) skipped
    v0 = morph_variations(fit, "dm0_pt1")
    assert len(v0) == 20
    assert all(x.target == "columns" and x.processes == ["DY"] for x in v0)
    by_name = {x.name: x for x in v0}
    lab = morph_point_label("tes_dm0", 1.005)
    # per-column laws: sqrt(f) for the mass-like observable, f for pt
    assert by_name[lab].factors["x"] == pytest.approx(math.sqrt(1.005))
    assert by_name[lab].factors["pt_2"] == pytest.approx(1.005)
    assert len(morph_variations(fit, "dm1_pt1")) == 4  # 5-point grid - nominal

    # default scans cover rate POIs then morph POIs
    assert [s.pois for s in fit.resolved_scans()] == [["sf0"], ["sf1"], ["tes_dm0"], ["tes_dm1"]]


def test_morph_needs_scales() -> None:
    with pytest.raises(ValueError):
        FitConfig.model_validate(_morph_fit(scales={}))


def test_morph_unknown_category() -> None:
    with pytest.raises(ValueError, match="morph 'tes_dm0' restricted to unknown categories"):
        FitConfig.model_validate(_morph_fit(categories=["nope"]))


def _morph_workspace_fit(workspace: dict):
    """Load a TES-morph fit against the synthetic workspace + fill its hists.

    POI is named `mu` (not `r`): the CombineHarvester morph backend reserves `r`
    for the default signal strength. cmssw is set so the CH backend can resolve a
    release; only the cmsenv integration test actually runs combine."""
    fit_yaml = workspace["tmp"] / "morphfit.yaml"
    fit_yaml.write_text(
        f"""
name: morphfit
analysis: {workspace["yaml"]}
categories:
  - {{name: SR, variable: met_phi}}
model:
  pois:
    mu: {{init: 1, range: [0, 3]}}
  processes:
    DY: {{scale: "mu"}}
  morphs:
    tes:
      process: DY
      grid: {{from: 0.9, to: 1.1, step: 0.05}}
      categories: [SR]
      scales: {{met_phi: sqrt}}
      range: [0.9, 1.1]
asimov: {{enabled: true}}
combine: {{cmssw: {CMSSW_DIR or "/nonexistent/CMSSW"}}}
scans:
  - {{pois: [tes], points: 8}}
""",
        encoding="utf-8",
    )
    fit, cfg, _ = load_fit_config(fit_yaml)
    samples, _w = discover_samples(cfg)
    skims = ensure_skims(union_analysis(fit, cfg), samples, workers=1)
    hists = {}
    for cat in fit.categories:
        ccfg = category_analysis(fit, cfg, cat)
        hists[cat.name] = fill_all(ccfg, samples, skims, families=fit_families(fit),
                                   only_vars=(cat.variable,), workers=1, sidecars=False)
    return fit, cfg, hists


def test_morph_harvest_export(workspace: dict) -> None:
    """A morph fit generates a CombineHarvester driver (cmsenv backend), not the
    container datacard. Checks the generated harvest.py + the TES grid shapes."""
    fit, cfg, hists = _morph_workspace_fit(workspace)
    templates, _signals, _bkgs = fit_templates(fit, cfg, hists)
    present = {c: set(t) for c, t in templates.items()}
    src = harvest_source(fit, cfg, present, template_parents(fit, cfg))

    # CMSHistFunc morph (not RooMomentMorph), the signal, the TES grid + POI names
    assert "BuildCMSHistFuncFactory" in src
    assert 'SIGNAL = "DY"' in src and 'TES = "tes"' in src
    assert "$BIN/$PROCESS_TES$MASS" in src           # CH signal shape pattern
    for mass in ("0.900", "1.000", "1.100"):
        assert f"'{mass}'" in src, mass
    assert "tid" not in src and "'name': 'mu'" in src  # the bare-POI scale -> rateParam

    # the 5-point grid (0.9..1.1 step .05) incl. the nominal point is in shapes.root
    shapes = workspace["tmp"] / "shapes_check.root"
    write_shapes(shapes, templates)
    with uproot.open(shapes) as f:
        names = {k.split("/")[-1].split(";")[0] for k in f.keys()}
    for tag in ("DY_TES0.900", "DY_TES1.000", "DY_TES1.100"):
        assert tag in names, tag


def test_model_source_components(fit_setup) -> None:
    src = model_source(fit_setup("components")["fit"])
    assert 'self.modelBuilder.doVar("mu[1,0,5]")' in src
    assert 'self.modelBuilder.doVar("alpha[0.7854,0,1.5708]")' in src
    assert "expr::scale_DY_even(\"(@0*pow(cos(@1),2))\", mu, alpha)" in src
    assert "expr::scale_DY_odd(\"(@0*pow(sin(@1),2))\", mu, alpha)" in src
    assert "'DY_even': 'scale_DY_even'" in src
    assert 'self.modelBuilder.doSet("POI", "mu,alpha")' in src


# ------------------------------------------------------------- templates


def test_templates_scale(fit_setup) -> None:
    s = fit_setup("scale")
    templates, signals, backgrounds = fit_templates(s["fit"], s["cfg"], s["hists"])
    assert signals == ["DY"]
    t = templates["SR"]
    assert "data_obs" in t and "DY" in t and "TT" in t
    assert "TT" in backgrounds and "DY" not in backgrounds


def test_templates_components(fit_setup) -> None:
    s = fit_setup("components")
    templates, signals, backgrounds = fit_templates(s["fit"], s["cfg"], s["hists"])
    assert signals == ["DY_even", "DY_odd"]
    t = templates["SR"]
    for name in signals:
        assert t[name].view()["value"].sum() > 0
    assert "DY" not in t and "TT" in backgrounds


def test_templates_two_categories(fit_setup) -> None:
    s = fit_setup("scale", two_cats=True)
    templates, signals, _ = fit_templates(s["fit"], s["cfg"], s["hists"])
    assert set(templates) == {"lowj", "highj"}
    # the categories partition the events: their data_obs sums to the
    # single-category yield (same baseline selection)
    full = fit_setup("scale")
    total = fit_templates(full["fit"], full["cfg"], full["hists"])[0]
    got = sum(templates[c]["data_obs"].view()["value"].sum() for c in templates)
    want = total["SR"]["data_obs"].view()["value"].sum()
    assert got == pytest.approx(want, rel=1e-9)


def test_toy_injection() -> None:
    import hist

    h = hist.Hist(hist.axis.Regular(8, 0, 6.2832, name="x"), storage=hist.storage.Weight())
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 6.2832, 10000)
    h.fill(x=x, weight=np.ones_like(x))
    even, odd = h.copy(), h.copy()

    inject_toy_asymmetry(even, odd, 0.3)
    assert not np.allclose(even.view()["value"], odd.view()["value"])
    assert even.view()["value"].sum() == pytest.approx(h.view()["value"].sum(), rel=1e-12)
    assert odd.view()["value"].sum() == pytest.approx(h.view()["value"].sum(), rel=1e-12)
    assert np.all(even.view()["variance"] >= 0) and np.all(odd.view()["variance"] >= 0)


# ------------------------------------------------------------- writers


def test_shapes_roundtrip(fit_setup, tmp_path: Path) -> None:
    s = fit_setup("components", two_cats=True)
    templates, _, _ = fit_templates(s["fit"], s["cfg"], s["hists"])
    path = tmp_path / "shapes.root"
    write_shapes(path, templates)

    with uproot.open(path) as f:
        for cat, by_name in templates.items():
            for name, h1 in by_name.items():
                rh = f[f"{cat}/{name}"]
                assert np.allclose(rh.values(), h1.view()["value"]), (cat, name)
                assert np.allclose(rh.variances(), h1.view()["variance"]), (cat, name)


@pytest.mark.parametrize("field", ["value", "variance"])
def test_write_shapes_rejects_nonfinite_templates(tmp_path: Path, field: str) -> None:
    import hist

    h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
    h.fill([0.5, 1.5], weight=[1.0, 1.0])
    h.view()[field][1] = np.nan

    with pytest.raises(ValueError, match=rf"cat/signal.*non-finite {field}"):
        write_shapes(tmp_path / "bad.root", {"cat": {"signal": h}})


def test_datacard_multibin(fit_setup, tmp_path: Path) -> None:
    s = fit_setup(
        "scale", two_cats=True,
        extra=("  - {name: norm_tt, effect: rateParam, processes: [TT], range: [0.1, 5]}\n"
               "  - {name: only_low, effect: lnN, processes: [DY], scaleFactor: 1.1,\n"
               "     categories: [lowj]}"),
    )
    fit = s["fit"]
    templates, signals, backgrounds = fit_templates(fit, s["cfg"], s["hists"])
    card = tmp_path / "datacard.txt"
    write_datacard(card, fit=fit, templates=templates, signal_names=signals,
                   background_names=backgrounds,
                   parents=template_parents(fit, s["cfg"]))
    text = card.read_text()
    rows = {l.split()[0]: l.split() for l in text.splitlines() if l.strip()}

    assert "imax 2" in text
    assert "shapes * * shapes.root $CHANNEL/$PROCESS" in text
    assert "* autoMCStats 10" in text

    # one column per (bin, process present there); ids consistent across bins
    bin_cells, proc_rows = None, []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "bin":
            bin_cells = parts[1:]  # last "bin" row = the column row
        if parts and parts[0] == "process":
            proc_rows.append(parts[1:])
    names, ids = proc_rows
    by_col = list(zip(bin_cells, names, ids))
    id_of = {}
    for b, n, i in by_col:
        assert id_of.setdefault(n, i) == i, "process id differs between bins"
    assert id_of["DY"] == "0"
    assert int(id_of["TT"]) > 0

    # rateParam: one line per bin, same (shared) parameter name
    rp = [l for l in text.splitlines() if l.startswith("norm_tt rateParam")]
    assert rp == [
        "norm_tt rateParam lowj TT 1 [0.1,5]",
        "norm_tt rateParam highj TT 1 [0.1,5]",
    ]
    # category-restricted lnN: applies in lowj, '-' in highj
    cells = dict(zip(zip(bin_cells, names), rows["only_low"][2:]))
    assert cells[("lowj", "DY")] == "1.1"
    assert cells[("highj", "DY")] == "-"


def test_datacard_parent_matching(fit_setup, tmp_path: Path) -> None:
    # patterns written against config names must hit sanitized datacard
    # columns ('T+T' -> 'T_T') and component templates via their parent
    s = fit_setup("components")
    fit = s["fit"].model_copy(update={"systematics": [
        SystematicCfg(name="xsec_dy", effect="lnN", processes=["DY"], scaleFactor=1.02),
    ]})
    templates, signals, backgrounds = fit_templates(fit, s["cfg"], s["hists"])
    card = tmp_path / "datacard.txt"
    write_datacard(card, fit=fit, templates=templates, signal_names=signals,
                   background_names=backgrounds,
                   parents=template_parents(fit, s["cfg"]))
    text = card.read_text()
    row = next(l.split() for l in text.splitlines() if l.startswith("xsec_dy"))
    procs = [l.split()[1:] for l in text.splitlines() if l.split()[:1] == ["process"]][0]
    by_proc = dict(zip(procs, row[2:]))
    assert by_proc["DY_even"] == "1.02" and by_proc["DY_odd"] == "1.02"
    assert by_proc["TT"] == "-"


_TT_SHAPE = ('  - {name: tt_shape, effect: shape, processes: [TT],\n'
             '     weight_up: "weight * 2", weight_down: "weight * 0.5"}')


def test_shape_systematic_templates_and_datacard(fit_setup, tmp_path: Path) -> None:
    from wham.fitconfig import shape_affected

    s = fit_setup("scale", extra=_TT_SHAPE)
    fit = s["fit"]
    templates, signals, backgrounds = fit_templates(fit, s["cfg"], s["hists"])
    t = templates["SR"]

    assert "TT_tt_shapeUp" in t and "TT_tt_shapeDown" in t
    assert t["TT_tt_shapeUp"].view()["value"].sum() == pytest.approx(
        2 * t["TT"].view()["value"].sum(), rel=1e-9)
    assert "DY_tt_shapeUp" not in t  # unmatched process: no varied template
    if "QCD" in t:  # the varied subtraction propagates into the QCD estimate
        assert "QCD_tt_shapeUp" in t

    card = tmp_path / "datacard.txt"
    syst = next(x for x in fit.systematics if x.effect == "shape")
    write_datacard(card, fit=fit, templates=templates, signal_names=signals,
                   background_names=backgrounds,
                   parents=template_parents(fit, s["cfg"]),
                   shape_affected_map={syst.name: shape_affected(fit, s["cfg"], syst)})
    text = card.read_text()
    row = next(l.split() for l in text.splitlines() if l.startswith("tt_shape"))
    assert row[1] == "shape"
    procs = [l.split()[1:] for l in text.splitlines() if l.split()[:1] == ["process"]][0]
    by_proc = dict(zip(procs, row[2:]))
    assert by_proc["TT"] == "1" and by_proc["DY"] == "-"

    # shapes roundtrip includes the Up/Down histograms
    shapes_path = tmp_path / "shapes.root"
    write_shapes(shapes_path, templates)
    with uproot.open(shapes_path) as f:
        assert np.allclose(f["SR/TT_tt_shapeUp"].values(),
                           t["TT_tt_shapeUp"].view()["value"])


def test_shape_resolution_errors(workspace: dict) -> None:
    fit_yaml = workspace["tmp"] / "bad_shape.yaml"
    fit_yaml.write_text(
        f"""
name: bad_shape
analysis: {workspace["yaml"]}
{_ONE_CATEGORY}
{_SCALE_MODEL}
systematics:
  - {{name: ff_shape, effect: shape, processes: [QCD],
     weight_up: "pt_2 / 50", weight_down: "pt_2 / 200"}}
""",
        encoding="utf-8",
    )
    # the synthetic analysis uses qcd.method=abcd
    with pytest.raises(ValueError, match="requires qcd.method=ff"):
        load_fit_config(fit_yaml)


def test_export_key_sensitivity(fit_setup) -> None:
    s0 = fit_setup("components", toy=0.0)
    s3 = fit_setup("components", toy=0.3)
    keys = {("SR", "datamc", "met_phi"): "k1", ("SR", "fitcp", "met_phi"): "k2"}
    assert export_key(s0["fit"], keys) != export_key(s3["fit"], keys)
    assert export_key(s0["fit"], keys) != export_key(
        s0["fit"], {**keys, ("SR", "datamc", "met_phi"): "OTHER"})
    assert export_key(s0["fit"], keys) == export_key(s0["fit"], dict(keys))


def test_datacard_only_skip_logic(fit_setup) -> None:
    s = fit_setup("scale")
    keys = {("SR", "datamc", "met_phi"): "k"}
    fitdir = run_fit(s["fit"], s["cfg"], s["hists"], keys, datacard_only=True)
    card = fitdir / "datacard.txt"
    assert (fitdir / "whammodel.py").is_file()
    assert (fitdir / "fitconfig.yaml").is_file()
    mtime = card.stat().st_mtime_ns

    # unchanged inputs -> export skipped
    run_fit(s["fit"], s["cfg"], s["hists"], keys, datacard_only=True)
    assert card.stat().st_mtime_ns == mtime

    # changed hist key -> re-export
    run_fit(s["fit"], s["cfg"], s["hists"],
            {("SR", "datamc", "met_phi"): "DIFFERENT"}, datacard_only=True)
    assert card.stat().st_mtime_ns != mtime

    # a datacard-only run must NOT mark the combine products fresh: run_key is
    # only stamped by a completed full run, so a later full run reruns
    # text2workspace/fit/scans even though old product files exist on disk
    manifest = json.loads((fitdir / "manifest.json").read_text())
    assert "run_key" not in manifest
    # a full-run stamp survives datacard-only while the datacard is unchanged...
    key = manifest["export_key"]
    (fitdir / "manifest.json").write_text(json.dumps({"export_key": key, "run_key": key}))
    run_fit(s["fit"], s["cfg"], s["hists"],
            {("SR", "datamc", "met_phi"): "DIFFERENT"}, datacard_only=True)
    assert json.loads((fitdir / "manifest.json").read_text()).get("run_key") == key
    # ...and is dropped when the datacard regenerates (stale workspace)
    run_fit(s["fit"], s["cfg"], s["hists"],
            {("SR", "datamc", "met_phi"): "DIFFERENT2"}, datacard_only=True)
    assert "run_key" not in json.loads((fitdir / "manifest.json").read_text())


# ------------------------------------------------------------- container


@pytest.mark.combine
@pytest.mark.skipif(
    shutil.which("apptainer") is None or not COMBINE_IMAGE.exists(),
    reason="apptainer or combine image unavailable",
)
def test_full_rate_fit_in_container(fit_setup) -> None:
    s = fit_setup("scale", extra=_TT_SHAPE)
    fit = s["fit"]
    fitdir = run_fit(fit, s["cfg"], s["hists"], {("SR", "datamc", "met_phi"): "k"})

    assert (fitdir / "workspace.root").is_file()
    fd = uproot.open(fitdir / f"fitDiagnostics.{fit.name}.root")
    r_fit = fd["tree_fit_sb"]["r"].array()[0]
    assert r_fit == pytest.approx(1.0, abs=0.05)  # Asimov with r=1 injected

    result = json.loads((fitdir / "fitresult.json").read_text())
    assert result["params"]["r"]["value"] == pytest.approx(1.0, abs=0.05)
    assert result["params"]["r"]["error"] > 0
    assert "r" in result["impacts"]
    # the shape nuisance is fitted: Asimov pull ~ 0, constrained ~ 1
    assert result["params"]["tt_shape"]["value"] == pytest.approx(0.0, abs=0.2)
    assert 0.05 < result["params"]["tt_shape"]["error"] <= 1.2

    scan = uproot.open(fitdir / scan_file(fit, fit.resolved_scans()[0]))
    assert len(scan["limit"]["r"].array()) >= 10


@pytest.mark.combine
@pytest.mark.skipif(CMSSW_DIR is None, reason="no CMSSW with CombineHarvester available")
def test_morph_fit_cmsenv(workspace: dict) -> None:
    """Full TES-morph path through the CombineHarvester/cmsenv backend: the
    harvester builds the datacard with a CMSHistFunc morph (autoMCStats on),
    text2workspace + FitDiagnostics run, and the morph POI is floated/reported
    (physics recovery is covered separately; synthetic samples are featureless)."""
    fit, cfg, hists = _morph_workspace_fit(workspace)
    fitdir = run_fit(fit, cfg, hists, {("SR", "datamc", "met_phi"): "k"})

    # the CH backend writes its own datacard + shapes, then the workspace
    assert (fitdir / "harvest.py").is_file()
    assert (fitdir / "ch_shapes.root").is_file()
    assert (fitdir / "workspace.root").is_file()
    card = (fitdir / "datacard.txt").read_text()
    assert "autoMCStats" in card                           # BBB on (one-to-one postfit)

    result = json.loads((fitdir / "fitresult.json").read_text())
    assert "tes" in result["params"]                       # morph POI promoted + fitted
    assert result["params"]["tes"]["error"] >= 0
    assert result["params"]["mu"]["value"] == pytest.approx(1.0, abs=0.1)  # rate constrained
    assert (fitdir / scan_file(fit, fit.resolved_scans()[0])).is_file()   # tes scan ran


@pytest.mark.combine
@pytest.mark.skipif(
    shutil.which("apptainer") is None or not COMBINE_IMAGE.exists(),
    reason="apptainer or combine image unavailable",
)
def test_multipoi_two_category_recovery_in_container(fit_setup, workspace: dict) -> None:
    """3 POIs (mu, alpha + a TT rate param via model), 2 categories, toy
    asymmetry: the Asimov fit must recover the injected truth."""
    s = fit_setup(
        "components", toy=0.3, two_cats=True,
        extra="""
  - {name: norm_tt_free, effect: rateParam, processes: [TT], range: [0.1, 5]}
""",
    )
    # inject alpha=0.4 truth and add a 2D scan
    fit = s["fit"].model_copy(update={
        "asimov": s["fit"].asimov.model_copy(update={"parameters": {"alpha": 0.4}}),
        "scans": [
            s["fit"].scans[0],
            s["fit"].scans[0].model_copy(update={"pois": ["alpha", "mu"], "points": 8,
                                                 "range": None}),
        ],
    })
    keys = {(c.name, f, "met_phi"): f"k_{c.name}_{f}"
            for c in fit.categories for f in ("datamc", "fitcp")}
    fitdir = run_fit(fit, s["cfg"], s["hists"], keys)

    result = json.loads((fitdir / "fitresult.json").read_text())
    assert result["params"]["alpha"]["value"] == pytest.approx(0.4, abs=0.05)
    assert result["params"]["mu"]["value"] == pytest.approx(1.0, abs=0.05)
    assert result["params"]["norm_tt_free"]["value"] == pytest.approx(1.0, abs=0.1)
    assert result["impacts"]["alpha"]  # covariance impacts present

    # both scans produced output
    for scan in fit.resolved_scans():
        assert (fitdir / scan_file(fit, scan)).is_file(), scan.pois

    # datacard is genuinely multi-bin
    text = (fitdir / "datacard.txt").read_text()
    assert "imax 2" in text


def test_shape_systematic_scales_schema() -> None:
    from wham.fitconfig import SystematicCfg

    ok = SystematicCfg(name="jtf", effect="shape", processes=["DY_jfake"],
                       scales={"m_vis": "sqrt", "pt_2": "linear"}, shift=0.10)
    assert ok.shift == 0.10

    with pytest.raises(ValueError, match="either weight_up/weight_down or scales"):
        SystematicCfg(name="bad", effect="shape", processes=["DY"])  # neither flavor
    with pytest.raises(ValueError, match="either weight_up/weight_down or scales"):
        SystematicCfg(name="bad", effect="shape", processes=["DY"],
                      weight_up="w*2", weight_down="w*0.5",
                      scales={"pt_2": "linear"}, shift=0.03)  # both flavors
    with pytest.raises(ValueError, match="shape-only"):
        SystematicCfg(name="bad", effect="lnN", processes=["DY"], scaleFactor=1.1,
                      scales={"pt_2": "linear"}, shift=0.03)


# ------------------------------------------------- control categories (CR)


def _control_yaml(workspace: dict, *, clash: bool = False) -> Path:
    """Second synthetic analysis (same data_dir, own name/selection/processes)
    for cross-analysis control-category tests. clash=True names a process like
    the model process to trigger the collision rejection."""
    dy = "DY" if clash else "CDY"
    path = workspace["tmp"] / f"control{'_clash' if clash else ''}.yaml"
    path.write_text(
        f"""
name: test_control{"_clash" if clash else ""}
lumi: 1000.0
data_dir: {workspace["data_dir"]}
output_dir: {workspace["tmp"] / "plots_control"}
selection: "pt_1 > 30"
trigger: "trg == 1"
weight: weight
processes:
  CQCD: {{kind: qcd, color: "tab:olive"}}
  CTT:  {{samples: ["TT_*"], color: "tab:purple"}}
  {dy}:  {{samples: ["DY_*"], color: "tab:orange"}}
  data: {{kind: data, samples: ["Muon_*"], color: black}}
qcd:
  method: abcd
  os: "os == 1"
  iso: "id_2 >= 5"
  antiiso: "id_2 > 1 & id_2 < 5"
sample_params:
  TT_test: {{xs: 100.0, eff: 50000}}
  DY_test: {{xs: 200.0, eff: 80000, filter_efficiency: 0.5}}
variables:
  m_vis: {{bins: 1, range: [50, 150]}}
plots:
  datamc: [m_vis]
""",
        encoding="utf-8",
    )
    return path


def _fill_by_analysis(fit, cat_cfgs) -> dict:
    """Mirror the cli fill driver: group categories by analysis, one
    skim+fill pass per analysis with its own samples."""
    groups: dict[int, tuple] = {}
    for cat in fit.categories:
        acfg = cat_cfgs[cat.name]
        groups.setdefault(id(acfg), (acfg, []))[1].append(cat)
    hists = {}
    for acfg, cats in groups.values():
        samples, _ = discover_samples(acfg)
        skims = ensure_skims(union_analysis(fit, acfg, cats), samples, workers=1)
        for cat in cats:
            ccfg = category_analysis(fit, acfg, cat)
            hists[cat.name] = fill_all(ccfg, samples, skims,
                                       families=fit_families(fit),
                                       only_vars=(cat.variable,), workers=1,
                                       sidecars=False)
    return hists


def test_control_category_datacard(workspace: dict, tmp_path: Path) -> None:
    """Container path: a control category from a second analysis becomes a
    backgrounds-only bin; scoped systematics stay out of it, a shared-name
    nuisance spans both analyses' DY columns."""
    from wham.fitconfig import union_template_parents

    control = _control_yaml(workspace)
    fit_yaml = workspace["tmp"] / "crfit.yaml"
    fit_yaml.write_text(
        f"""
name: crfit
analysis: {workspace["yaml"]}
categories:
  - {{name: SR, variable: met_phi}}
  - {{name: CR, variable: m_vis, analysis: {control}}}
{_SCALE_MODEL}
systematics:
  - {{name: lumi, effect: lnN, processes: [TT, DY], scaleFactor: 1.025,
      categories: [SR]}}
  - {{name: xsec_dy, effect: lnN, processes: [DY, CDY], scaleFactor: 1.02}}
""",
        encoding="utf-8",
    )
    fit, cfg, cat_cfgs = load_fit_config(fit_yaml)
    assert cat_cfgs["SR"] is cfg
    assert cat_cfgs["CR"].name == "test_control"

    hists = _fill_by_analysis(fit, cat_cfgs)
    templates, signals, backgrounds = fit_templates(fit, cfg, hists,
                                                    cat_cfgs=cat_cfgs)
    assert signals == ["DY"]
    t = templates["CR"]
    assert "data_obs" in t and "CTT" in t and "CDY" in t
    assert "DY" not in t and "TT" not in t  # main processes stay out of the CR
    assert t["CDY"].axes[0].size == 1       # single counting bin
    for name in ("CTT", "CDY"):
        assert name in backgrounds

    card = tmp_path / "datacard.txt"
    write_datacard(card, fit=fit, templates=templates, signal_names=signals,
                   background_names=backgrounds,
                   parents=union_template_parents(fit, cfg, cat_cfgs))
    text = card.read_text()
    assert "imax 2" in text
    bin_cells, proc_rows = None, []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "bin":
            bin_cells = parts[1:]
        if parts and parts[0] == "process":
            proc_rows.append(parts[1:])
    names, ids = proc_rows
    cols = list(zip(bin_cells, names, ids))
    assert all(int(i) > 0 for b, n, i in cols if b == "CR")  # backgrounds-only
    rows = {parts[0]: parts for parts in map(str.split, text.splitlines()) if parts}
    lumi = dict(zip(zip(bin_cells, names), rows["lumi"][2:]))
    assert lumi[("SR", "DY")] == "1.025"
    assert all(v == "-" for (b, _n), v in lumi.items() if b == "CR")
    xsec = dict(zip(zip(bin_cells, names), rows["xsec_dy"][2:]))
    assert xsec[("SR", "DY")] == "1.02" and xsec[("CR", "CDY")] == "1.02"
    assert all(v == "-" for (b, n), v in xsec.items() if n not in ("DY", "CDY"))


def _split_fit(workspace: dict):
    """Morph fit + control category: the CH split-cards path."""
    control = _control_yaml(workspace)
    fit_yaml = workspace["tmp"] / "splitfit.yaml"
    fit_yaml.write_text(
        f"""
name: splitfit
analysis: {workspace["yaml"]}
categories:
  - {{name: SR, variable: met_phi}}
  - {{name: CR, variable: m_vis, analysis: {control}}}
model:
  pois:
    mu: {{init: 1, range: [0, 3]}}
  processes:
    DY: {{scale: "mu"}}
  morphs:
    tes:
      process: DY
      grid: {{from: 0.9, to: 1.1, step: 0.05}}
      categories: [SR]
      scales: {{met_phi: sqrt}}
      range: [0.9, 1.1]
asimov: {{enabled: true}}
combine: {{cmssw: {CMSSW_DIR or "/nonexistent/CMSSW"}}}
scans:
  - {{pois: [tes], points: 8}}
systematics:
  - {{name: lumi, effect: lnN, processes: [TT, DY], scaleFactor: 1.025,
      categories: [SR]}}
  - {{name: xsec_dy, effect: lnN, processes: [DY, CDY], scaleFactor: 1.02}}
""",
        encoding="utf-8",
    )
    return load_fit_config(fit_yaml)


def test_control_category_ch_split(workspace: dict, tmp_path: Path) -> None:
    """CH/morph fit with a control bin: harvest covers the main bins only
    (the 1-bin CR is exempt from the shared-binning rule), the control card is
    backgrounds-only WITH autoMCStats, and combineCards merges the two."""
    from wham.fitconfig import union_template_parents

    fit, cfg, cat_cfgs = _split_fit(workspace)  # loads: binning check passed
    hists = _fill_by_analysis(fit, cat_cfgs)
    templates, signals, backgrounds = fit_templates(fit, cfg, hists,
                                                    cat_cfgs=cat_cfgs)
    # morph grid templates exist in the main bin only
    assert "DY_TES0.900" in templates["SR"] and "DY_TES0.900" not in templates["CR"]

    parents = union_template_parents(fit, cfg, cat_cfgs)
    present = {c: set(t) for c, t in templates.items()}
    src = harvest_source(fit, cfg, present, parents, datacard=HARVEST_CARD_FILE)
    assert "BINS = [[0, 'SR']]" in src          # control bin stays out of CH
    assert "CDY" not in src and "'CR'" not in src
    assert f'cb.WriteDatacard("{HARVEST_CARD_FILE}"' in src

    card = tmp_path / CONTROL_CARD_FILE
    write_datacard(card, fit=fit, templates=templates, signal_names=[],
                   background_names=backgrounds, parents=parents,
                   categories=["CR"])
    text = card.read_text()
    assert "imax 1" in text
    assert "* autoMCStats 10" in text           # plain-TH1 bin keeps BBB
    assert " DY " not in text                   # backgrounds-only: no model column
    row = next(parts for parts in map(str.split, text.splitlines())
               if parts and parts[0] == "xsec_dy")
    assert "1.02" in row and "lumi" not in text.split()  # scoped syst absent

    assert merge_command(fit) == (
        f"combineCards.py .={HARVEST_CARD_FILE} .={CONTROL_CARD_FILE} "
        f"> {DATACARD_FILE}")


def test_control_category_rejections(workspace: dict) -> None:
    control = _control_yaml(workspace)

    # morph acting in a control bin
    with pytest.raises(ValueError, match="morphs live in the main analysis"):
        FitConfig.model_validate(_fit_dict(categories=[
            {"name": "SR", "variable": "x"},
            {"name": "CR", "variable": "x", "analysis": "other.yaml"},
        ], model={
            "pois": {"sf": {"range": [0, 3]}},
            "processes": {"DY": {"scale": "sf"}},
            "morphs": {"tes": {"process": "DY",
                               "grid": {"from": 0.9, "to": 1.1, "step": 0.05},
                               "categories": ["CR"], "range": [0.9, 1.1],
                               "scales": {"x": "sqrt"}}},
        }))

    # scale map targeting a control bin
    with pytest.raises(ValueError, match="scale maps control categories"):
        FitConfig.model_validate(_fit_dict(categories=[
            {"name": "SR", "variable": "x"},
            {"name": "CR", "variable": "x", "analysis": "other.yaml"},
        ], model={
            "pois": {"sf": {"range": [0, 3]}},
            "processes": {"DY": {"scale": {"SR": "sf", "CR": "sf"}}},
        }))

    # every category a control category
    with pytest.raises(ValueError, match="at least one main-analysis category"):
        FitConfig.model_validate(_fit_dict(categories=[
            {"name": "CR", "variable": "x", "analysis": "other.yaml"},
        ]))

    def _load(cats: str, extra: str = "") -> None:
        fit_yaml = workspace["tmp"] / "bad_cr.yaml"
        fit_yaml.write_text(
            f"""
name: bad_cr
analysis: {workspace["yaml"]}
categories:
{cats}
{_SCALE_MODEL}
{extra}""",
            encoding="utf-8",
        )
        load_fit_config(fit_yaml)

    # control variable must exist in the CONTROL analysis
    with pytest.raises(ValueError, match="not defined in .*control"):
        _load(f"""  - {{name: SR, variable: met_phi}}
  - {{name: CR, variable: met_phi, analysis: {control}}}""")

    # control process colliding with a model process name
    with pytest.raises(ValueError, match="collide with model process"):
        _load(f"""  - {{name: SR, variable: met_phi}}
  - {{name: CR, variable: m_vis, analysis: {_control_yaml(workspace, clash=True)}}}""")

    # an unscoped shape systematic must resolve in the control analysis too
    with pytest.raises(ValueError, match="matches no kind=mc process"):
        _load(f"""  - {{name: SR, variable: met_phi}}
  - {{name: CR, variable: m_vis, analysis: {control}}}""",
              extra="""systematics:
  - {name: tt_shape, effect: shape, processes: [TT],
     weight_up: "weight * 2", weight_down: "weight * 0.5"}
""")

    # ... but a main-scoped one is fine
    _load(f"""  - {{name: SR, variable: met_phi}}
  - {{name: CR, variable: m_vis, analysis: {control}}}""",
          extra="""systematics:
  - {name: tt_shape, effect: shape, processes: [TT], categories: [SR],
     weight_up: "weight * 2", weight_down: "weight * 0.5"}
""")


@pytest.mark.combine
@pytest.mark.skipif(CMSSW_DIR is None, reason="no CMSSW with CombineHarvester available")
def test_control_ch_split_cmsenv(workspace: dict) -> None:
    """Split-cards export end-to-end under cmsenv: harvest writes the morph
    card, WHAM the control card, combineCards.py merges them with the bin
    names preserved."""
    fit, cfg, cat_cfgs = _split_fit(workspace)
    hists = _fill_by_analysis(fit, cat_cfgs)
    keys = {("SR", "datamc", "met_phi"): "k", ("CR", "datamc", "m_vis"): "k2"}
    fitdir = run_fit(fit, cfg, hists, keys, cat_cfgs=cat_cfgs, datacard_only=True)

    assert (fitdir / CONTROL_CARD_FILE).is_file()
    assert (fitdir / HARVEST_CARD_FILE).is_file()
    merged = (fitdir / DATACARD_FILE).read_text()
    assert "CR" in merged and "SR" in merged     # bin names preserved by '.='
    assert "CDY" in merged and "autoMCStats" in merged
    assert "morph" in merged                     # CH morph shapes lines intact


def test_shape_variations_column_shift(workspace: dict) -> None:
    """A scales+shift shape systematic resolves into an up/down pair of
    columns-target fill variations with the per-law factors."""
    import math

    from wham.fitconfig import shape_variations

    fit_yaml = workspace["tmp"] / "esfit.yaml"
    fit_yaml.write_text(
        f"""
name: esfit
analysis: {workspace["yaml"]}
categories:
  - {{name: SR, variable: m_vis}}
model:
  pois:
    mu: {{init: 1, range: [0, 3]}}
  processes:
    DY: {{scale: "mu"}}
systematics:
  - {{name: jtf, effect: shape, processes: [TT],
      scales: {{m_vis: sqrt, pt_2: linear}}, shift: 0.10}}
""",
        encoding="utf-8",
    )
    fit, cfg, _ = load_fit_config(fit_yaml)
    vars_ = shape_variations(fit, cfg)
    assert [v.name for v in vars_] == ["jtf_up", "jtf_down"]
    assert all(v.target == "columns" and v.processes == ["TT"] for v in vars_)
    up, down = vars_
    assert up.factors["pt_2"] == pytest.approx(1.10)
    assert up.factors["m_vis"] == pytest.approx(math.sqrt(1.10))
    assert down.factors["pt_2"] == pytest.approx(0.90)
    assert down.factors["m_vis"] == pytest.approx(math.sqrt(0.90))
    # slice labels line up with the datacard's {name}Up/Down template suffixes
    assert up.slice_labels() == ["jtf_up"] and down.slice_labels() == ["jtf_down"]


def test_tau_sf_fit_configs_load() -> None:
    """Repo guard: the 10 tau_sf fit configs stay loadable, with single top
    split out of tt (ST/ST_lfake/ST_jfake + xsec_st, ST in the zmm stack)."""
    repo = Path(__file__).resolve().parents[3]
    paths = sorted(repo.glob("Configurations/tau_sf/*/tau_sf_dm*.yaml"))
    if len(paths) != 10:
        pytest.skip("tau_sf configs not present")
    for path in paths:
        fit, cfg, cat_cfgs = load_fit_config(path)
        assert {"ST", "ST_lfake", "ST_jfake"} <= set(cfg.processes)
        for parent in ("tt", "tt_lfake", "tt_jfake"):
            assert cfg.processes[parent].samples == ["TTto*"]
        assert any(s.name == "xsec_st" for s in fit.systematics)
        assert "ST" in cat_cfgs["zmm"].processes
