from __future__ import annotations

import pytest

from wham.config import discover_samples, load_config, sample_scale


def test_load_and_discover(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    assert cfg.name == "test_analysis"
    assert cfg.stack_order() == ["QCD", "TT", "DY"]
    assert cfg.data_process() == "data"
    assert cfg.qcd_process() == "QCD"

    samples, warnings = discover_samples(cfg)
    assert [s.name for s in samples] == ["DY_test", "Muon_test", "TT_test"]
    assert not warnings
    by_name = {s.name: s for s in samples}
    assert by_name["Muon_test"].kind == "data"
    assert sample_scale(by_name["Muon_test"], cfg.lumi) == 1.0
    # lumi * xs * filter_eff / eff = 1000 * 200 * 0.5 / 80000
    assert sample_scale(by_name["DY_test"], cfg.lumi) == pytest.approx(1.25)


def test_required_columns(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    cols = cfg.required_columns()
    assert {"pt_1", "pt_2", "eta_1", "m_vis", "met_phi", "os", "id_2",
            "trg", "weight", "wt_cp_sm", "wt_cp_ps"} <= cols


def test_datamc_metrics_are_optional_render_config(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    assert cfg.plots.datamc_blind is False
    assert cfg.plots.datamc_metrics == []
    before = cfg.required_columns()

    plots = cfg.plots.model_copy(
        update={
            "datamc_blind": True,
            "datamc_metrics": ["normalization", "shape_chi2", "max_significance"],
        }
    )
    cfg = cfg.model_copy(update={"plots": plots})

    assert cfg.plots.datamc_blind is True
    assert cfg.plots.datamc_metrics == [
        "normalization",
        "shape_chi2",
        "max_significance",
    ]
    assert cfg.required_columns() == before


def _fake_factor_models(tmp_path):
    root = tmp_path / "models"
    for process in ("QCD", "Wjets", "ttbarMC"):
        mdir = root / f"model_mt_{process}"
        mdir.mkdir(parents=True)
        (mdir / "model.json").write_text("{}", encoding="utf-8")
    return root


def test_ff_estimate_validation_and_model_process_derivation(workspace: dict) -> None:
    base = workspace["yaml"].read_text()
    models = _fake_factor_models(workspace["tmp"])
    cfg_path = workspace["tmp"] / "ff_estimate.yaml"
    cfg_path.write_text(base.replace(
        "processes:\n",
        "processes:\n  jet_fakes: {kind: ff, color: tab:olive}\n",
    ).replace(
        "qcd:\n",
        """
fake_factors:
  models: {models}
  channel: mt
  era_label: 0
  estimate:
    enabled: true
    output_process: jet_fakes
    components: {{QCD: true, Wjets: true, ttbar: false}}
    application_region:
      selection: "os == 1"
      pass: "id_2 >= 5"
      fail: "id_2 > 1 & id_2 < 5"
    fake_cut: "id_2 < 5"
qcd:
""".format(models=models),
    ).replace(
        'DY:   {samples: ["DY_*"], color: "tab:orange"}',
        'DY:   {samples: ["DY_*"], color: "tab:orange", ff_component: Wjets}',
    ))

    cfg = load_config(cfg_path)
    assert cfg.fake_factors.processes == ["QCD", "Wjets"]


def test_ff_estimate_requires_qcd_for_w_or_top(workspace: dict) -> None:
    base = workspace["yaml"].read_text()
    models = _fake_factor_models(workspace["tmp"])
    cfg_path = workspace["tmp"] / "bad_ff_estimate.yaml"
    cfg_path.write_text(base + f"""
fake_factors:
  models: {models}
  channel: mt
  era_label: 0
  estimate:
    enabled: true
    output_process: jet_fakes
    components: {{QCD: false, Wjets: true, ttbar: false}}
    application_region:
      selection: "os == 1"
      pass: "id_2 >= 5"
      fail: "id_2 > 1 & id_2 < 5"
""")

    with pytest.raises(ValueError, match="requires QCD=true"):
        load_config(cfg_path)


def test_ff_estimate_output_process_must_be_kind_ff(workspace: dict) -> None:
    base = workspace["yaml"].read_text()
    models = _fake_factor_models(workspace["tmp"])
    cfg_path = workspace["tmp"] / "bad_ff_output.yaml"
    cfg_path.write_text(base + f"""
fake_factors:
  models: {models}
  channel: mt
  era_label: 0
  estimate:
    enabled: true
    output_process: QCD
    components: {{QCD: true, Wjets: false, ttbar: false}}
    application_region:
      selection: "os == 1"
      pass: "id_2 >= 5"
      fail: "id_2 > 1 & id_2 < 5"
""")

    with pytest.raises(ValueError, match="kind=ff"):
        load_config(cfg_path)


@pytest.mark.parametrize(
    "patch, match",
    [
        ("selection: 'pt_1 >'", "invalid expression in 'selection'"),
        ("selection: 'sin(pt_1) > 1'", "only abs"),
        ("plots:\n  datamc: [nonexistent_var]", "not defined in 'variables'"),
        ("plots:\n  ffcheck: [m_vis]", "requires qcd.method=ff"),
        ("lumi: -5", "greater than 0"),
        ("typo_field: 1", "extra"),
    ],
)
def test_config_rejections(workspace: dict, patch: str, match: str) -> None:
    yaml_path = workspace["yaml"]
    base = yaml_path.read_text()
    bad = workspace["tmp"] / "bad.yaml"
    if patch.startswith(("selection:", "lumi:")):
        key = patch.split(":")[0]
        lines = [l for l in base.splitlines() if not l.startswith(f"{key}:")]
        bad.write_text("\n".join(lines) + f"\n{patch}\n")
    elif patch.startswith("plots:"):
        head = base.split("plots:")[0]
        bad.write_text(head + patch + "\n")
    else:
        bad.write_text(base + f"\n{patch}\n")
    with pytest.raises(Exception, match=match):
        load_config(bad)


def test_ambiguous_pattern_rejected(workspace: dict) -> None:
    # overlapping patterns without per-process cuts double count -> error
    base = workspace["yaml"].read_text()
    bad = workspace["tmp"] / "ambig.yaml"
    bad.write_text(base.replace('TT:   {samples: ["TT_*"]', 'TT:   {samples: ["*_test"]'))
    cfg = load_config(bad)
    with pytest.raises(ValueError, match="requires a disjoint 'cut'"):
        discover_samples(cfg)


def test_unmatched_pattern_warns(workspace: dict) -> None:
    base = workspace["yaml"].read_text()
    cfg_path = workspace["tmp"] / "warn.yaml"
    cfg_path.write_text(base.replace('["DY_*"]', '["DY_*", "Zprime_*"]'))
    _, warnings = discover_samples(load_config(cfg_path))
    assert any("Zprime_*" in w for w in warnings)


def test_unroll_validation(workspace: dict) -> None:
    from wham.config import VariableCfg

    with pytest.raises(ValueError, match="omit bins"):
        VariableCfg(unroll=("a", "b"), bins=10, range=(0, 1))
    with pytest.raises(ValueError, match="needs 'bins'"):
        VariableCfg()
    assert VariableCfg(unroll=("a", "b")).bins is None

    base = workspace["yaml"].read_text()
    bad = workspace["tmp"] / "unroll_bad.yaml"
    bad.write_text(base.replace(
        "  met_phi:", "  unr: {unroll: [m_vis, nonexistent]}\n  met_phi:"))
    with pytest.raises(ValueError, match="undefined variable"):
        load_config(bad)

    good = workspace["tmp"] / "unroll_good.yaml"
    good.write_text(base.replace(
        "  met_phi:", "  unr: {unroll: [m_vis, pt_1]}\n  met_phi:"))
    cfg = load_config(good)
    # an unrolled variable reads both source columns
    assert cfg.columns_of_var("unr") == {"m_vis", "pt_1"}


def test_variable_binning_validation() -> None:
    from wham.config import VariableCfg

    VariableCfg(bins=10, range=(0, 1))            # count + range
    v = VariableCfg(bins=[0, 1, 5, 20])           # explicit edges
    assert v.span() == (0, 20)
    with pytest.raises(ValueError, match="needs an explicit 'range'"):
        VariableCfg(bins=10)
    with pytest.raises(ValueError, match="must be omitted"):
        VariableCfg(bins=[0, 1], range=(0, 1))
    with pytest.raises(ValueError, match="strictly increasing"):
        VariableCfg(bins=[0, 5, 5, 10])
    with pytest.raises(ValueError, match="at least 2"):
        VariableCfg(bins=[3])
    with pytest.raises(ValueError, match="increasing"):
        VariableCfg(bins=5, range=(1, 0))
    with pytest.raises(ValueError, match="plain column name"):
        VariableCfg(bins=5, range=(0, 1), column="pt_1 + pt_2")


def test_column_alias_resolution(workspace: dict) -> None:
    from wham.config import VariableCfg

    cfg = load_config(workspace["yaml"])
    variables = dict(cfg.variables)
    variables["m_vis_coarse"] = VariableCfg(column="m_vis", bins=[0.0, 50.0, 250.0])
    plots = cfg.plots.model_copy(update={"datamc": ["m_vis_coarse"]})
    cfg = cfg.model_copy(update={"variables": variables, "plots": plots})

    assert cfg.column_of("m_vis_coarse") == "m_vis"
    assert cfg.column_of("pt_1") == "pt_1"
    cols = cfg.required_columns()
    assert "m_vis" in cols
    assert "m_vis_coarse" not in cols  # aliases are not real file columns


def test_params_side_file(workspace: dict) -> None:
    base = workspace["yaml"].read_text()
    lines = base.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("sample_params:"))
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith(" "))
    side_dir = workspace["tmp"] / "side"
    side_dir.mkdir()
    (side_dir / "analysis.yaml").write_text("\n".join(lines[:start] + lines[end:]))
    (side_dir / "params.yaml").write_text(
        "lumi: 999\nTT_test:\n  xs: 100.0\n  eff: 50000\nDY_test:\n  xs: 200.0\n  eff: 80000\n"
    )
    cfg = load_config(side_dir / "analysis.yaml")
    assert cfg.lumi == 1000.0  # analysis yaml wins, side-file lumi ignored
    assert cfg.sample_params["TT_test"].xs == 100.0


def test_shared_sample_needs_cuts(workspace: dict) -> None:
    """One sample may feed several processes only when every one has a cut
    (a genmatch split); without cuts the overlap double counts."""
    from wham.config import discover_samples, load_config

    cfg = load_config(workspace["yaml"])
    procs = dict(cfg.processes)
    procs["DY_lo"] = procs["DY"].model_copy(update={"cut": "id_2 < 4"})
    procs["DY_hi"] = procs["DY"].model_copy(update={"cut": "id_2 >= 4"})
    del procs["DY"]

    split = cfg.model_copy(update={"processes": procs})
    samples, _ = discover_samples(split)
    dy = [s for s in samples if s.name == "DY_test"]
    assert sorted(s.process for s in dy) == ["DY_hi", "DY_lo"]
    assert all(s.kind == "mc" for s in dy)

    # same overlap without a cut on one side -> config error
    procs["DY_hi"] = procs["DY_hi"].model_copy(update={"cut": None})
    with pytest.raises(ValueError, match="requires a disjoint 'cut'"):
        discover_samples(cfg.model_copy(update={"processes": procs}))
