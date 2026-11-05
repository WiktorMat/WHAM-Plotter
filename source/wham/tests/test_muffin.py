from __future__ import annotations

import numpy as np
import pytest

from wham.config import discover_samples, load_config
from wham.fill import build_fill_spec, fill_sample, merge_hists
from wham.qcd import estimate_qcd
from wham.skim import ensure_skims

xgb = pytest.importorskip("xgboost")

FEATURES = ["pt", "jpt_pt", "met_var_qcd", "era_label", "is_lead_tau"]


@pytest.fixture
def models_dir(tmp_path):
    """A tiny but real multi:softprob model in the BDTFFModel layout."""
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, (500, len(FEATURES)))
    y = rng.integers(0, 4, 500)
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURES)
    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": 4, "max_depth": 2}, dtrain,
        num_boost_round=2,
    )
    mdir = tmp_path / "BDTFFModel" / "model_mt_QCD"
    mdir.mkdir(parents=True)
    booster.save_model(str(mdir / "model.json"))
    return tmp_path / "BDTFFModel"


def _ff_config(workspace, models_dir):
    cfg = load_config(workspace["yaml"])
    from wham.config import FakeFactorsCfg

    return cfg.model_copy(update={
        "fake_factors": FakeFactorsCfg(models=models_dir, era_label=0),
        "qcd": cfg.qcd.model_copy(update={
            "method": "ff", "ff_weight": "BDT_FF_score_QCD_sublead",
        }),
    })


def test_scores_in_skims_and_ff_qcd(workspace, models_dir) -> None:
    import pyarrow.parquet as pq

    cfg = _ff_config(workspace, models_dir)
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)

    # score column is baked into every skim and is finite/positive-capable
    for s in samples:
        table = pq.read_table(skims[s.name].path)
        assert "BDT_FF_score_QCD_sublead" in table.column_names
        assert np.isfinite(table.column("BDT_FF_score_QCD_sublead").to_numpy()).all()

    spec = build_fill_spec(cfg, families=["datamc"])
    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))
    h = hists[("datamc", "m_vis")]
    assert list(h.axes["region"]) == ["OS_iso", "OS_antiiso"]

    # Data uses the configured event weight as well as the FF score.
    data_sample = next(s for s in samples if s.kind == "data")
    df = pq.read_table(skims[data_sample.name].path).to_pandas()
    df = df[(df.pt_1 > 25) & (df.eta_1.abs() < 2.4) & (df.trg == 1) & (df.os == 1)]
    anti = df[(df.id_2 > 1) & (df.id_2 < 5) & (df.m_vis >= 0) & (df.m_vis < 250)]
    combined_weight = anti["weight"] * anti["BDT_FF_score_QCD_sublead"]
    expected = combined_weight.sum()
    got = h[{"process": "data", "region": "OS_antiiso", "variation": "nominal"}]
    assert float(got.view()["value"].sum()) == pytest.approx(float(expected), rel=1e-9)
    # and the variance is the sum of squared FF weights
    expected_w2 = (combined_weight ** 2).sum()
    assert float(got.view()["variance"].sum()) == pytest.approx(float(expected_w2), rel=1e-9)

    # QCD estimate lands in the signal region as clip(data - mc, 0)
    estimate_qcd(cfg, {("datamc", "m_vis"): h})
    qcd = h[{"process": "QCD", "region": "OS_iso", "variation": "nominal"}].view()["value"]
    data = h[{"process": "data", "region": "OS_antiiso", "variation": "nominal"}].view()["value"]
    mc = sum(
        h[{"process": p, "region": "OS_antiiso", "variation": "nominal"}].view()["value"]
        for p in ("TT", "DY")
    )
    assert np.allclose(qcd, np.maximum(data - mc, 0.0))


def test_skim_rebuilds_when_model_changes(workspace, models_dir) -> None:
    from wham.muffin import signature
    from wham.skim import find_skim

    cfg = _ff_config(workspace, models_dir)
    samples, _ = discover_samples(cfg)
    ensure_skims(cfg, samples, workers=1)

    sig1 = signature(cfg.fake_factors)
    assert find_skim(cfg.name, samples[0], cfg.required_columns(), sig1) is not None

    # touching the model file invalidates the skim match
    model_file = models_dir / "model_mt_QCD" / "model.json"
    model_file.write_bytes(model_file.read_bytes() + b" ")
    sig2 = signature(cfg.fake_factors)
    assert sig1 != sig2
    assert find_skim(cfg.name, samples[0], cfg.required_columns(), sig2) is None


def test_fake_factors_validation(tmp_path, models_dir) -> None:
    from wham.config import FakeFactorsCfg

    FakeFactorsCfg(models=models_dir, era="Run3_2022")
    FakeFactorsCfg(models=models_dir, era_label=3)
    with pytest.raises(ValueError, match="exactly one of"):
        FakeFactorsCfg(models=models_dir)
    with pytest.raises(ValueError, match="exactly one of"):
        FakeFactorsCfg(models=models_dir, era="Run3_2022", era_label=0)
    with pytest.raises(ValueError, match="not trained"):
        FakeFactorsCfg(models=models_dir, era="Run3_2024")
    with pytest.raises(ValueError, match="no model for"):
        FakeFactorsCfg(models=models_dir, era_label=0, processes=["Wjets"])
    with pytest.raises(ValueError, match="does not exist"):
        FakeFactorsCfg(models=tmp_path / "nowhere", era_label=0)


def test_trainings_layout(workspace, models_dir) -> None:
    """<models>/<channel>_<process>/best_model.json is found too."""
    import shutil

    from wham.config import FakeFactorsCfg
    from wham.muffin import FFModel, model_file

    alt = models_dir.parent / "muffin_trainings"
    (alt / "mt_QCD").mkdir(parents=True)
    shutil.copy(models_dir / "model_mt_QCD" / "model.json",
                alt / "mt_QCD" / "best_model.json")

    assert model_file(alt, "mt", "QCD").name == "best_model.json"
    FakeFactorsCfg(models=alt, era_label=0)  # validates
    model = FFModel(alt, "mt", "QCD", want_bootstrap=False)
    assert model.multiclass


def test_qcd_ff_validation() -> None:
    from wham.config import QCDCfg

    QCDCfg(method="ff", iso="id_2 >= 5", antiiso="id_2 < 5", ff_weight="bdt")
    with pytest.raises(ValueError, match="ff_weight"):
        QCDCfg(method="ff", iso="id_2 >= 5", antiiso="id_2 < 5")
