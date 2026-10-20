"""Model-specific input preparation and mixed-model skim regression tests."""

import numpy as np
import pytest

BASE_FEATURES = ["decayModePNet", "jpt_pt", "pt", "eta", "phi", "n_jets", "n_bjets"]


def _minimal_table(with_ip=False):
    import pyarrow as pa

    columns = {
        "decayModePNet_2": [0, 10], "jpt_2": [60., -5.],
        "pt_2": [40., 50.], "eta_2": [0.5, -1.], "phi_2": [1., 2.],
        "n_jets": [2, 3], "n_bjets": [0, 1],
    }
    if with_ip:
        columns["ip_LengthSig_2"] = [1.2, 3.4]
    return pa.table(columns)


def test_only_model_features_are_required():
    from wham.muffin import build_features

    frame = build_features(_minimal_table(), 0, BASE_FEATURES)
    assert list(frame) == BASE_FEATURES
    np.testing.assert_allclose(frame.jpt_pt, [1.5, 0.])
    with pytest.raises(RuntimeError, match="ip_LengthSig_2"):
        build_features(_minimal_table(), 0, BASE_FEATURES + ["ip_LengthSig"])


def test_feature_aliases_and_direct_inputs():
    from wham.muffin import build_features

    table = _minimal_table().append_column("pt_1", [[100., 200.]])
    table = table.append_column("seeding_jpt_1", [[150., 100.]])
    table = table.append_column("seeding_jpt_2", [[80., 100.]])
    table = table.append_column("custom_global", [[7., 8.]])
    names = ["pt_other", "jpt_pt_other", "jpt_pt_1", "pt_2", "jpt_pt",
             "custom_global", "era_label", "is_lead_tau"]
    frame = build_features(table, 3, names)
    assert list(frame) == names
    np.testing.assert_allclose(frame.pt_other, [100., 200.])
    np.testing.assert_allclose(frame.jpt_pt_other, [1.5, 0.5])
    np.testing.assert_allclose(frame.jpt_pt_1, [1.5, 0.5])
    np.testing.assert_allclose(frame.jpt_pt, [2., 2.])
    np.testing.assert_allclose(frame.custom_global, [7., 8.])
    assert (frame.era_label == 3).all()
    assert (frame.is_lead_tau == 0).all()
    table = table.append_column("jpt_pt_2", [[9., 10.]])
    np.testing.assert_allclose(build_features(table, 0, ["jpt_pt"]).jpt_pt, [9., 10.])
    assert build_features(table.slice(0, 0), 0, names).shape == (0, len(names))


def test_lead_leg_remaps_unsuffixed_and_other_features():
    from wham.muffin import build_features

    table = _minimal_table().append_column("pt_1", [[100., 200.]])
    table = table.append_column("seeding_jpt_1", [[150., 100.]])
    frame = build_features(
        table,
        3,
        ["pt", "pt_other", "jpt_pt", "era_label", "is_lead_tau"],
        scored_leg=1,
    )

    np.testing.assert_allclose(frame.pt, [100., 200.])
    np.testing.assert_allclose(frame.pt_other, [40., 50.])
    np.testing.assert_allclose(frame.jpt_pt, [1.5, 0.5])
    np.testing.assert_array_equal(frame.era_label, [3, 3])
    np.testing.assert_array_equal(frame.is_lead_tau, [1, 1])


def test_legacy_met_feature():
    from wham.muffin import build_features

    table = _minimal_table().append_column("met_pt", [[80., 100.]])
    table = table.append_column("met_dphi_2", [[0., np.pi]])
    frame = build_features(table, 0, ["met_var_qcd"])
    np.testing.assert_allclose(frame.met_var_qcd, [2., -2.])


def test_model_metadata_controls_skim_columns(tmp_path):
    import json
    from wham.config import FakeFactorsCfg
    from wham.muffin import source_columns

    for process, names in (("QCD", BASE_FEATURES),
                           ("Wjets", BASE_FEATURES + ["ip_LengthSig", "new_global"])):
        directory = tmp_path / f"mt_{process}"
        directory.mkdir()
        (directory / "best_model.json").write_text(
            json.dumps({"learner": {"feature_names": names}}), encoding="utf-8")
    ff = FakeFactorsCfg(models=tmp_path, era_label=0, processes=["QCD"])
    required = source_columns(ff)
    assert "ip_LengthSig_2" not in required
    assert "decayMode_2" not in required
    assert "met_pt" not in required
    assert {"jpt_pt_2", "seeding_jpt_2", "jpt_2", "pt_2"} <= required
    both = source_columns(ff.model_copy(update={"score_legs": "both"}))
    assert {"jpt_pt_1", "seeding_jpt_1", "jpt_1", "pt_1"} <= both
    ff = ff.model_copy(update={"processes": ["QCD", "Wjets"]})
    assert {"ip_LengthSig_2", "new_global"} <= source_columns(ff)
    (tmp_path / "mt_QCD" / "best_model.json").write_text(
        '{"learner": {"feature_names": []}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="carries no feature names"):
        source_columns(ff)


@pytest.mark.parametrize("systematics", [False, True])
@pytest.mark.parametrize("explicit_suffix", [False, True])
def test_mixed_model_inputs_through_skim(tmp_path, monkeypatch, systematics, explicit_suffix):
    xgb = pytest.importorskip("xgboost")
    pytest.importorskip("hist")
    import pyarrow.parquet as pq
    from wham.config import FakeFactorsCfg, Sample
    from wham.muffin import FFModel, build_features, signature, source_columns
    from wham.skim import build_skim
    import wham.util

    monkeypatch.chdir(tmp_path)  # isolate diagnostics
    monkeypatch.setattr(wham.util, "cache_root", lambda: tmp_path / "cache")
    rng = np.random.default_rng(21)
    for process in ("QCD", "Wjets"):
        names = BASE_FEATURES + (["ip_LengthSig"] if process == "Wjets" else [])
        if explicit_suffix:
            names = [n if n.startswith("n_") else n + "_2" for n in names]
        names = names[::-1]  # model order must win over source order
        train = xgb.DMatrix(rng.random((40, len(names))),
                            label=np.arange(40) % 4, feature_names=names)
        booster = xgb.train({"objective": "multi:softprob", "num_class": 4,
                             "max_depth": 1, "nthread": 1}, train, num_boost_round=1)
        directory = tmp_path / "models" / f"mt_{process}"
        directory.mkdir(parents=True)
        booster.save_model(directory / "best_model.json")

    ff = FakeFactorsCfg(models=tmp_path / "models", era_label=0,
                        processes=["QCD", "Wjets"], systematics=systematics)
    required = source_columns(ff)
    assert "ip_LengthSig_2" in required
    assert "met_pt" not in required
    assert "decayMode_2" not in required
    qcd_only = ff.model_copy(update={"processes": ["QCD"]})
    assert "ip_LengthSig_2" not in source_columns(qcd_only)

    for config, with_ip in ((qcd_only, False), (ff, True)):
        table = _minimal_table(with_ip)
        path = tmp_path / f"input_{with_ip}.parquet"
        pq.write_table(table, path)
        sample = Sample(name=f"data_{with_ip}", process="data", kind="data",
                        path=path, variation="nominal", size=path.stat().st_size,
                        mtime_ns=path.stat().st_mtime_ns, params=None)
        skim = build_skim("mixed", sample, source_columns(config), config, signature(config))
        result = pq.read_table(skim.path)
        for process in config.processes:
            model = FFModel(config.models, "mt", process, want_bootstrap=False)
            frame = build_features(table, 0, model.features)
            pred = model.booster.predict(xgb.DMatrix(frame))
            expected = model._ff_from_pred(pred)
            tag = f"BDT_FF_score_{process}_sublead"
            np.testing.assert_allclose(result[tag].to_numpy(), expected, equal_nan=True)
            if systematics:
                assert f"{tag}_Extrapolation_up" in result.column_names


