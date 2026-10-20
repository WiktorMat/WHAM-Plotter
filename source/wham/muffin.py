"""muffin: BDT fake-factor scores baked into the skims at plotting level.

Adapted from higgs-dna-waw scripts/ditau/post_processing/add_bdtfakefactorscores.py.

When the analysis YAML carries a `fake_factors:` block, the skim build applies
the XGBoost BDT FF models to the configured hadronic tau leg(s) and appends
BDT_FF_score_<process>_sublead/lead columns (plus _<Unc>_up/down with
systematics: true). The original EOS inputs are never touched; the scores live
only in the local skim cache and behave like any other column afterwards:
plot them as variables or point qcd.ff_weight at one.

Skims are keyed on the model files, so swapping or retraining a model rebuilds
them automatically.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import tarfile
from pathlib import Path
from typing import Any

from wham.util import file_signature

# Era labels the BDT FF models were trained with. Run3_2024 is not in the
# training set yet; set era_label explicitly to borrow the closest trained
# era until the models are retrained.
ERA_LABELS = {
    "Run3_2022": 0,
    "Run3_2022EE": 1,
    "Run3_2023": 2,
    "Run3_2023BPix": 3,
}

# model feature -> candidate source columns for the scored (tau_h) leg,
# tried in order. Engineered features (jpt_pt, met_var_qcd, ...) build on these.
SUBLEAD_SOURCES = {
    "decayMode": ("decayMode_2",),
    "decayModePNet": ("decayModePNet_2",),
    "pt": ("pt_2",),
    "eta": ("eta_2",),
    "phi": ("phi_2",),
    "ip_LengthSig": ("ip_LengthSig_2",),
    "charge": ("charge_2",),
    "met_dphi": ("met_dphi_2",),
    "n_jets": ("n_jets",),
    "n_bjets": ("n_bjets",),
    "dR": ("dR",),
    "met_pt": ("met_pt",),
    "jpt": ("seeding_jpt_2", "jpt_2"),
}

# Uncertainty maps lifted verbatim from the higgs-dna script (mt entries).
NONCLOSURE = {"QCD": 0.05, "Wjets": 0.05, "WjetsMC": 0.05, "ttbarMC": 0.05}
MODELLING = {"QCD": 0.02, "Wjets": 0.01, "WjetsMC": 0.02, "ttbarMC": 0.005}
EXTRAPOLATION = {
    "QCD": [(0, 75, 0.10), (75, float("inf"), 0.20)],
    "Wjets": [(0, float("inf"), 0.05)],
    "WjetsMC": [(0, 75, 0.05), (75, float("inf"), 0.10)],
    "ttbarMC": [(0, float("inf"), 0.02)],
}
BKGSUB_VARIATION = {"QCD": 0.20, "Wjets": 0.10}


def _feature_recipe(feature: str, scored_leg: int = 2):
    """Source candidates and optional derivation for one scored tau leg.

    Unsuffixed tau features refer to ``scored_leg``, ``_other`` to the other leg;
    explicit ``_1``/``_2`` names retain their physical leg numbering.
    Other names are read directly (e.g. new global training variables).
    """
    primary = str(scored_leg)
    other = "1" if scored_leg == 2 else "2"
    base, suffix, leg = feature, "", primary
    for ending, number in (("_other", other), ("_1", "1"), ("_2", "2")):
        if feature.endswith(ending):
            base, suffix, leg = feature[:-len(ending)], ending, number
            break
    if feature in {"era_label", "is_lead_tau"}:
        return (), "constant", ()
    if base in {"jpt_pt", "met_var_qcd"}:
        candidates = tuple(dict.fromkeys((f"{base}_{leg}", feature)))
        dependencies = ((f"jpt{suffix}", f"pt{suffix}") if base == "jpt_pt"
                        else ("met_pt", f"pt{suffix}", f"met_dphi{suffix}"))
        return candidates, base, dependencies
    if base == "jpt":
        return (f"seeding_jpt_{leg}", f"jpt_{leg}"), None, ()
    if base in SUBLEAD_SOURCES and SUBLEAD_SOURCES[base][0].endswith("_2"):
        return (f"{base}_{leg}",), None, ()
    if suffix == "_other":
        return (f"{base}_{other}",), None, ()
    return (feature,), None, ()


def source_columns(ff: Any = None) -> frozenset[str]:
    """Candidate raw inputs, including dependencies, for configured models.

    Read only JSON metadata here: planning a skim need not load the trees.
    Candidates absent from the source are allowed; build_features validates
    that each requested feature can actually be provided.
    """
    features = set(SUBLEAD_SOURCES) | {"jpt_pt", "era_label", "is_lead_tau"}
    if ff is not None:
        features = set()
        for process in ff.processes:
            path = model_file(ff.models, ff.channel, process)
            with path.open(encoding="utf-8") as stream:
                names = json.load(stream).get("learner", {}).get("feature_names")
            if not names:
                raise RuntimeError(f"{path} carries no feature names")
            features.update(names)
        if ff.systematics:
            features.add("pt")  # extrapolation uncertainty

    columns = set()

    scored_legs = (1, 2) if ff is not None and ff.score_legs == "both" else (2,)

    def collect(feature, scored_leg):
        sources, _, dependencies = _feature_recipe(feature, scored_leg)
        columns.update(sources)
        for dependency in dependencies:
            collect(dependency, scored_leg)

    for feature in features:
        for scored_leg in scored_legs:
            collect(feature, scored_leg)
    return frozenset(columns)


def model_file(models_dir: Path, channel: str, process: str) -> Path:
    """Locate the model, supporting both layouts:
    higgs-dna   <models>/model_<channel>_<process>/model.json
    trainings   <models>/<channel>_<process>/best_model.json
    """
    candidates = [
        Path(models_dir) / d / f
        for d in (f"model_{channel}_{process}", f"{channel}_{process}")
        for f in ("model.json", "best_model.json")
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"no model for {channel}_{process}; looked for " +
        ", ".join(str(p) for p in candidates)
    )


def score_column(process: str, leg: str = "sublead") -> str:
    if leg not in {"lead", "sublead"}:
        raise ValueError(f"unknown tau leg {leg!r}")
    return f"BDT_FF_score_{process}_{leg}"


def signature(ff: Any) -> dict | None:
    """Cache payload identifying the models; changes when a model is retrained."""
    if ff is None:
        return None
    sig: dict[str, Any] = {
        "feature_builder_version": 3,
        "channel": ff.channel,
        "score_legs": ff.score_legs,
        "era_label": ff.resolved_era_label(),
        "processes": sorted(ff.processes),
        "systematics": ff.systematics,
        "models": {},
    }
    for process in ff.processes:
        mfile = model_file(ff.models, ff.channel, process)
        entry = {"model": file_signature(mfile)}
        tpath = mfile.parent / "temperature_scaling_results.json"
        if tpath.is_file():
            entry["temperature"] = file_signature(tpath)
        sig["models"][process] = entry
    return sig


# ------------------------------------------------------------- scoring math


def mask_denominator(probabilities,
                     epsilon=3e-4,
                     relative_epsilon=0.04):
    """Zero the subtraction classes where the FF denominator is ill-conditioned."""
    masked = probabilities.copy()

    denominator = masked[:, 1] - masked[:, 3]
    sr_scale = masked[:, 1] + masked[:, 3]

    relative_denominator = np.abs(denominator) / np.maximum(sr_scale, 1e-12)

    ill = (
        (denominator <= epsilon)
        | (relative_denominator <= relative_epsilon)
    )

    masked[ill, 2] = 0.0
    masked[ill, 3] = 0.0

    return masked


def renormalise_probabilities(probabilities):
    import numpy as np

    s = probabilities.sum(axis=1, keepdims=True)
    return probabilities / np.where(s == 0, 1, s)


def multiclass_ff(probabilities, denominator_epsilon=1e-3, fallback_epsilon=1e-6):
    import numpy as np

    probabilities = np.asarray(probabilities, dtype=float)
    probability_sum = probabilities.sum(axis=1, keepdims=True)
    p = probabilities / np.where(probability_sum == 0.0, 1.0, probability_sum)

    p_data_ar = p[:, 0]
    p_data_sr = p[:, 1]
    p_mc_ar = p[:, 2]
    p_mc_sr = p[:, 3]

    numerator = p_data_ar - p_mc_ar
    denominator = p_data_sr - p_mc_sr

    sr_scale = p_data_sr + p_mc_sr
    ar_scale = p_data_ar + p_mc_ar

    relative_denominator = np.abs(denominator) / np.maximum(sr_scale, 1e-12)
    relative_numerator = np.abs(numerator) / np.maximum(ar_scale, 1e-12)
    strong_sr_cancellation = relative_denominator < 0.05

    ff_subtracted = np.full(len(p), np.nan, dtype=float)
    np.divide(numerator, denominator, out=ff_subtracted, where=denominator != 0.0)

    ff_fallback = np.full(len(p), np.nan, dtype=float)
    np.divide(p_data_ar, p_data_sr, out=ff_fallback, where=p_data_sr > fallback_epsilon)

    invalid_probabilities = (
        ~np.all(np.isfinite(p), axis=1)
        | np.any(p < 0.0, axis=1)
        | np.any(p > 1.0, axis=1)
    )
    small_denominator = denominator <= denominator_epsilon
    negative_subtracted_ff = ff_subtracted < 0.0
    nonfinite_subtracted_ff = ~np.isfinite(ff_subtracted)

    use_fallback = small_denominator | negative_subtracted_ff | nonfinite_subtracted_ff
    score = np.where(use_fallback, ff_fallback, ff_subtracted)

    invalid_fallback = use_fallback & ~np.isfinite(ff_fallback)
    invalid_score = invalid_probabilities | invalid_fallback | ~np.isfinite(score) | (score < 0.0)
    score[invalid_score] = np.nan

    diagnostics = {
        "p_data_ar": p_data_ar,
        "p_data_sr": p_data_sr,
        "p_mc_ar": p_mc_ar,
        "p_mc_sr": p_mc_sr,
        "numerator": numerator,
        "denominator": denominator,
        "ff_subtracted": ff_subtracted,
        "ff_fallback": ff_fallback,
        "use_fallback": use_fallback,
        "small_denominator": small_denominator,
        "negative_subtracted_ff": negative_subtracted_ff,
        "invalid_probabilities": invalid_probabilities,
        "invalid_fallback": invalid_fallback,
        "invalid_score": invalid_score,
        "relative_denominator": relative_denominator,
        "relative_numerator": relative_numerator,
        "strong_sr_cancellation": strong_sr_cancellation,
    }
    return score, diagnostics


def apply_temperature(probabilities, temperature: float):
    import numpy as np

    if temperature == 1.0:
        return probabilities
    if probabilities.ndim == 2:  # multiclass softprob
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
        logits -= logits.mean(axis=1, keepdims=True)
        return np.exp(logits / temperature)
    logits = np.log(np.clip(probabilities, 1e-12, 1.0) / np.clip(1 - probabilities, 1e-12, 1.0))
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def bkgsub_scores(probabilities, variation: float):
    """FF recomputed with the MC-subtraction classes scaled by +-variation."""
    import numpy as np

    nominal_ff, nominal_diag = multiclass_ff(probabilities)
    nominal_fallback = nominal_diag["use_fallback"]

    out = {}
    for sign, key in ((1, "down"), (-1, "up")):
        scaled = probabilities.copy()
        scaled[:, 2] *= 1 + sign * variation
        scaled[:, 3] *= 1 + sign * variation

        varied_ff, _ = multiclass_ff(scaled)

        # BkgSub should not affect events which use the nominal no-subtraction fallback.
        out[key] = np.where(nominal_fallback, nominal_ff, varied_ff)

    return out["up"], out["down"]


def extrapolation_uncertainty(process: str, pt):
    import numpy as np

    unc = np.zeros(len(pt), dtype=float)
    for low, high, val in EXTRAPOLATION.get(process, []):
        unc[(pt >= low) & (pt < high)] = val
    return unc


# ------------------------------------------------------------- models


class FFModel:
    """One process's model + calibration, applied to feature frames."""

    def __init__(self, models_dir: Path, channel: str, process: str, want_bootstrap: bool):
        import xgboost as xgb

        self.process = process
        mfile = model_file(models_dir, channel, process)
        mdir = mfile.parent
        self.booster = xgb.Booster()
        self.booster.load_model(str(mfile))
        self.features = self.booster.feature_names
        if not self.features:
            raise RuntimeError(f"{mfile} carries no feature names")
        self.multiclass = "multi" in json.loads(
            self.booster.save_config())["learner"]["objective"]["name"]

        self.temperature = 1.0
        tpath = mdir / "temperature_scaling_results.json"
        if tpath.is_file():
            with open(tpath) as f:
                self.temperature = float(json.load(f).get("optimal_temperature", 1.0))

        self.bootstrap: list = []
        if want_bootstrap:
            self.bootstrap = self._load_bootstrap(
                mdir / "bootstrap_models" / "bootstrap_models.tar.gz")

    def _load_bootstrap(self, tar_path: Path) -> list:
        import xgboost as xgb

        # silently absent or a git-lfs pointer stub -> no Bootstrap columns
        if not tar_path.is_file() or not tarfile.is_tarfile(tar_path):
            return []
        out = []
        with tarfile.open(tar_path, "r:gz") as tar:
            for mem in sorted(tar.getmembers(), key=lambda m: m.name):
                if not (mem.isfile() and mem.name.endswith(".json")):
                    continue
                fobj = tar.extractfile(mem)
                if fobj is None:
                    continue
                model = xgb.Booster()
                model.load_model(bytearray(fobj.read()))
                out.append(model)
        return out

    def _ff_from_pred(self, pred):
        if self.multiclass:
            return multiclass_ff(pred)[0]
        return (1.0 - pred) / pred

    def score(self, frame, systematics: bool, diagnostics_dir: Path | None = None,
          outlier_threshold: float = 5.0, leg: str = "sublead") -> dict:
        """{column name: np.ndarray} for one engineered feature frame."""
        import numpy as np
        import xgboost as xgb

        dmat = xgb.DMatrix(frame[self.features])
        pred = apply_temperature(self.booster.predict(dmat), self.temperature)

        diagnostics = None
        if self.multiclass:
            score, diagnostics = multiclass_ff(pred)
        else:
            score = np.full(len(pred), np.nan, dtype=float)
            np.divide(1.0 - pred, pred, out=score, where=pred > 0.0)

        tag = score_column(self.process, leg)
        out = {tag: score}

        if diagnostics_dir is not None and diagnostics is not None:
            summary = summarise_ff_diagnostics(score, diagnostics)
            summary.update({
                "process": self.process,
                "temperature": self.temperature,
                "features": list(self.features),
            })
            write_diagnostic_summary(summary, diagnostics_dir, self.process)

            outliers = build_outlier_frame(
                frame, score, diagnostics, threshold=outlier_threshold
            )
            write_outlier_frame(outliers, diagnostics_dir, self.process)
        if not systematics:
            return out

        if self.multiclass:
            up, down = bkgsub_scores(pred, BKGSUB_VARIATION[self.process])
        else:
            up = down = np.zeros_like(score)
        out[f"{tag}_BkgSub_up"], out[f"{tag}_BkgSub_down"] = up, down

        mod = MODELLING.get(self.process, 0.0)
        out[f"{tag}_Modelling_up"] = score * (1 + mod)
        out[f"{tag}_Modelling_down"] = score * (1 - mod)

        ext = extrapolation_uncertainty(self.process, np.asarray(frame["pt"]))
        out[f"{tag}_Extrapolation_up"] = score * (1 + ext)
        out[f"{tag}_Extrapolation_down"] = score * (1 - ext)

        nc = NONCLOSURE.get(self.process, 0.0)
        out[f"{tag}_NonClosure_up"] = score * (1 + nc)
        out[f"{tag}_NonClosure_down"] = score * (1 - nc)

        if self.bootstrap:
            preds = np.stack([self._ff_from_pred(m.predict(dmat)) for m in self.bootstrap])
            sigma = preds.std(axis=0)
            out[f"{tag}_Bootstrap_up"] = score + sigma
            out[f"{tag}_Bootstrap_down"] = score - sigma
        return out


def build_features(table, era_label: int, features: Iterable[str] | None = None,
                   scored_leg: int = 2):
    """Build only requested features, in model order, from a pyarrow table."""
    import numpy as np
    import pandas as pd

    requested = list(dict.fromkeys(features if features is not None else
                                  [*SUBLEAD_SOURCES, "jpt_pt", "era_label", "is_lead_tau"]))
    cols = {}

    def resolve(feature):
        if feature in cols:
            return cols[feature]
        sources, operation, dependencies = _feature_recipe(feature, scored_leg)
        src = next((s for s in sources if s in table.column_names), None)
        if operation == "constant":
            values = np.full(
                table.num_rows,
                era_label if feature == "era_label" else int(scored_leg == 1),
            )
        elif src is not None:
            values = table.column(src).to_numpy(zero_copy_only=False)
        elif dependencies:
            args = [resolve(name) for name in dependencies]
            with np.errstate(divide="ignore", invalid="ignore"):
                if operation == "jpt_pt":
                    values = np.maximum(args[0] / args[1], 0)
                else:
                    values = (args[0] / args[1]) * np.cos(args[2])
        else:
            raise RuntimeError(f"{feature} (looked for {', '.join(sources)})")
        cols[feature] = values
        return values

    for feature in requested:
        try:
            resolve(feature)
        except RuntimeError as exc:
            raise RuntimeError(
                f"fake_factors: cannot build model feature {feature!r}: missing {exc}"
            ) from exc
    return pd.DataFrame({name: cols[name] for name in requested},
                        index=pd.RangeIndex(table.num_rows))


def augment_table(ff: Any, table):
    """Append the BDT FF score columns to a pyarrow table."""
    import pyarrow as pa

    diagnostics_dir = Path("ff_diagnostics")
    diagnostics_dir = Path(diagnostics_dir) if diagnostics_dir else None
    outlier_threshold = getattr(ff, "outlier_threshold", 5.0)

    for process in ff.processes:
        model = FFModel(ff.models, ff.channel, process, want_bootstrap=ff.systematics)
        features = list(model.features)
        if ff.systematics and "pt" not in features:
            features.append("pt")
        legs = ((1, "lead"), (2, "sublead")) if ff.score_legs == "both" else ((2, "sublead"),)
        for scored_leg, leg_name in legs:
            try:
                frame = build_features(
                    table, ff.resolved_era_label(), features, scored_leg=scored_leg
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"fake_factors: {ff.channel}_{process}_{leg_name}: {exc}"
                ) from exc
            scores = model.score(
                frame,
                ff.systematics,
                diagnostics_dir=diagnostics_dir,
                outlier_threshold=outlier_threshold,
                leg=leg_name,
            )
            for name, values in scores.items():
                column = pa.array(values, type=pa.float64())
                index = table.schema.get_field_index(name)
                if index >= 0:
                    table = table.set_column(index, name, column)
                else:
                    table = table.append_column(name, column)

    return table

def summarise_ff_diagnostics(score, diagnostics):
    import numpy as np

    finite = np.isfinite(score)
    valid_score = score[finite]

    large = finite & (score > 5.0)

    summary = {
        "n_events": int(len(score)),
        "n_valid": int(finite.sum()),
        "n_invalid": int((~finite).sum()),
        "n_fallback": int(diagnostics["use_fallback"].sum()),
        "n_small_denominator": int(diagnostics["small_denominator"].sum()),
        "n_negative_subtracted_ff": int(diagnostics["negative_subtracted_ff"].sum()),
        "n_invalid_probabilities": int(diagnostics["invalid_probabilities"].sum()),
        "n_invalid_fallback": int(diagnostics["invalid_fallback"].sum()),
        "n_ff_gt_2": int(np.sum(valid_score > 2.0)),
        "n_ff_gt_5": int(np.sum(valid_score > 5.0)),
        "n_ff_gt_10": int(np.sum(valid_score > 10.0)),
        "n_ff_gt_20": int(np.sum(valid_score > 20.0)),
        "sum_ff": float(np.sum(valid_score)),
        "sum_ff2": float(np.sum(valid_score**2)),
        "n_large_from_subtraction": int(np.sum(large & ~diagnostics["use_fallback"])),
        "n_large_from_fallback": int(np.sum(large & diagnostics["use_fallback"])),
        "n_strong_sr_cancellation": int(diagnostics["strong_sr_cancellation"].sum()),
        "n_large_with_strong_sr_cancellation": int(
            np.sum(large & diagnostics["strong_sr_cancellation"])
    ),
    }

    if len(valid_score):
        summary.update({
            "min_ff": float(np.min(valid_score)),
            "max_ff": float(np.max(valid_score)),
            "mean_ff": float(np.mean(valid_score)),
            "median_ff": float(np.median(valid_score)),
            "p90_ff": float(np.quantile(valid_score, 0.90)),
            "p99_ff": float(np.quantile(valid_score, 0.99)),
            "p999_ff": float(np.quantile(valid_score, 0.999)),
        })
    
    FF_BINS = np.array([
        0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0,
        10.0, 20.0, 50.0, 100.0, np.inf,
    ])

    counts, _ = np.histogram(valid_score, bins=FF_BINS)
    summary["ff_histogram"] = counts.tolist()
    summary["ff_bins"] = FF_BINS.tolist()

    return summary

def write_diagnostic_summary(summary, output_dir: Path, process: str):
    import json
    import os
    import uuid

    directory = Path(output_dir) / process
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{os.getpid()}_{uuid.uuid4().hex}.json"
    temporary_path = path.with_suffix(".tmp")

    with open(temporary_path, "w") as f:
        json.dump(summary, f, indent=2)

    temporary_path.replace(path)

def merge_diagnostic_summaries(input_dir: Path, output_path: Path):
    import json

    additive_fields = {
        "n_events",
        "n_valid",
        "n_invalid",
        "n_fallback",
        "n_small_denominator",
        "n_negative_subtracted_ff",
        "n_invalid_probabilities",
        "n_invalid_fallback",
        "n_ff_gt_2",
        "n_ff_gt_5",
        "n_ff_gt_10",
        "n_ff_gt_20",
        "n_large_from_subtraction",
        "n_large_from_fallback",
        "n_strong_sr_cancellation",
        "n_large_with_strong_sr_cancellation",
        "sum_ff",
        "sum_ff2",
    }

    merged = {}
    maximum_ff = float("-inf")

    merged_histogram = None
    ff_bins = None

    for path in Path(input_dir).rglob("*.json"):
        with open(path) as f:
            summary = json.load(f)

        histogram = summary.get("ff_histogram")
        if histogram is not None:
            if merged_histogram is None:
                merged_histogram = histogram.copy()
                ff_bins = summary["ff_bins"]
            else:
                if summary["ff_bins"] != ff_bins:
                    raise ValueError(f"Inconsistent FF bins in {path}")
                merged_histogram = [
                    current + added
                    for current, added in zip(merged_histogram, histogram)
                ]

        for field in additive_fields:
            merged[field] = merged.get(field, 0) + summary.get(field, 0)

        maximum_ff = max(maximum_ff, summary.get("max_ff", float("-inf")))

    merged["max_ff"] = maximum_ff
    merged["max_ff"] = None if maximum_ff == float("-inf") else maximum_ff
    merged["mean_ff"] = merged.get("sum_ff", 0.0) / max(merged.get("n_valid", 0), 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    if merged_histogram is not None:
        merged["ff_histogram"] = merged_histogram
        merged["ff_bins"] = ff_bins

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

def build_outlier_frame(frame, score, diagnostics, threshold=5.0):
    import pandas as pd
    import numpy as np

    selected = (
        (np.isfinite(score) & (score > threshold))
        | diagnostics["invalid_score"]
    )

    outliers = frame.loc[selected].copy()
    outliers["ff"] = score[selected]

    for name, values in diagnostics.items():
        outliers[name] = values[selected]

    return outliers

def write_outlier_frame(outliers, output_dir: Path, process: str):
    import os
    import uuid

    if outliers.empty:
        return

    directory = Path(output_dir) / process / "outliers"
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{os.getpid()}_{uuid.uuid4().hex}.parquet"
    temporary_path = path.with_suffix(".tmp.parquet")

    outliers.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
