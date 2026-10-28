"""Nominal mixed MUFFIN jet->tau_h background estimate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wham.config import AnalysisConfig, Sample, sample_scale
from wham.expr import parse
from wham.muffin import score_column
from wham.skim import SkimInfo, read_skim

FF_PASS = "ff_pass"
FF_FAIL = "ff_fail"
FF_NONFINITE_COMPONENT = "ff_nonfinite_component"
FF_NONFINITE_COMBINED = "ff_nonfinite_combined"

COMPONENT_MODEL = {"QCD": "QCD", "Wjets": "Wjets", "ttbar": "ttbarMC"}


@dataclass(frozen=True)
class JetFakeFractions:
    fractions: dict[str, float]
    diagnostics: dict[str, float]


def active_estimate(cfg: AnalysisConfig) -> bool:
    est = cfg.fake_factors.estimate if cfg.fake_factors is not None else None
    return est is not None and est.active()


def enabled_components(cfg: AnalysisConfig) -> tuple[str, ...]:
    if not active_estimate(cfg):
        return ()
    return cfg.fake_factors.estimate.components.enabled()


def enabled_fake_processes(cfg: AnalysisConfig) -> set[str]:
    active = set(enabled_components(cfg))
    return {
        name for name, proc in cfg.processes.items()
        if proc.ff_component in active
    }


def component_score_columns(components: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((c, score_column(COMPONENT_MODEL[c])) for c in components)


class _Columns:
    def __init__(self, table: Any):
        self._table = table
        self._cache: dict[str, np.ndarray] = {}
        self.names = set(table.column_names)

    def get(self, name: str) -> np.ndarray:
        arr = self._cache.get(name)
        if arr is None:
            arr = self._table.column(name).to_numpy(zero_copy_only=False)
            self._cache[name] = arr
        return arr

    def eval(self, source: str) -> np.ndarray:
        parsed = parse(source)
        return parsed.evaluate({c: self.get(c) for c in parsed.columns})


def _sample_weights(cfg: AnalysisConfig, sample: Sample, cols: _Columns) -> np.ndarray:
    n = cols._table.num_rows
    scale = sample_scale(sample, cfg.lumi)
    parsed_weight = parse(cfg.weight)
    if parsed_weight.columns <= cols.names:
        return cols.eval(cfg.weight).astype(float) * scale
    return np.full(n, scale)


def _process_mask(cfg: AnalysisConfig, sample: Sample, cols: _Columns) -> np.ndarray:
    n = cols._table.num_rows
    mask = np.ones(n, dtype=bool)
    proc_cut = cfg.processes[sample.process].cut
    if proc_cut is not None:
        parsed = parse(proc_cut)
        if parsed.columns <= cols.names:
            mask &= cols.eval(proc_cut).astype(bool)
    return mask


def _fail_mask(cfg: AnalysisConfig, cols: _Columns) -> np.ndarray:
    est = cfg.fake_factors.estimate
    ar = est.application_region
    n = cols._table.num_rows
    mask = np.ones(n, dtype=bool)
    for expr in (cfg.selection, cfg.trigger, ar.selection, ar.fail):
        if expr is None:
            continue
        parsed = parse(expr)
        if not parsed.columns <= cols.names:
            return np.zeros(n, dtype=bool)
        mask &= cols.eval(expr).astype(bool)
    return mask


def compute_ff_fractions(
    cfg: AnalysisConfig,
    samples: list[Sample],
    skims: dict[str, SkimInfo],
    *,
    console=None,
) -> JetFakeFractions | None:
    if not active_estimate(cfg):
        return None

    components = enabled_components(cfg)
    enabled_fake = enabled_fake_processes(cfg)
    data_yield = 0.0
    mc_yields: dict[str, float] = {p: 0.0 for p in cfg.processes}
    score_cols = component_score_columns(components)
    nonfinite_component = 0.0

    est = cfg.fake_factors.estimate
    ar = est.application_region
    needed = set(parse(cfg.selection).columns)
    if cfg.trigger:
        needed |= parse(cfg.trigger).columns
    needed |= parse(ar.selection).columns | parse(ar.fail).columns
    needed |= parse(cfg.weight).columns
    for _, col in score_cols:
        needed.add(col)
    for proc in cfg.processes.values():
        if proc.cut:
            needed |= parse(proc.cut).columns

    for sample in samples:
        if sample.name not in skims:
            continue
        table = read_skim(skims[sample.name], columns=sorted(needed))
        cols = _Columns(table)
        missing_scores = [col for _, col in score_cols if col not in cols.names]
        if missing_scores:
            raise ValueError(
                "fake_factors.estimate missing MUFFIN score columns in skim "
                f"{sample.name}: {', '.join(missing_scores)}"
            )
        mask = _fail_mask(cfg, cols) & _process_mask(cfg, sample, cols)
        if not np.any(mask):
            continue
        weights = _sample_weights(cfg, sample, cols)
        if sample.kind == "data":
            data_yield += float(np.sum(weights[mask]))
        elif sample.kind == "mc":
            mc_yields[sample.process] = mc_yields.get(sample.process, 0.0) + float(
                np.sum(weights[mask])
            )
        component_bad = np.zeros(table.num_rows, dtype=bool)
        for _, col in score_cols:
            if col in cols.names:
                component_bad |= ~np.isfinite(cols.get(col).astype(float))
        nonfinite_component += float(np.sum(mask & component_bad))

    enabled_w = sum(
        y for p, y in mc_yields.items()
        if cfg.processes[p].ff_component == "Wjets" and p in enabled_fake
    )
    enabled_t = sum(
        y for p, y in mc_yields.items()
        if cfg.processes[p].ff_component == "ttbar" and p in enabled_fake
    )
    non_target = sum(
        y for p, y in mc_yields.items()
        if p not in enabled_fake and cfg.processes[p].kind == "mc"
    )
    disabled_fake = sum(
        y for p, y in mc_yields.items()
        if cfg.processes[p].ff_component is not None and p not in enabled_fake
    )
    genuine = non_target - disabled_fake
    jet_fail = data_yield - non_target
    qcd_residual = jet_fail - enabled_w - enabled_t

    fractions: dict[str, float]
    if components == ("QCD",):
        fractions = {"QCD": 1.0}
    else:
        if jet_fail <= 0.0:
            raise ValueError(
                "fake_factors.estimate cannot compute fractions: "
                f"target fail yield is {jet_fail:.6g}"
            )
        if qcd_residual < 0.0:
            raise ValueError(
                "fake_factors.estimate cannot compute fractions: "
                f"QCD residual is negative ({qcd_residual:.6g})"
            )
        fractions = {"QCD": qcd_residual / jet_fail}
        if "Wjets" in components:
            fractions["Wjets"] = enabled_w / jet_fail
        if "ttbar" in components:
            fractions["ttbar"] = enabled_t / jet_fail
        if not all(np.isfinite(v) for v in fractions.values()):
            raise ValueError(f"fake_factors.estimate produced non-finite fractions: {fractions}")
        total = sum(fractions.values())
        if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-9):
            raise ValueError(
                "fake_factors.estimate fractions do not sum to one: "
                f"{total:.12g} from {fractions}"
            )

    diagnostics = {
        "fail_data_yield": data_yield,
        "nonjet_genuine_mc_yield": genuine,
        "disabled_fake_mc_yield": disabled_fake,
        "enabled_w_fake_mc_yield": enabled_w,
        "enabled_tt_fake_mc_yield": enabled_t,
        "qcd_residual": qcd_residual,
        "target_fail_yield": jet_fail,
        "sum_fractions": sum(fractions.values()),
        "nonfinite_component_ffs": nonfinite_component,
    }
    if console is not None:
        console.print(
            "  jet_fakes fractions: "
            f"fail_data={data_yield:.6g}, genuine_mc={genuine:.6g}, "
            f"disabled_fake_mc={disabled_fake:.6g}, enabled_w_fake={enabled_w:.6g}, "
            f"enabled_tt_fake={enabled_t:.6g}, qcd_residual={qcd_residual:.6g}, "
            f"f_QCD={fractions.get('QCD', 0.0):.6g}, "
            f"f_Wjets={fractions.get('Wjets', 0.0):.6g}, "
            f"f_ttbar={fractions.get('ttbar', 0.0):.6g}, "
            f"sum={sum(fractions.values()):.6g}, "
            f"nonfinite_component_ffs={nonfinite_component:.0f}"
        )
        if components == ("QCD",) and jet_fail <= 0.0:
            console.print(
                f"  [yellow]jet_fakes: QCD-only target fail yield is {jet_fail:.6g}[/yellow]"
            )
    return JetFakeFractions(fractions=fractions, diagnostics=diagnostics)


def estimate_jet_fakes(
    cfg: AnalysisConfig,
    datamc_hists: dict[Any, Any],
    fractions: JetFakeFractions,
    *,
    console=None,
) -> None:
    """Fill the configured kind=ff process from MUFFIN-weighted fail AR.

    This nominal estimator does not clip negative bins. Enabled W/top fake MC
    processes are removed from the signal-region stack after they have served
    the fraction calculation and are excluded from the fail-region subtraction.
    """
    est = cfg.fake_factors.estimate
    output = est.output_process
    enabled_fake = enabled_fake_processes(cfg)
    data_proc = cfg.data_process()
    qcd_proc = cfg.qcd_process()
    if data_proc is None or output is None:
        return

    legacy_signal_region = "OS" if cfg.qcd.method == "ss" else "OS_iso"
    for (_family, var), h in datamc_hists.items():
        regions = list(h.axes["region"])
        procs = list(h.axes["process"])
        if FF_FAIL not in regions or FF_PASS not in regions or output not in procs:
            continue
        nbins = h.axes[-1].size
        data = np.zeros(nbins)
        data_w2 = np.zeros(nbins)
        mc = np.zeros(nbins)
        mc_w2 = np.zeros(nbins)
        nonfinite_combined = 0.0
        for proc in procs:
            fail = h[{"process": proc, "region": FF_FAIL, "variation": "nominal"}].view()
            if proc == data_proc:
                data += fail["value"]
                data_w2 += fail["variance"]
            elif proc not in enabled_fake and proc not in {qcd_proc, output}:
                mc += fail["value"]
                mc_w2 += fail["variance"]
            diag = h[{
                "process": proc,
                "region": FF_NONFINITE_COMBINED,
                "variation": "nominal",
            }].view()
            nonfinite_combined += float(np.sum(diag["value"]))

        counts = data - mc
        sumw2 = data_w2 + mc_w2
        output_regions = [FF_PASS]
        if legacy_signal_region in regions and legacy_signal_region != FF_PASS:
            output_regions.append(legacy_signal_region)
        for region in output_regions:
            _set_process_region(h, output, region, counts, sumw2)

        for proc in enabled_fake | ({qcd_proc} if qcd_proc is not None else set()):
            if proc in procs:
                for region in output_regions:
                    _set_process_region(h, proc, region, np.zeros(nbins), np.zeros(nbins))

        neg = counts < 0.0
        if console is not None:
            console.print(
                f"  jet_fakes/{var}: negative_bins={int(np.sum(neg))}, "
                f"negative_yield={float(np.sum(counts[neg])):.6g}, "
                f"nonfinite_combined_ffs={nonfinite_combined:.0f}"
            )


def _set_process_region(
    h: Any,
    process: str,
    region: str,
    counts: np.ndarray,
    sumw2: np.ndarray,
) -> None:
    idx_p = list(h.axes["process"]).index(process)
    idx_r = list(h.axes["region"]).index(region)
    idx_v = list(h.axes["variation"]).index("nominal")
    view = h.view()
    view[idx_p, idx_r, idx_v, :]["value"] = counts
    view[idx_p, idx_r, idx_v, :]["variance"] = sumw2
