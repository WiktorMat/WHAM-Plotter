"""Single-pass histogram filling from skims.

One hist.Hist per (family, variable) with identical axes everywhere:
    StrCategory(process) x StrCategory(region) x StrCategory(variation) x Regular(var)
    storage = Weight()  (value + variance, i.e. sumw2)

so per-sample partial hists merge across workers with plain `+`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np

from wham.config import AnalysisConfig, Sample, VariableCfg, sample_scale
from wham.expr import parse, widened_arrow
from wham.skim import SkimInfo, read_skim

REGION_NOMINAL = "nominal"
REGIONS_ABCD = ("OS_iso", "SS_iso", "OS_antiiso", "SS_antiiso")
REGIONS_SS = ("OS", "SS")
REGIONS_FF = ("OS_iso", "OS_antiiso")
REGIONS_FFCHECK = ("OS_antiiso_raw", "OS_antiiso_ff")
REGIONS_FFCLOSURE = ("pass", "fail", "nan_weight")
REGIONS_FFESTIMATE = (
    "ff_pass",
    "ff_fail",
    "ff_nonfinite_component",
    "ff_nonfinite_combined",
)
REGIONS_CP = ("even", "odd")

HistKey = tuple[str, str]  # (family, variable-or-pair-name)


def resolution_name(reco: str, ref: str) -> str:
    return f"{reco}_from_{ref}"


def regions_for(family: str, qcd_method: str) -> tuple[str, ...]:
    # NB: "fitcp" regions are the configured component names; make_hist
    # derives them from the spec instead of calling this.
    if family == "datamc":
        if qcd_method == "abcd":
            return REGIONS_ABCD
        if qcd_method == "ff":
            return REGIONS_FF
        return REGIONS_SS
    if family == "ffcheck":
        return REGIONS_FFCHECK
    if family == "ffclosure":
        return REGIONS_FFCLOSURE
    if family == "cp":
        return REGIONS_CP
    return (REGION_NOMINAL,)


@dataclass(frozen=True)
class FillSpec:
    """Everything a worker needs, as plain picklable values."""

    selection: str
    trigger: str | None
    weight: str
    qcd_method: str
    qcd_os: str
    qcd_iso: str | None
    qcd_antiiso: str | None
    qcd_ff_weight: str | None
    lumi: float
    processes: tuple[str, ...]            # category axis content, fixed order
    # requested fills
    resolution: tuple[tuple[str, str, dict, dict], ...]  # (reco, ref, rescfg, refcfg)
    datamc: tuple[tuple[str, dict], ...]
    cp: tuple[tuple[str, str, str, dict], ...]       # (var, even_col, odd_col, varcfg)
    # weighted fit templates with signal-region cuts (trigger & os & iso):
    # (var, process, ((component, weight_expr), ...), varcfg)
    fitcp: tuple[tuple[str, str, tuple[tuple[str, str], ...], dict], ...] = ()
    # anti-iso fills with and without the FF weight (qcd.method=ff only)
    ffcheck: tuple[tuple[str, dict], ...] = ()
    # MUFFIN closure fills in a determination region:
    # (var, varcfg), process, dr_selection, pass, fail, muffin weight expression.
    ffclosure: tuple[tuple[str, dict], ...] = ()
    ffclosure_process: str | None = None
    ffclosure_selection: str | None = None
    ffclosure_pass: str | None = None
    ffclosure_fail: str | None = None
    ffclosure_weight: str | None = None
    ffestimate_enabled: bool = False
    ffestimate_selection: str | None = None
    ffestimate_pass: str | None = None
    ffestimate_fail: str | None = None
    ffestimate_fractions: tuple[tuple[str, float], ...] = ()
    ffestimate_scores: tuple[tuple[str, str], ...] = ()
    # shape variations filling the variation axis of the datamc/fitcp hists:
    # (name, target, processes, weight_up, weight_down, factors). weight/qcd_ff
    # targets use weight_up/down (factors empty); a columns target carries
    # ((column, factor), ...) — the listed processes are refilled with those
    # columns scaled before cuts and observables are evaluated (weights None).
    variations: tuple[
        tuple[str, str, tuple[str, ...], str | None, str | None,
              tuple[tuple[str, float], ...]], ...
    ] = ()
    # variables whose source column differs from their name (alias -> column)
    var_columns: tuple[tuple[str, str], ...] = ()
    # per-process extra selection (process -> cut expression), folded into the
    # read filter of that process's samples (skipped where columns are absent)
    process_cuts: tuple[tuple[str, str], ...] = ()


def _axis(var: str, vcfg: dict):
    import hist

    bins = vcfg["bins"]
    if isinstance(bins, int):
        return hist.axis.Regular(
            bins, float(vcfg["range"][0]), float(vcfg["range"][1]), name=var
        )
    return hist.axis.Variable([float(e) for e in bins], name=var)


def variation_labels(spec: FillSpec, family: str) -> list[str]:
    labels = ["nominal"]
    if family in ("datamc", "fitcp"):
        for name, target, *_ in spec.variations:
            if target == "columns":
                labels.append(name)  # one slice, named by the variation itself
            else:
                labels += [f"{name}_up", f"{name}_down"]
    return labels


def make_hist(spec: FillSpec, family: str, var: str, vcfg: dict):
    import hist

    if family == "fitcp":
        # region axis = the configured component names (sorted union so every
        # worker builds identical axes and partial hists merge with +)
        regions = sorted({c for v, _, comps, _ in spec.fitcp for c, _ in comps
                          if v == var})
    else:
        regions = list(regions_for(family, spec.qcd_method))
        if family == "datamc" and spec.ffestimate_enabled:
            regions = list(dict.fromkeys([*regions, *REGIONS_FFESTIMATE]))
    return hist.Hist(
        hist.axis.StrCategory(list(spec.processes), name="process"),
        hist.axis.StrCategory(regions, name="region"),
        hist.axis.StrCategory(variation_labels(spec, family), name="variation"),
        _axis(var, vcfg),
        storage=hist.storage.Weight(),
    )


def resolution_binning(cfg: AnalysisConfig, reco: str, ref: str) -> dict:
    """Binning for the derived resolution variable.

    Uses an explicit `variables` entry named '<reco>_from_<ref>' when present,
    otherwise a sensible default for the formula used.
    """
    name = resolution_name(reco, ref)
    if name in cfg.variables:
        return cfg.variables[name].model_dump()
    ref_cfg = cfg.variables[ref]
    if ref_cfg.kind == "angle":
        return {"bins": 40, "range": (-np.pi, np.pi), "kind": "angle", "label": None,
                "relative": False}
    if not ref_cfg.relative:
        lo, hi = ref_cfg.span()
        span = hi - lo
        return {"bins": 40, "range": (-span / 2, span / 2), "kind": "scalar",
                "label": None, "relative": False}
    return {"bins": 40, "range": (-2.0, 2.0), "kind": "scalar", "label": None,
            "relative": True}


def build_fill_spec(
    cfg: AnalysisConfig,
    *,
    families: list[str],
    only_vars: tuple[str, ...] | None = None,
) -> FillSpec:
    def want(var: str) -> bool:
        return only_vars is None or var in only_vars

    def vdump(var: str) -> dict:
        vcfg = cfg.variables[var]
        d = vcfg.model_dump()
        if vcfg.unroll is not None:
            # resolve the 2D unroll into a worker-ready payload: a unit-width
            # index axis of nx*ny bins plus the sub-variables' edges/columns
            x, y = vcfg.unroll
            xe = cfg.variables[x].edges()
            ye = cfg.variables[y].edges()
            d["bins"] = (len(xe) - 1) * (len(ye) - 1)
            d["range"] = (0.0, float(d["bins"]))
            d["unroll"] = {
                "x_column": cfg.column_of(x), "x_edges": xe,
                "y_column": cfg.column_of(y), "y_edges": ye,
            }
        return d

    resolution = tuple(
        (reco, ref, resolution_binning(cfg, reco, ref), vdump(ref))
        for reco, ref in cfg.plots.resolution
        if "resolution" in families and (only_vars is None or want(reco) or want(ref))
    )
    datamc = tuple(
        (v, vdump(v)) for v in cfg.plots.datamc if "datamc" in families and want(v)
    )
    cp = tuple(
        (c.var, c.even, c.odd, vdump(c.var))
        for c in cfg.plots.cp
        if "cp" in families and want(c.var)
    )
    fitcp = tuple(
        (c.var, c.process, tuple(sorted(c.components.items())), vdump(c.var))
        for c in cfg.plots.fitcp
        if "fitcp" in families and want(c.var)
    )
    ffcheck = tuple(
        (v, vdump(v)) for v in cfg.plots.ffcheck if "ffcheck" in families and want(v)
    )
    closure = cfg.fake_factors.closure if cfg.fake_factors is not None else None
    ffclosure = tuple(
        (v, vdump(v)) for v in (closure.variables if closure is not None and closure.enabled else [])
        if "ffclosure" in families and want(v)
    )
    estimate = cfg.fake_factors.estimate if cfg.fake_factors is not None else None
    estimate_active = estimate is not None and estimate.active()
    estimate_ar = estimate.application_region if estimate_active else None
    if estimate_active:
        from wham.jetfakes import component_score_columns

        estimate_scores = component_score_columns(estimate.components.enabled())
    else:
        estimate_scores = ()

    return FillSpec(
        selection=cfg.selection,
        trigger=cfg.trigger,
        weight=cfg.weight,
        qcd_method=cfg.qcd.method,
        qcd_os=cfg.qcd.os,
        qcd_iso=cfg.qcd.iso,
        qcd_antiiso=cfg.qcd.antiiso,
        qcd_ff_weight=cfg.qcd.ff_weight,
        lumi=cfg.lumi,
        processes=tuple(cfg.processes.keys()),
        resolution=resolution,
        datamc=datamc,
        cp=cp,
        fitcp=fitcp,
        ffcheck=ffcheck,
        ffclosure=ffclosure,
        ffclosure_process=closure.process if closure is not None and closure.enabled else None,
        ffclosure_selection=closure.selection if closure is not None and closure.enabled else None,
        ffclosure_pass=closure.pass_ if closure is not None and closure.enabled else None,
        ffclosure_fail=closure.fail if closure is not None and closure.enabled else None,
        ffclosure_weight=closure.weight_expr() if closure is not None and closure.enabled else None,
        ffestimate_enabled=estimate_active,
        ffestimate_selection=estimate_ar.selection if estimate_ar is not None else None,
        ffestimate_pass=estimate_ar.pass_ if estimate_ar is not None else None,
        ffestimate_fail=estimate_ar.fail if estimate_ar is not None else None,
        ffestimate_scores=estimate_scores,
        variations=tuple(
            (v.name, v.target, tuple(v.processes), v.weight_up, v.weight_down,
             tuple(sorted(v.factors.items())))
            for v in cfg.variations
        ),
        var_columns=tuple(
            (v, vcfg.column) for v, vcfg in cfg.variables.items() if vcfg.column
        ),
        process_cuts=tuple(
            (n, p.cut) for n, p in cfg.processes.items() if p.cut
        ),
    )


def hist_keys(spec: FillSpec) -> list[tuple[str, str, dict]]:
    """(family, name, binning) for every histogram the spec defines."""
    out: list[tuple[str, str, dict]] = []
    out += [
        ("resolution", resolution_name(reco, ref), rescfg)
        for reco, ref, rescfg, _ in spec.resolution
    ]
    out += [("datamc", v, vcfg) for v, vcfg in spec.datamc]
    out += [("cp", v, vcfg) for v, _, _, vcfg in spec.cp]
    # several processes may share a fitcp variable -> one hist
    seen: set[str] = set()
    for v, _, _, vcfg in spec.fitcp:
        if v not in seen:
            seen.add(v)
            out.append(("fitcp", v, vcfg))
    out += [("ffcheck", v, vcfg) for v, vcfg in spec.ffcheck]
    out += [("ffclosure", v, vcfg) for v, vcfg in spec.ffclosure]
    return out


def spec_extras(spec: FillSpec) -> dict[HistKey, dict]:
    """Per-hist extra cache-key payload (CP weight columns, column aliases)."""
    extras: dict[HistKey, dict] = {}
    for var, even_col, odd_col, _ in spec.cp:
        extras[("cp", var)] = {"even": even_col, "odd": odd_col}
    for var, process, comps, _ in spec.fitcp:
        key = ("fitcp", var)
        extras.setdefault(key, {"components": {}})["components"][process] = dict(comps)
    if spec.variations:
        # note: v[5] (the column factors) must stay in this cache-key payload —
        # dropping it would serve stale histograms across morph grid changes
        payload = [list(v[:2]) + [list(v[2]), v[3], v[4], [list(p) for p in v[5]]]
                   for v in spec.variations]
        for var, _ in spec.datamc:
            extras.setdefault(("datamc", var), {})["variations"] = payload
        for var, _, _, _ in spec.fitcp:
            extras.setdefault(("fitcp", var), {})["variations"] = payload
    colmap = dict(spec.var_columns)
    for reco, ref, _, _ in spec.resolution:
        rc, fc = colmap.get(reco, reco), colmap.get(ref, ref)
        if (rc, fc) != (reco, ref):
            key = ("resolution", resolution_name(reco, ref))
            extras[key] = {**extras.get(key, {}), "columns": [rc, fc]}
    return extras


# ---------------------------------------------------------------- worker


class _Columns:
    """Numpy column access with caching; columns converted at most once."""

    def __init__(self, table: Any):
        self._table = table
        self._cache: dict[str, np.ndarray] = {}
        self.names = set(table.column_names)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def get(self, name: str) -> np.ndarray:
        arr = self._cache.get(name)
        if arr is None:
            arr = self._table.column(name).to_numpy(zero_copy_only=False)
            self._cache[name] = arr
        return arr

    def eval(self, source: str) -> np.ndarray:
        parsed = parse(source)
        return parsed.evaluate({c: self.get(c) for c in parsed.columns})

    def finite(self, name: str) -> np.ndarray:
        key = f"__finite__{name}"
        arr = self._cache.get(key)
        if arr is None:
            arr = np.isfinite(self.get(name))
            self._cache[key] = arr
        return arr


class _ShiftedColumns(_Columns):
    """View of a _Columns with chosen columns scaled by constant factors:
    the same events with shifted kinematics (one morph grid point). Scaled
    arrays live in the view's own cache; other columns come from the base."""

    def __init__(self, base: _Columns, factors: dict[str, float]):
        self._base = base
        self._factors = factors
        self._cache: dict[str, np.ndarray] = {}
        self.names = base.names

    def get(self, name: str) -> np.ndarray:
        factor = self._factors.get(name)
        if factor is None:
            return self._base.get(name)
        arr = self._cache.get(name)
        if arr is None:
            arr = self._base.get(name) * factor
            self._cache[name] = arr
        return arr


def fill_sample(sample: Sample, skim: SkimInfo, spec: FillSpec) -> dict[HistKey, Any]:
    """Fill every requested histogram for one sample. Runs in a worker."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(skim.path)
    selection = parse(spec.selection)
    needed = set(skim.columns)  # read everything the skim has; it is already minimal

    # per-process cut, pushed into the read filter when the skim carries its
    # columns (so e.g. a genmatch cut is a no-op on data, which has no gen info)
    process_cut = dict(spec.process_cuts).get(sample.process)
    parsed_cut = None
    if process_cut is not None:
        candidate = parse(process_cut)
        if candidate.columns <= needed:
            parsed_cut = candidate

    # columns variations targeting this sample scale columns before cuts are
    # evaluated, so the read filter must be a superset over every grid point:
    # widen it and re-apply the exact selection in memory (per grid point, on
    # the scaled columns — events then migrate across selection edges)
    scale_bounds: dict[str, tuple[float, float]] = {}
    for _, target, procs, _, _, factors in spec.variations:
        if target != "columns" or sample.kind != "mc" or sample.process not in procs:
            continue
        for col, factor in factors:
            lo, hi = scale_bounds.get(col, (1.0, 1.0))
            scale_bounds[col] = (min(lo, factor), max(hi, factor))

    if scale_bounds:
        filter_expr = widened_arrow(selection, scale_bounds, schema)
        if parsed_cut is not None:
            cut_expr = widened_arrow(parsed_cut, scale_bounds, schema)
            if cut_expr is not None:
                filter_expr = cut_expr if filter_expr is None else filter_expr & cut_expr
    else:
        filter_expr = selection.arrow(schema)
        if parsed_cut is not None:
            filter_expr = filter_expr & parsed_cut.arrow(schema)

    table = read_skim(skim, columns=sorted(needed), filter_expr=filter_expr)
    cols = _Columns(table)
    n = table.num_rows

    scale = sample_scale(sample, spec.lumi)
    if sample.kind == "data":
        weights = np.ones(n)
    elif parse(spec.weight).columns <= cols.names:
        weights = cols.eval(spec.weight).astype(float) * scale
    else:
        weights = np.full(n, scale)

    # ---- evaluation contexts: nominal + one per columns variation. Each has
    # a (possibly scaled) column view and its own memoized region masks; under
    # a widened read filter the exact selection is re-applied in memory, on
    # the context's own columns (so scaled cuts migrate events)
    class _Ctx:
        def __init__(self, view: _Columns):
            self.cols = view
            self.parts: dict[str, np.ndarray | None] = {}
            self.residual: np.ndarray | None = None
            if scale_bounds:
                mask = view.eval(spec.selection).astype(bool)
                if parsed_cut is not None:
                    mask = mask & view.eval(process_cut).astype(bool)
                self.residual = mask

    nominal_ctx = _Ctx(cols)

    weight_cols = parse(spec.weight).columns if sample.kind != "data" else frozenset()

    def ctx_weights(ctx: "_Ctx", factors: dict[str, float]) -> np.ndarray:
        """Per-event weights in this context (re-derived only when the weight
        expression touches a scaled column)."""
        if weight_cols and weight_cols <= ctx.cols.names and weight_cols & set(factors):
            return ctx.cols.eval(spec.weight).astype(float) * scale
        return weights

    hists: dict[HistKey, Any] = {}
    dropped_nonfinite: dict[tuple[str, str, str, str], int] = {}
    colmap = dict(spec.var_columns)

    def column(var: str) -> str:
        return colmap.get(var, var)

    def fill(family: str, name: str, vcfg: dict, region: str,
             values: np.ndarray, mask: np.ndarray, w: np.ndarray,
             variation: str = "nominal") -> None:
        key = (family, name)
        h = hists.get(key)
        if h is None:
            h = hists[key] = make_hist(spec, family, name, vcfg)
        # boost-histogram propagates one NaN weight into the complete bin sum
        # and sumw2. Drop only the offending events while they are still
        # identifiable; replacing an already poisoned output bin would also
        # discard all valid events accumulated in that bin.
        finite = np.isfinite(values) & np.isfinite(w)
        bad = mask & ~finite
        if np.any(bad):
            dkey = (family, name, region, variation)
            dropped_nonfinite[dkey] = dropped_nonfinite.get(dkey, 0) + int(np.sum(bad))
        clean_mask = mask & finite
        h.fill(
            process=sample.process,
            region=region,
            variation=variation,
            **{name: values[clean_mask]},
            weight=w[clean_mask],
        )

    def var_data(ctx: "_Ctx", var: str, vcfg: dict) -> tuple[np.ndarray, np.ndarray] | None:
        """(values, valid_mask) for a plotted variable, or None if columns
        are missing. Unrolled variables map (x, y) onto a unit index axis.
        Read through the context, so scaled columns shift the values (and,
        for unrolled variables, migrate events between index bins)."""
        ccols = ctx.cols
        unroll = vcfg.get("unroll")
        if unroll is None:
            col = column(var)
            if col not in ccols:
                return None
            return ccols.get(col), ccols.finite(col)
        xcol, ycol = unroll["x_column"], unroll["y_column"]
        if xcol not in ccols or ycol not in ccols:
            return None
        xe = np.asarray(unroll["x_edges"])
        ye = np.asarray(unroll["y_edges"])
        ix = np.searchsorted(xe, ccols.get(xcol), side="right") - 1
        iy = np.searchsorted(ye, ccols.get(ycol), side="right") - 1
        nx, ny = len(xe) - 1, len(ye) - 1
        valid = (ccols.finite(xcol) & ccols.finite(ycol)
                 & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny))
        values = (ix + nx * iy).astype(float) + 0.5  # unit-axis bin centers
        return values, valid

    # ---- resolution: derived variable (nominal columns; the residual keeps
    # a widened read filter from leaking out-of-selection rows in)
    for reco, ref, rescfg, refcfg in spec.resolution:
        reco_col, ref_col = column(reco), column(ref)
        if reco_col not in cols or ref_col not in cols:
            continue
        rv, cv = cols.get(reco_col), cols.get(ref_col)
        is_angle = refcfg.get("kind") == "angle"
        relative = bool(rescfg.get("relative", True)) and not is_angle
        mask = cols.finite(reco_col) & cols.finite(ref_col)
        if nominal_ctx.residual is not None:
            mask = mask & nominal_ctx.residual
        if relative:
            mask = mask & (cv != 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                res = np.where(mask, (rv - cv) / np.where(cv == 0, 1.0, cv), 0.0)
        else:
            res = rv - cv
            if is_angle:
                res = (res + np.pi) % (2 * np.pi) - np.pi
        fill("resolution", resolution_name(reco, ref), rescfg,
             REGION_NOMINAL, res, mask, weights)

    # ---- region inputs shared by datamc and fitcp (memoized per context)
    def region_part(ctx: "_Ctx", name: str) -> np.ndarray | None:
        """Boolean mask for trigger/os/iso/anti, or None if undefined/missing cols."""
        if name not in ctx.parts:
            src = {
                "trigger": spec.trigger,
                "os": spec.qcd_os,
                "iso": spec.qcd_iso,
                "anti": spec.qcd_antiiso,
            }[name]
            if src is not None and parse(src).columns <= ctx.cols.names:
                ctx.parts[name] = ctx.cols.eval(src).astype(bool)
            else:
                ctx.parts[name] = None
        return ctx.parts[name]

    def base_mask(ctx: "_Ctx") -> np.ndarray:
        trig = region_part(ctx, "trigger")
        base = trig if trig is not None else np.ones(n, dtype=bool)
        if ctx.residual is not None:
            base = base & ctx.residual
        return base

    def sr_mask(ctx: "_Ctx") -> np.ndarray | None:
        """Signal region: trigger & OS (& iso for abcd/ff)."""
        os_mask = region_part(ctx, "os")
        if os_mask is None:
            return None
        sr = base_mask(ctx) & os_mask
        if spec.qcd_method in ("abcd", "ff"):
            iso = region_part(ctx, "iso")
            if iso is None:
                return None
            sr = sr & iso
        return sr

    # ---- datamc: trigger x charge x isolation regions (mask, weights) each
    def datamc_region_fills(ctx: "_Ctx", w: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        os_mask = region_part(ctx, "os")
        if os_mask is None:
            return {}
        base = base_mask(ctx)
        if spec.qcd_method == "abcd":
            iso = region_part(ctx, "iso")
            anti = region_part(ctx, "anti")
            if iso is None or anti is None:
                return {}
            return {
                "OS_iso": (base & os_mask & iso, w),
                "SS_iso": (base & ~os_mask & iso, w),
                "OS_antiiso": (base & os_mask & anti, w),
                "SS_antiiso": (base & ~os_mask & anti, w),
            }
        if spec.qcd_method == "ff":
            iso = region_part(ctx, "iso")
            anti = region_part(ctx, "anti")
            if iso is None or anti is None:
                return {}
            out = {"OS_iso": (base & os_mask & iso, w)}
            # anti-iso entries carry the per-event fake-factor weight
            if parse(spec.qcd_ff_weight).columns <= ctx.cols.names:
                out["OS_antiiso"] = (
                    base & os_mask & anti,
                    w * ctx.cols.eval(spec.qcd_ff_weight).astype(float),
                )
            return out
        return {"OS": (base & os_mask, w), "SS": (base & ~os_mask, w)}

    def fill_datamc(ctx: "_Ctx", region_fills: dict, variation: str = "nominal") -> None:
        for var, vcfg in spec.datamc:
            data = var_data(ctx, var, vcfg)
            if data is None:
                continue
            values, valid = data
            for region, (rmask, w) in region_fills.items():
                fill("datamc", var, vcfg, region, values, rmask & valid, w, variation)

    if spec.datamc:
        fill_datamc(nominal_ctx, datamc_region_fills(nominal_ctx, weights))

    # ---- mixed MUFFIN jet-fake estimate application region. The final
    # output process is materialized after all samples are merged.
    if (spec.datamc and spec.ffestimate_enabled
            and spec.ffestimate_selection is not None
            and spec.ffestimate_pass is not None
            and spec.ffestimate_fail is not None
            and spec.ffestimate_fractions):
        parsed_est_sel = parse(spec.ffestimate_selection)
        parsed_est_pass = parse(spec.ffestimate_pass)
        parsed_est_fail = parse(spec.ffestimate_fail)
        needed_est = parsed_est_sel.columns | parsed_est_pass.columns | parsed_est_fail.columns
        needed_est |= {col for _, col in spec.ffestimate_scores}
        if needed_est <= cols.names:
            est_base = base_mask(nominal_ctx) & cols.eval(spec.ffestimate_selection).astype(bool)
            est_pass = est_base & cols.eval(spec.ffestimate_pass).astype(bool)
            est_fail = est_base & cols.eval(spec.ffestimate_fail).astype(bool)
            overlap = est_pass & est_fail
            if np.any(overlap):
                raise RuntimeError(
                    "fake_factors.estimate application_region pass/fail selections "
                    f"overlap for {int(np.sum(overlap))} events in sample {sample.name}"
                )
            fractions = dict(spec.ffestimate_fractions)
            combined = np.zeros(n, dtype=float)
            bad_component = np.zeros(n, dtype=bool)
            for component, score_expr in spec.ffestimate_scores:
                values = cols.eval(score_expr).astype(float)
                bad_component |= ~np.isfinite(values)
                combined += fractions.get(component, 0.0) * values
            finite_combined = np.isfinite(combined)
            for var, vcfg in spec.datamc:
                data = var_data(nominal_ctx, var, vcfg)
                if data is None:
                    continue
                values, valid = data
                fill("datamc", var, vcfg, "ff_pass", values,
                     est_pass & valid, weights)
                fill("datamc", var, vcfg, "ff_fail", values,
                     est_fail & valid & ~bad_component & finite_combined,
                     weights * combined)
                fill("datamc", var, vcfg, "ff_nonfinite_component", values,
                     est_fail & valid & bad_component, np.ones(n))
                fill("datamc", var, vcfg, "ff_nonfinite_combined", values,
                     est_fail & valid & ~finite_combined, np.ones(n))

    # ---- ffcheck: anti-iso fills with and without the per-event FF weight
    if spec.ffcheck and spec.qcd_ff_weight is not None:
        os_mask = region_part(nominal_ctx, "os")
        anti = region_part(nominal_ctx, "anti")
        if (os_mask is not None and anti is not None
                and parse(spec.qcd_ff_weight).columns <= cols.names):
            amask = base_mask(nominal_ctx) & os_mask & anti
            ff_w = weights * cols.eval(spec.qcd_ff_weight).astype(float)
            for var, vcfg in spec.ffcheck:
                data = var_data(nominal_ctx, var, vcfg)
                if data is None:
                    continue
                values, valid = data
                mask = amask & valid
                fill("ffcheck", var, vcfg, "OS_antiiso_raw", values, mask, weights)
                fill("ffcheck", var, vcfg, "OS_antiiso_ff", values, mask, ff_w)

    # ---- MUFFIN closure in the configured determination region. This is
    # independent from estimate_qcd: it fills data and ordinary MC components,
    # then the renderer forms data - MC without clipping.
    if (spec.ffclosure and spec.ffclosure_selection is not None
            and spec.ffclosure_pass is not None and spec.ffclosure_fail is not None
            and spec.ffclosure_weight is not None):
        parsed_dr = parse(spec.ffclosure_selection)
        parsed_pass = parse(spec.ffclosure_pass)
        parsed_fail = parse(spec.ffclosure_fail)
        parsed_ff_weight = parse(spec.ffclosure_weight)
        needed_closure = (
            parsed_dr.columns
            | parsed_pass.columns
            | parsed_fail.columns
            | parsed_ff_weight.columns
        )
        if needed_closure <= cols.names:
            dr = base_mask(nominal_ctx) & cols.eval(spec.ffclosure_selection).astype(bool)
            pass_mask = dr & cols.eval(spec.ffclosure_pass).astype(bool)
            fail_mask = dr & cols.eval(spec.ffclosure_fail).astype(bool)
            overlap = pass_mask & fail_mask
            if np.any(overlap):
                raise RuntimeError(
                    "fake_factors.closure pass/fail selections overlap "
                    f"for {int(np.sum(overlap))} events in sample {sample.name}"
                )
            ff_values = cols.eval(spec.ffclosure_weight).astype(float)
            finite_ff = np.isfinite(ff_values)
            for var, vcfg in spec.ffclosure:
                data = var_data(nominal_ctx, var, vcfg)
                if data is None:
                    continue
                values, valid = data
                fill("ffclosure", var, vcfg, "pass", values,
                     pass_mask & valid, weights)
                fill("ffclosure", var, vcfg, "fail", values,
                     fail_mask & valid & finite_ff, weights * ff_values)
                fill("ffclosure", var, vcfg, "nan_weight", values,
                     fail_mask & valid & ~finite_ff, np.ones(n))

    # ---- cp: even/odd CP weights (MC only)
    if sample.kind != "data":
        for var, even_col, odd_col, vcfg in spec.cp:
            data = var_data(nominal_ctx, var, vcfg)
            if data is None or even_col not in cols or odd_col not in cols:
                continue
            values, valid = data
            if nominal_ctx.residual is not None:
                valid = valid & nominal_ctx.residual
            fill("cp", var, vcfg, "even", values, valid, weights * cols.get(even_col))
            fill("cp", var, vcfg, "odd", values, valid, weights * cols.get(odd_col))

    # ---- fitcp: weighted fit templates in the signal region, per process
    def fill_fitcp(ctx: "_Ctx", base_weights: np.ndarray,
                   variation: str = "nominal") -> None:
        sr = sr_mask(ctx)
        if sr is None:
            return
        for var, process, comps, vcfg in spec.fitcp:
            if process != sample.process:
                continue
            data = var_data(ctx, var, vcfg)
            if data is None:
                continue
            values, valid = data
            mask = sr & valid
            for comp, weight_expr in comps:
                if not parse(weight_expr).columns <= ctx.cols.names:
                    continue
                fill("fitcp", var, vcfg, comp, values, mask,
                     base_weights * ctx.cols.eval(weight_expr).astype(float), variation)

    if spec.fitcp and sample.kind != "data":
        fill_fitcp(nominal_ctx, weights)

    # ---- shape variations: refill matched processes with the replacement
    # weight (target=weight), with scaled columns (target=columns), or refill
    # every sample's anti-iso entries with the varied FF weight (target=qcd_ff)
    for name, target, procs, w_up, w_dn, factors in spec.variations:
        if target == "columns":
            # one slice: re-evaluate the matched MC processes' fills with the
            # listed columns scaled — the selection residual, region masks,
            # observables and (if touched) weights all shift together
            if sample.kind != "mc" or sample.process not in procs:
                continue
            fdict = dict(factors)
            ctx = _Ctx(_ShiftedColumns(cols, fdict))
            w = ctx_weights(ctx, fdict)
            if spec.datamc:
                fill_datamc(ctx, datamc_region_fills(ctx, w), variation=name)
            if spec.fitcp:
                fill_fitcp(ctx, w, variation=name)
            continue
        for direction, weight_expr in (("up", w_up), ("down", w_dn)):
            label = f"{name}_{direction}"
            if not parse(weight_expr).columns <= cols.names:
                continue
            if target == "weight":
                if sample.kind != "mc" or sample.process not in procs:
                    continue
                vweights = cols.eval(weight_expr).astype(float) * scale
                if spec.datamc:
                    fill_datamc(nominal_ctx, datamc_region_fills(nominal_ctx, vweights),
                                variation=label)
                if spec.fitcp:
                    fill_fitcp(nominal_ctx, vweights, variation=label)
            elif target == "qcd_ff" and spec.qcd_method == "ff" and spec.datamc:
                os_mask = region_part(nominal_ctx, "os")
                anti = region_part(nominal_ctx, "anti")
                if os_mask is None or anti is None:
                    continue
                amask = base_mask(nominal_ctx) & os_mask & anti
                vw = weights * cols.eval(weight_expr).astype(float)
                for var, vcfg in spec.datamc:
                    data = var_data(nominal_ctx, var, vcfg)
                    if data is None:
                        continue
                    values, valid = data
                    fill("datamc", var, vcfg, "OS_antiiso", values, amask & valid,
                         vw, label)

    for (family, name, region, variation), count in sorted(dropped_nonfinite.items()):
        warnings.warn(
            f"dropped {count} event(s) with non-finite value/weight while filling "
            f"sample={sample.name}, process={sample.process}, family={family}, "
            f"variable={name}, region={region}, variation={variation}",
            RuntimeWarning,
            stacklevel=2,
        )

    return hists


def complete_variation_slices(spec: FillSpec, hists: dict[HistKey, Any]) -> None:
    """Copy nominal content into every (process, region) a variation does not
    vary, making each variation slice a complete alternative universe (so the
    QCD estimate and the datacard export can treat slices uniformly)."""
    if not spec.variations:
        return
    for (family, _var), h in hists.items():
        if family not in ("datamc", "fitcp"):
            continue
        labels = list(h.axes["variation"])
        if len(labels) <= 1:
            continue
        regions = list(h.axes["region"])
        procs = list(h.axes["process"])
        view = h.view()
        nom = labels.index("nominal")
        for name, target, varied_procs, _, _, _ in spec.variations:
            slice_labels = ([name] if target == "columns"
                            else [f"{name}_up", f"{name}_down"])
            for label in slice_labels:
                if label not in labels:
                    continue
                vi = labels.index(label)
                for pi, proc in enumerate(procs):
                    for ri, region in enumerate(regions):
                        if target in ("weight", "columns"):
                            varied = proc in varied_procs
                        else:  # qcd_ff varies every sample's anti-iso entries
                            varied = family == "datamc" and region == "OS_antiiso"
                        if not varied:
                            view[pi, ri, vi, :] = view[pi, ri, nom, :]


def merge_hists(into: dict[HistKey, Any], part: dict[HistKey, Any]) -> None:
    for key, h in part.items():
        if key in into:
            into[key] = into[key] + h
        else:
            into[key] = h


# ---------------------------------------------------------------- driver


def fill_all(
    cfg: AnalysisConfig,
    samples: list[Sample],
    skims: dict[str, SkimInfo],
    *,
    families: list[str],
    only_vars: tuple[str, ...] | None = None,
    workers: int = 6,
    use_cache: bool = True,
    cache_only: bool = False,
    write_cache: bool = True,
    sidecars: bool = True,
    console=None,
) -> dict[HistKey, Any]:
    """Return all requested histograms, filling only what the cache lacks."""
    from wham.histcache import cache_key, load_hist, save_hist, write_sidecars
    from wham.parallel import run_parallel
    from wham.qcd import estimate_qcd

    spec = build_fill_spec(cfg, families=families, only_vars=only_vars)
    jetfake_fractions = None
    if "datamc" in families:
        from dataclasses import replace

        from wham.jetfakes import active_estimate, compute_ff_fractions

        if active_estimate(cfg):
            jetfake_fractions = compute_ff_fractions(
                cfg, samples, skims, console=console
            )
            spec = replace(
                spec,
                ffestimate_fractions=tuple(jetfake_fractions.fractions.items()),
            )
    wanted = hist_keys(spec)
    extras = spec_extras(spec)

    def _key(family: str, name: str, vcfg: dict) -> str:
        return cache_key(cfg, samples, skims, spec, family, name, vcfg,
                         extra=extras.get((family, name)))

    hists: dict[HistKey, Any] = {}
    missing: list[tuple[str, str, dict]] = []
    for family, name, vcfg in wanted:
        key = _key(family, name, vcfg)
        cached = load_hist(cfg.name, family, name, key) if use_cache else None
        if cached is not None:
            hists[(family, name)] = cached
        else:
            missing.append((family, name, vcfg))

    if missing and cache_only:
        names = ", ".join(f"{f}/{n}" for f, n, _ in missing)
        raise RuntimeError(
            f"histogram cache is stale or missing for: {names}. Run `wham plot`."
        )

    if missing:
        # One pass over the skims fills every missing histogram at once.
        missing_keys = {(f, n) for f, n, _ in missing}
        sub_spec = _restrict_spec(spec, missing_keys)
        jobs = [(s, skims[s.name], sub_spec) for s in samples if s.name in skims]
        fresh: dict[HistKey, Any] = {}
        done = 0
        for part in run_parallel(fill_sample, jobs, workers=workers,
                                 size_of=lambda j: j[1].path.stat().st_size):
            merge_hists(fresh, part)
            done += 1
            if console is not None:
                console.print(f"  filled {done}/{len(jobs)} samples", end="\r")

        # Samples can skip hists whose columns they lack; materialize empties.
        for family, name, vcfg in missing:
            if (family, name) not in fresh:
                fresh[(family, name)] = make_hist(sub_spec, family, name, vcfg)

        complete_variation_slices(sub_spec, fresh)
        if "datamc" in families:
            datamc = {k: h for k, h in fresh.items() if k[0] == "datamc"}
            if jetfake_fractions is not None:
                from wham.jetfakes import estimate_jet_fakes

                estimate_jet_fakes(cfg, datamc, jetfake_fractions, console=console)
            else:
                estimate_qcd(cfg, datamc)

        for family, name, vcfg in missing:
            h = fresh[(family, name)]
            if write_cache:
                save_hist(cfg.name, family, name, _key(family, name, vcfg), h)
            hists[(family, name)] = h

        if sidecars and write_cache:
            write_sidecars(cfg, hists)

    return hists


def _restrict_spec(spec: FillSpec, keys: set[HistKey]) -> FillSpec:
    from dataclasses import replace

    return replace(
        spec,
        resolution=tuple(
            x for x in spec.resolution
            if ("resolution", resolution_name(x[0], x[1])) in keys
        ),
        datamc=tuple(x for x in spec.datamc if ("datamc", x[0]) in keys),
        cp=tuple(x for x in spec.cp if ("cp", x[0]) in keys),
        fitcp=tuple(x for x in spec.fitcp if ("fitcp", x[0]) in keys),
        ffcheck=tuple(x for x in spec.ffcheck if ("ffcheck", x[0]) in keys),
        ffclosure=tuple(x for x in spec.ffclosure if ("ffclosure", x[0]) in keys),
    )
