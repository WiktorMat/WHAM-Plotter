"""Fit configuration: one YAML per fit, referencing an analysis config.

The fitter is generic — `wham fit <config>` and the YAML carries everything:
POIs, how each process's yield depends on them (optionally via weighted
template components), categories (datacard bins), systematics and scans.
The engine knows no physics modes; see Configurations/fits/TEMPLATE.yaml.
"""

from __future__ import annotations

import ast
import fnmatch
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from wham import util
from wham.config import (
    AnalysisConfig,
    ComponentFillCfg,
    PlotsCfg,
    VariationCfg,
    _Model,
    load_config,
)
from wham.expr import ExprError, parse

FITS_DIR_NAME = "fits"

DEFAULT_COMBINE_IMAGE = (
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/combine-standalone:latest"
)


def dc_name(name: str) -> str:
    """Datacard-safe name (combine chokes on '+' etc.)."""
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in name)


# ------------------------------------------------------------- formulas

FORMULA_FUNCS = ("cos", "sin", "tan", "atan", "sqrt", "exp", "log", "abs", "pow")


def translate_formula(expr: str, pois: list[str]) -> tuple[str, list[str]]:
    """Validate a POI expression and emit a TFormula with positional @i refs.

    Returns (tformula, deps); deps are the referenced POIs in @-index order.
    Powers may be written `^` or `**`; allowed functions: FORMULA_FUNCS.
    """
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid formula {expr!r}: {e.msg}") from None

    deps: list[str] = []
    binops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}

    def emit(node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return emit(node.body)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                return f"pow({emit(node.left)},{emit(node.right)})"
            op = binops.get(type(node.op))
            if op is None:
                raise ValueError(f"invalid formula {expr!r}: operator not allowed")
            return f"({emit(node.left)}{op}{emit(node.right)})"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            sign = "-" if isinstance(node.op, ast.USub) else "+"
            return f"({sign}{emit(node.operand)})"
        if isinstance(node, ast.Call):
            if (not isinstance(node.func, ast.Name)
                    or node.func.id not in FORMULA_FUNCS or node.keywords):
                raise ValueError(
                    f"invalid formula {expr!r}: only functions "
                    f"{', '.join(FORMULA_FUNCS)} are allowed"
                )
            return f"{node.func.id}({','.join(emit(a) for a in node.args)})"
        if isinstance(node, ast.Name):
            if node.id not in pois:
                raise ValueError(
                    f"invalid formula {expr!r}: unknown name '{node.id}' "
                    f"(POIs: {sorted(pois)})"
                )
            if node.id not in deps:
                deps.append(node.id)
            return f"@{deps.index(node.id)}"
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            return repr(node.value)
        raise ValueError(f"invalid formula {expr!r}: unsupported syntax")

    return emit(tree), deps


# ------------------------------------------------------------- schema


class PoiCfg(_Model):
    init: float = 1.0
    range: tuple[float, float]

    @model_validator(mode="after")
    def _ordered(self) -> "PoiCfg":
        lo, hi = self.range
        if not lo < hi:
            raise ValueError(f"POI range must be increasing, got [{lo}, {hi}]")
        if not lo <= self.init <= hi:
            raise ValueError(f"POI init {self.init} outside range [{lo}, {hi}]")
        return self


def _scale_exprs(scale: "str | dict[str, str]") -> list[str]:
    """The yield expressions of a scale, whether shared or per-category."""
    return [scale] if isinstance(scale, str) else list(scale.values())


class ComponentCfg(_Model):
    weight: str  # per-event weight column/expression for this template
    # yield expression in the POIs, or a {category: expression} map to scale
    # this template differently per datacard bin (e.g. a per-bin SF POI)
    scale: str | dict[str, str]


class ProcessModelCfg(_Model):
    # single template: a shared expression, or {category: expression} per bin
    scale: str | dict[str, str] | None = None
    components: dict[str, ComponentCfg] | None = None  # weighted templates

    @model_validator(mode="after")
    def _exactly_one(self) -> "ProcessModelCfg":
        if (self.scale is None) == (self.components is None):
            raise ValueError("a model process needs exactly one of 'scale' or 'components'")
        if self.components is not None and not self.components:
            raise ValueError("'components' must not be empty")
        return self


class GridCfg(_Model):
    from_: float = Field(alias="from")  # template grid start (TES factor f)
    to: float
    step: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "GridCfg":
        if not self.from_ < self.to:
            raise ValueError(f"grid 'from' must be < 'to', got [{self.from_}, {self.to}]")
        if self.from_ <= 0:
            raise ValueError(f"grid 'from' must be > 0 (a scale factor), got {self.from_}")
        return self


class MorphCfg(_Model):
    """A continuous shape-morphing POI (TES): at each template grid point f
    the listed columns are scaled by f (linear: momenta) or sqrt(f)
    (invariant masses, TauFW m_vis convention) BEFORE the selection and
    observables are evaluated. Cuts on scaled columns therefore migrate
    events across category edges (pt_2 bins) and scaled observables shift
    (m_vis), so both shape and yield vary along the grid. Realized as a
    combine CMSHistFunc, which interpolates template integrals in f."""

    process: str               # analysis (kind=mc) process whose shape morphs
    grid: GridCfg              # template points in f
    categories: list[str] = Field(min_length=1)  # bins this morph acts in
    # column -> scaling law at grid point f, e.g. {m_vis: sqrt, pt_2: linear}
    scales: dict[str, Literal["sqrt", "linear"]] = Field(min_length=1)
    init: float = 1.0
    range: tuple[float, float]

    @model_validator(mode="after")
    def _ranges(self) -> "MorphCfg":
        lo, hi = self.range
        if not lo < hi:
            raise ValueError(f"morph range must be increasing, got [{lo}, {hi}]")
        if not lo <= self.init <= hi:
            raise ValueError(f"morph init {self.init} outside range [{lo}, {hi}]")
        return self


class ModelCfg(_Model):
    pois: dict[str, PoiCfg]
    processes: dict[str, ProcessModelCfg]
    # continuous shape-morphing POIs (e.g. TES per decay mode)
    morphs: dict[str, MorphCfg] = {}

    @model_validator(mode="after")
    def _consistent(self) -> "ModelCfg":
        if not self.pois and not self.morphs:
            raise ValueError("model must declare at least one POI (pois or morphs)")
        if not self.processes:
            raise ValueError("model.processes must scale at least one process")
        clash = set(self.pois) & set(self.morphs)
        if clash:
            raise ValueError(f"names used as both POI and morph: {sorted(clash)}")
        pois = list(self.pois)
        used: set[str] = set()
        for proc, pm in self.processes.items():
            raw = ([pm.scale] if pm.scale is not None
                   else [c.scale for c in pm.components.values()])
            scales = [e for s in raw for e in _scale_exprs(s)]
            for scale in scales:
                try:
                    _, deps = translate_formula(scale, pois)
                except ValueError as e:
                    raise ValueError(f"model.processes.{proc}: {e}") from None
                used.update(deps)
        unused = set(pois) - used
        if unused:
            raise ValueError(f"POIs not used by any scale expression: {sorted(unused)}")
        return self


class CategoryCfg(_Model):
    name: str       # datacard bin name
    variable: str   # observable, a variable of the analysis config
    cut: str | None = None  # extra selection on top of the analysis selection
    # Control category: events/selection/processes come from ANOTHER analysis
    # config (path resolved like FitConfig.analysis, relative to the fit YAML).
    # None = the fit's main analysis. Control bins are plain backgrounds-only
    # channels (no model scales/morphs) — e.g. a Z->mumu bin anchoring the DY
    # normalization through a nuisance shared with the main bins.
    analysis: str | None = None

    @model_validator(mode="after")
    def _name_safe(self) -> "CategoryCfg":
        if not self.name or dc_name(self.name) != self.name:
            raise ValueError(
                f"category name '{self.name}' must be datacard-safe "
                "(letters, digits, underscores)"
            )
        return self


class ScanCfg(_Model):
    pois: list[str] = Field(min_length=1, max_length=2)  # 1 -> 1D, 2 -> 2D grid
    # per axis; a 2D scan may give per-POI counts [n_first, n_second] (in `pois`
    # order) for an asymmetric grid, e.g. a dense tes axis: [15, 51]
    points: int | tuple[int, int] = 50
    # explicit windows; without them the scan covers the full POI range
    range: tuple[float, float] | None = None  # 1D only
    ranges: list[tuple[float, float]] | None = None  # 2D only, one per POI

    @model_validator(mode="after")
    def _ranges_match_dim(self) -> "ScanCfg":
        if isinstance(self.points, int):
            if self.points <= 1:
                raise ValueError(f"scan needs at least 2 points, got {self.points}")
        else:
            if len(self.pois) != 2:
                raise ValueError("per-POI scan points [n1, n2] apply to 2D scans only")
            if any(n <= 1 for n in self.points):
                raise ValueError(
                    f"scan needs at least 2 points per axis, got {list(self.points)}")
        if self.range is not None and len(self.pois) != 1:
            raise ValueError("scan 'range' applies to 1D scans only (2D: 'ranges')")
        if self.ranges is not None:
            if len(self.pois) != 2:
                raise ValueError("scan 'ranges' applies to 2D scans only (1D: 'range')")
            if len(self.ranges) != 2:
                raise ValueError("scan 'ranges' needs exactly two [lo, hi] pairs")
        if len(set(self.pois)) != len(self.pois):
            raise ValueError(f"duplicate POI in scan {self.pois}")
        return self


class SystematicCfg(_Model):
    name: str
    effect: Literal["lnN", "rateParam", "shape"] = "lnN"
    processes: list[str]  # fnmatch patterns over process/template names
    categories: list[str] | None = None  # restrict to these bins; default all
    # symmetric factor, or [Down, Up] factors for an asymmetric lnN
    scaleFactor: float | tuple[float, float] | None = None  # lnN only
    init: float = 1.0  # rateParam starting value
    range: tuple[float, float] | None = None  # rateParam bounds, e.g. [0.1, 5]
    # shape only: replacement expressions for the per-event weight (or, when
    # the pattern matches the QCD process, for qcd.ff_weight) ...
    weight_up: str | None = None
    weight_down: str | None = None
    # ... OR a column shift: the matched (kind=mc) processes are refilled with
    # these columns scaled by (1 +- shift) (law per column, as in a morph's
    # `scales`) — an energy-scale nuisance for fake taus, cuts re-evaluated on
    # the shifted values (bin migration included)
    scales: dict[str, Literal["sqrt", "linear"]] | None = None
    shift: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def _fields_match_effect(self) -> "SystematicCfg":
        has_weights = self.weight_up is not None or self.weight_down is not None
        has_scales = self.scales is not None or self.shift is not None
        if self.effect == "lnN":
            if self.scaleFactor is None:
                raise ValueError(f"systematic '{self.name}': lnN needs a scaleFactor")
            factors = ((self.scaleFactor,) if isinstance(self.scaleFactor, float)
                       else self.scaleFactor)
            if any(factor <= 0 for factor in factors):
                raise ValueError(f"systematic '{self.name}': scaleFactor must be > 0")
            if self.range is not None or self.init != 1.0:
                raise ValueError(f"systematic '{self.name}': init/range are rateParam-only")
            if has_weights or has_scales:
                raise ValueError(
                    f"systematic '{self.name}': weight_up/down and scales/shift are shape-only"
                )
        elif self.effect == "rateParam":
            if self.scaleFactor is not None:
                raise ValueError(f"systematic '{self.name}': rateParam takes no scaleFactor")
            if has_weights or has_scales:
                raise ValueError(
                    f"systematic '{self.name}': weight_up/down and scales/shift are shape-only"
                )
        else:  # shape
            weights_ok = self.weight_up is not None and self.weight_down is not None
            scales_ok = self.scales and self.shift is not None
            if weights_ok == bool(scales_ok):
                raise ValueError(
                    f"systematic '{self.name}': shape needs either weight_up/weight_down "
                    "or scales+shift (a column-shift energy scale), not both"
                )
            if self.scaleFactor is not None or self.range is not None or self.init != 1.0:
                raise ValueError(
                    f"systematic '{self.name}': scaleFactor/init/range do not apply to shape"
                )
        return self


class AsimovCfg(_Model):
    enabled: bool = True  # combine -t -1
    parameters: dict[str, float] = {}  # injected truth; default: POIs at init


class ToyCfg(_Model):
    # Validation knob: modulates the two templates of every 2-component
    # process by (1 +- A*cos(x)) at datacard export, yield-preserving.
    # Every output is labeled TOY when nonzero.
    asymmetry: float = 0.0


class CombineCfg(_Model):
    image: str = DEFAULT_COMBINE_IMAGE
    # Path to a CMSSW release with combine + CombineHarvester built. REQUIRED
    # only for fits that declare a TES `morph`: those run under this cmsenv
    # instead of the standalone `image`, because the continuous morph must be a
    # combine-native CMSHistFunc (built by CombineHarvester's
    # BuildCMSHistFuncFactory) for autoMCStats to work — the RooMomentMorph used
    # by the container backend lacks getXVar() and crashes autoMCStats. Without
    # autoMCStats the postfit is not one-to-one. Plain (morph-free) fits ignore
    # this and use the container. See Configurations/tau_sf/README.md.
    cmssw: str | None = None


class ExportCfg(_Model):
    """Optional staging of WHAM shapes for an external datacard workflow."""

    target: str
    stage_dir: str | None = None
    output_file: str = "shapes.root"
    write_native_datacard: bool = True
    write_manifest: bool = True
    process_map: dict[str, str] = {}


class FitConfig(_Model):
    name: str
    analysis: str  # bare name or path of the analysis YAML
    categories: list[CategoryCfg] = Field(min_length=1)
    model: ModelCfg
    systematics: list[SystematicCfg] = []
    scans: list[ScanCfg] | None = None  # default: one 1D scan per POI
    asimov: AsimovCfg = AsimovCfg()
    toy: ToyCfg = ToyCfg()
    auto_mc_stats: int | None = 10
    combine: CombineCfg = CombineCfg()
    export: ExportCfg | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "FitConfig":
        names = [c.name for c in self.categories]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate category names: {names}")
        templates = [t for proc in self.model.processes for t in template_names(self, proc)]
        if len(set(templates)) != len(templates):
            raise ValueError(f"model template names collide after sanitization: {templates}")
        # the model lives in the main analysis; control bins are backgrounds-only
        control = {c.name for c in self.categories if c.analysis is not None}
        if len(control) == len(names):
            raise ValueError("every category is a control category — the model "
                             "needs at least one main-analysis category")
        # per-category scale maps may only reference declared main categories
        cat_set = set(names)
        for proc, pm in self.model.processes.items():
            raw = [pm.scale] if pm.components is None else [
                c.scale for c in pm.components.values()]
            for s in raw:
                if isinstance(s, dict):
                    unknown = set(s) - cat_set
                    if unknown:
                        raise ValueError(
                            f"model process '{proc}': scale maps unknown categories "
                            f"{sorted(unknown)} (categories: {sorted(cat_set)})"
                        )
                    bad = set(s) & control
                    if bad:
                        raise ValueError(
                            f"model process '{proc}': scale maps control categories "
                            f"{sorted(bad)} — model scales act in main-analysis bins only"
                        )
        for mname, morph in self.model.morphs.items():
            unknown = set(morph.categories) - cat_set
            if unknown:
                raise ValueError(
                    f"morph '{mname}' restricted to unknown categories "
                    f"{sorted(unknown)} (categories: {sorted(cat_set)})"
                )
            bad = set(morph.categories) & control
            if bad:
                raise ValueError(
                    f"morph '{mname}' acts in control categories {sorted(bad)} — "
                    "morphs live in the main analysis"
                )
        pois = set(self.model.pois) | set(self.model.morphs)
        for scan in self.scans or []:
            missing = set(scan.pois) - pois
            if missing:
                raise ValueError(f"scan over undeclared POIs: {sorted(missing)}")
        bad = set(self.asimov.parameters) - pois
        if bad:
            raise ValueError(f"asimov.parameters for undeclared POIs: {sorted(bad)}")
        for syst in self.systematics:
            if syst.categories is not None:
                unknown = set(syst.categories) - set(names)
                if unknown:
                    raise ValueError(
                        f"systematic '{syst.name}' restricted to unknown "
                        f"categories: {sorted(unknown)}"
                    )
        if self.toy.asymmetry != 0.0 and not any(
            pm.components is not None and len(pm.components) == 2
            for pm in self.model.processes.values()
        ):
            raise ValueError("toy.asymmetry needs a process with exactly two components")
        return self

    def resolved_scans(self) -> list[ScanCfg]:
        if self.scans is not None:
            return self.scans
        return [ScanCfg(pois=[p]) for p in all_pois(self)]


# ------------------------------------------------------------- derived


def template_names(fit: FitConfig, process: str) -> list[str]:
    """Datacard template names of one model process."""
    pm = fit.model.processes[process]
    if pm.components is not None:
        return [dc_name(f"{process}_{comp}") for comp in pm.components]
    return [dc_name(process)]


def scale_entries(fit: FitConfig) -> list[tuple[str | None, str, str]]:
    """(category | None, datacard template name, yield-scale expression).

    category=None: the expression applies in every datacard bin. A per-category
    `scale` map yields one entry per listed category (so the same template can be
    scaled by a different POI in each bin, e.g. a per-(DM,pT) ID scale factor)."""

    def expand(scale: str | dict[str, str], tmpl: str) -> list[tuple[str | None, str, str]]:
        if isinstance(scale, str):
            return [(None, tmpl, scale)]
        return [(cat, tmpl, expr) for cat, expr in scale.items()]

    out: list[tuple[str | None, str, str]] = []
    for proc, pm in fit.model.processes.items():
        if pm.components is not None:
            for comp, c in pm.components.items():
                out += expand(c.scale, dc_name(f"{proc}_{comp}"))
        else:
            out += expand(pm.scale, dc_name(proc))
    return out


def datacard_processes(fit: FitConfig, cfg: AnalysisConfig) -> tuple[list[str], list[str]]:
    """(model template names, background names) in stack order; model
    templates are the datacard 'signals' (ids <= 0)."""
    signals: list[str] = []
    backgrounds: list[str] = []
    for proc in cfg.stack_order():
        if proc in fit.model.processes:
            signals += template_names(fit, proc)
        else:
            backgrounds.append(dc_name(proc))
    return signals, backgrounds


def template_parents(fit: FitConfig, cfg: AnalysisConfig) -> dict[str, str]:
    """Datacard name -> config process name, for systematic pattern matching
    (covers both component templates and sanitized names like W+jets)."""
    parents = {dc_name(p): p for p in cfg.stack_order()}
    for proc in fit.model.processes:
        for name in template_names(fit, proc):
            parents[name] = proc
    return parents


def main_categories(fit: FitConfig) -> list[CategoryCfg]:
    """Categories of the fit's main analysis (the model acts here)."""
    return [c for c in fit.categories if c.analysis is None]


def control_categories(fit: FitConfig) -> list[CategoryCfg]:
    """Backgrounds-only categories drawing from another analysis config."""
    return [c for c in fit.categories if c.analysis is not None]


def union_datacard_processes(
    fit: FitConfig, cfg: AnalysisConfig, cat_cfgs: dict[str, AnalysisConfig],
) -> tuple[list[str], list[str]]:
    """datacard_processes with the control analyses' backgrounds appended
    (dc-name dedup: a name shared with a main process is one datacard column
    name — and thus one nuisance target — across bins)."""
    signals, backgrounds = datacard_processes(fit, cfg)
    seen = set(signals) | set(backgrounds)
    for cat in control_categories(fit):
        for proc in cat_cfgs[cat.name].stack_order():
            n = dc_name(proc)
            if n not in seen:
                seen.add(n)
                backgrounds.append(n)
    return signals, backgrounds


def union_template_parents(
    fit: FitConfig, cfg: AnalysisConfig, cat_cfgs: dict[str, AnalysisConfig],
) -> dict[str, str]:
    """template_parents across the main and every control analysis (main
    mapping wins on shared names)."""
    parents = template_parents(fit, cfg)
    for cat in control_categories(fit):
        for proc in cat_cfgs[cat.name].stack_order():
            parents.setdefault(dc_name(proc), proc)
    return parents


def syst_matches(patterns: list[str], dc: str, parent: str | None = None) -> bool:
    """A systematic pattern matches the datacard name or its parent process."""
    for pattern in patterns:
        if fnmatch.fnmatch(dc, pattern):
            return True
        if parent is not None and fnmatch.fnmatch(parent, pattern):
            return True
    return False


def fit_families(fit: FitConfig) -> list[str]:
    has_components = any(pm.components for pm in fit.model.processes.values())
    return ["datamc"] + (["fitcp"] if has_components else [])


def shape_affected(fit: FitConfig, cfg: AnalysisConfig, syst: SystematicCfg) -> set[str]:
    """Config processes whose templates move under a shape systematic.
    A varied MC weight also shifts the data-driven QCD estimate through the
    anti-iso subtraction, so the QCD process is included — EXCEPT for column
    shifts (scales+shift): a shifted MC subtraction can collapse the small
    QCD remainder to zero (or balloon it) in low-stat bins, and the violent
    vertical morph then drives channel pdfs negative. The QCD template stays
    nominal there; its rate lnNs cover the subtraction uncertainty."""
    qcd_proc = cfg.qcd_process()
    matched_mc = {p for p, pc in cfg.processes.items()
                  if pc.kind == "mc" and syst_matches(syst.processes, p)}
    if matched_mc:
        if syst.scales is not None:
            return matched_mc
        return matched_mc | ({qcd_proc} if qcd_proc is not None else set())
    return {qcd_proc} if qcd_proc is not None else set()


def shape_variations(fit: FitConfig, cfg: AnalysisConfig,
                     category: str | None = None) -> list[VariationCfg]:
    """Resolve effect=shape systematics into concrete fill variations:
    matched kind=mc processes get the replacement weight (weight_up/down) or a
    column-shift refill (scales+shift, e.g. a fake-τ energy scale); matching
    the QCD process instead varies qcd.ff_weight (the data-driven estimate).
    Restricted to `category` when given (a per-bin systematic, e.g. a fake ES
    decorrelated across pT bins, only fills its own bin)."""
    out: list[VariationCfg] = []
    qcd_proc = cfg.qcd_process()
    for syst in fit.systematics:
        if syst.effect != "shape":
            continue
        if (category is not None and syst.categories is not None
                and category not in syst.categories):
            continue
        matched_mc = [p for p, pc in cfg.processes.items()
                      if pc.kind == "mc" and syst_matches(syst.processes, p)]
        matches_qcd = qcd_proc is not None and syst_matches(syst.processes, qcd_proc)
        if matched_mc and matches_qcd:
            raise ValueError(
                f"shape systematic '{syst.name}' matches both MC processes "
                f"({matched_mc}) and the QCD process — split it in two"
            )
        if syst.scales is not None:
            if not matched_mc:
                raise ValueError(
                    f"shape systematic '{syst.name}': scales/shift needs kind=mc "
                    "processes (the data-driven QCD process has no columns to shift)"
                )
            for direction, f in (("up", 1 + syst.shift), ("down", 1 - syst.shift)):
                out.append(VariationCfg(
                    name=f"{syst.name}_{direction}", target="columns",
                    processes=matched_mc,
                    factors={col: (f ** 0.5 if law == "sqrt" else float(f))
                             for col, law in syst.scales.items()},
                ))
            continue
        if matches_qcd:
            if cfg.qcd.method != "ff":
                raise ValueError(
                    f"shape systematic '{syst.name}' varies the QCD estimate, "
                    "which requires qcd.method=ff"
                )
            out.append(VariationCfg(
                name=syst.name, target="qcd_ff",
                weight_up=syst.weight_up, weight_down=syst.weight_down,
            ))
        elif matched_mc:
            out.append(VariationCfg(
                name=syst.name, target="weight", processes=matched_mc,
                weight_up=syst.weight_up, weight_down=syst.weight_down,
            ))
        else:
            raise ValueError(
                f"shape systematic '{syst.name}' matches no kind=mc process "
                f"and not the QCD process (patterns {syst.processes})"
            )
    return out


def all_pois(fit: FitConfig) -> list[str]:
    """Every parameter of interest: rate-scale POIs then shape-morph POIs."""
    return list(fit.model.pois) + list(fit.model.morphs)


def grid_points(grid: "GridCfg") -> list[float]:
    """Template grid f-values, inclusive of both ends."""
    n = round((grid.to - grid.from_) / grid.step)
    return [round(grid.from_ + i * grid.step, 10) for i in range(n + 1)]


def fpoint_str(f: float) -> str:
    """Datacard-safe label for a grid point, e.g. 1.005 -> '1p005'."""
    return f"{f:.3f}".replace(".", "p").replace("-", "m")


def morph_point_label(morph_name: str, f: float) -> str:
    return f"{morph_name}__{fpoint_str(f)}"


def morph_factors(morph: "MorphCfg", f: float) -> dict[str, float]:
    """Per-column scale factors at grid point f (sqrt(f) or f per law)."""
    return {col: (f ** 0.5 if law == "sqrt" else float(f))
            for col, law in morph.scales.items()}


def morph_variations(fit: FitConfig, category: str | None = None) -> list[VariationCfg]:
    """Column-scale fill variations realizing the morph template grids.
    Restricted to `category` when given (each category only needs its own
    morphs' grid slices). The nominal point f=1.0 is the nominal template."""
    out: list[VariationCfg] = []
    for mname, morph in fit.model.morphs.items():
        if category is not None and category not in morph.categories:
            continue
        for f in grid_points(morph.grid):
            if abs(f - 1.0) < 1e-9:
                continue
            out.append(VariationCfg(
                name=morph_point_label(mname, f),
                target="columns",
                processes=[morph.process],
                factors=morph_factors(morph, f),
            ))
    return out


def _component_fills(fit: FitConfig, variable: str) -> list[ComponentFillCfg]:
    return [
        ComponentFillCfg(
            var=variable, process=proc,
            components={name: c.weight for name, c in pm.components.items()},
        )
        for proc, pm in fit.model.processes.items()
        if pm.components is not None
    ]


def category_analysis(fit: FitConfig, cfg: AnalysisConfig, cat: CategoryCfg) -> AnalysisConfig:
    """Analysis config for one category: cut folded into the selection,
    plots reduced to exactly the fills this category's templates need.
    `cfg` is the category's OWN analysis (a control category passes its
    control analysis; component fills only apply to processes it has)."""
    selection = cfg.selection if cat.cut is None else f"({cfg.selection}) & ({cat.cut})"
    fitcp = [f for f in _component_fills(fit, cat.variable) if f.process in cfg.processes]
    plots = PlotsCfg(datamc=[cat.variable], fitcp=fitcp)
    return cfg.model_copy(update={
        "selection": selection, "plots": plots,
        "variations": (shape_variations(fit, cfg, cat.name)
                       + morph_variations(fit, cat.name)),
    })


def union_analysis(fit: FitConfig, cfg: AnalysisConfig,
                   categories: list[CategoryCfg] | None = None) -> AnalysisConfig:
    """One config whose required_columns() covers every category fill —
    drives the skim build once. Its selection is never evaluated.
    `categories` restricts to the categories filled from `cfg` (the fill
    driver groups categories by analysis); default: all of them."""
    cats = fit.categories if categories is None else categories
    cuts = [c.cut for c in cats if c.cut]
    selection = " & ".join(f"({s})" for s in [cfg.selection, *cuts])
    variables = list(dict.fromkeys(c.variable for c in cats))
    fitcp = [f for v in variables for f in _component_fills(fit, v)
             if f.process in cfg.processes]
    plots = PlotsCfg(datamc=variables, fitcp=fitcp)
    # per-category union (deduped): an unscoped shape systematic resolves
    # against every analysis, but a categories:-scoped one must only resolve
    # against the analyses of the bins it acts in
    variations: dict[str, VariationCfg] = {}
    for c in cats:
        for v in shape_variations(fit, cfg, c.name):
            variations.setdefault(v.name, v)
    return cfg.model_copy(update={
        "selection": selection, "plots": plots,
        "variations": list(variations.values()),
    })


# ------------------------------------------------------------- loading


def resolve_fit_config(config: str | None) -> Path:
    """Accept a path, a bare name (Configurations/fits/<name>.yaml), or
    nothing when exactly one fit YAML exists."""
    fits_dir = util.repo_root() / "Configurations" / FITS_DIR_NAME

    if config is None:
        candidates = [p for p in sorted(fits_dir.glob("*.yaml")) if p.stem != "TEMPLATE"]
        if len(candidates) == 1:
            return candidates[0]
        names = ", ".join(p.stem for p in candidates) or "<none>"
        raise FileNotFoundError(
            f"choose a fit config: found {len(candidates)} in {fits_dir} ({names})"
        )

    path = Path(config)
    if path.is_file():
        return path
    candidate = fits_dir / f"{config}.yaml"
    if candidate.is_file():
        return candidate
    available = ", ".join(p.stem for p in sorted(fits_dir.glob("*.yaml"))) or "<none>"
    raise FileNotFoundError(
        f"no fit config '{config}' and no {candidate} (available: {available})"
    )


def _resolve_analysis(ref: str, fit_yaml_dir: Path) -> Path:
    path = Path(ref)
    for candidate in (
        path,
        fit_yaml_dir / path,
        util.repo_root() / "Configurations" / f"{ref}.yaml",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"analysis config '{ref}' not found")


def load_fit_config(
    path: str | Path,
) -> tuple[FitConfig, AnalysisConfig, dict[str, AnalysisConfig]]:
    """Load a fit YAML. Returns (fit, main analysis, category name -> its
    analysis) — main categories map to the shared main config object, control
    categories to their own analysis (each distinct path loaded once)."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    fit = FitConfig.model_validate(raw)

    loaded: dict[Path, AnalysisConfig] = {}

    def _load(ref: str) -> AnalysisConfig:
        p = _resolve_analysis(ref, path.parent).resolve()
        if p not in loaded:
            loaded[p] = load_config(p)
        return loaded[p]

    cfg = _load(fit.analysis)
    cat_cfgs = {c.name: (_load(c.analysis) if c.analysis is not None else cfg)
                for c in fit.categories}

    # Normalization-only nuisances need no alternative event inputs.  Attach
    # the shared defaults automatically, preserving an explicit same-name
    # definition from the fit YAML as an override.
    from uncertainties.registry import DEFAULT_NORMALIZATION_UNCERTAINTIES

    declared = {s.name.lower() for s in fit.systematics}
    aliases = {
        "lumi": {"lumi", "lumi_13p6tev", "lumi_13p6tev_2223"},
        "dy_xs": {"dy_xs", "xsec_dy", "cross_section_z"},
        "top_xs": {"top_xs", "xsec_tt", "xsec_top", "cross_section_ttbar"},
    }
    automatic = []
    for uncertainty in DEFAULT_NORMALIZATION_UNCERTAINTIES:
        processes = sorted({
            process
            for acfg in loaded.values()
            for process in uncertainty.processes(acfg)
        })
        already_declared = bool(declared & aliases.get(
            uncertainty.name, {uncertainty.name.lower()}))
        if processes and not already_declared:
            automatic.append(SystematicCfg(
                name=uncertainty.name,
                effect="lnN",
                processes=processes,
                scaleFactor=uncertainty.factors,
            ))
    if automatic:
        fit = fit.model_copy(update={"systematics": [*fit.systematics, *automatic]})

    # ---- cross-validation against the analysis configs
    for cat in fit.categories:
        ccfg = cat_cfgs[cat.name]
        if cat.variable not in ccfg.variables:
            raise ValueError(
                f"category '{cat.name}': variable '{cat.variable}' is not defined "
                f"in '{cat.analysis or fit.analysis}' ({sorted(ccfg.variables)})"
            )
        if cat.cut is not None:
            try:
                parse(cat.cut)
            except ExprError as e:
                raise ValueError(f"category '{cat.name}': invalid cut: {e}") from None

    for proc, pm in fit.model.processes.items():
        pcfg = cfg.processes.get(proc)
        if pcfg is None:
            raise ValueError(
                f"model process '{proc}' is not a process of '{fit.analysis}' "
                f"({sorted(cfg.processes)})"
            )
        if pcfg.kind == "data":
            raise ValueError(f"model process '{proc}' is the data process")
        if pm.components is not None:
            if pcfg.kind != "mc":
                raise ValueError(
                    f"model process '{proc}': components need per-event weights, "
                    f"so only kind=mc processes can be split (got kind={pcfg.kind})"
                )
            for comp, c in pm.components.items():
                try:
                    parse(c.weight)
                except ExprError as e:
                    raise ValueError(
                        f"model process '{proc}', component '{comp}': "
                        f"invalid weight: {e}"
                    ) from None

    for mname, morph in fit.model.morphs.items():
        pcfg = cfg.processes.get(morph.process)
        if pcfg is None:
            raise ValueError(
                f"morph '{mname}': process '{morph.process}' is not a process of "
                f"'{fit.analysis}' ({sorted(cfg.processes)})"
            )
        if pcfg.kind != "mc":
            raise ValueError(
                f"morph '{mname}': process '{morph.process}' must be kind=mc "
                f"(got kind={pcfg.kind}) to carry a TES grid"
            )

    if fit.model.morphs:
        # the morph pdf lives on combine's CMS_th1x, which is one observable
        # padded to the largest channel's bin count, so every MAIN category
        # must share the same observable binning for the morph to align.
        # Control categories are exempt: they are merged in as plain-TH1
        # channels at the datacard level (combineCards.py), outside the
        # CombineHarvester morph workspace.
        def _nbins(var: str) -> int:
            v = cfg.variables[var]
            if v.unroll is not None:
                x, y = v.unroll
                return (len(cfg.variables[x].edges()) - 1) * (len(cfg.variables[y].edges()) - 1)
            return len(v.edges()) - 1

        nbins = {c.name: _nbins(c.variable) for c in main_categories(fit)}
        if len(set(nbins.values())) != 1:
            raise ValueError(
                "TES morphing requires all main categories to share one observable "
                f"binning (combine uses a single CMS_th1x); got {nbins}"
            )

    for acfg in loaded.values():
        if acfg.data_process() is None:
            raise ValueError(f"analysis '{acfg.name}' has no kind=data process")

    # a control process sharing a datacard name with a model template would make
    # combineCards.py see one name as both signal and background — forbidden
    model_names = {t for proc in fit.model.processes for t in template_names(fit, proc)}
    model_names |= {dc_name(p) for p in fit.model.processes}
    for cat in control_categories(fit):
        clash = sorted({dc_name(p) for p in cat_cfgs[cat.name].stack_order()} & model_names)
        if clash:
            raise ValueError(
                f"control category '{cat.name}': processes {clash} of "
                f"'{cat.analysis}' collide with model process/template names — "
                "control bins are backgrounds-only; rename the process"
            )

    signals, backgrounds = union_datacard_processes(fit, cfg, cat_cfgs)
    parents = union_template_parents(fit, cfg, cat_cfgs)
    all_dc = signals + backgrounds
    for syst in fit.systematics:
        if not any(syst_matches(syst.processes, n, parents.get(n)) for n in all_dc):
            raise ValueError(
                f"systematic '{syst.name}' matches no datacard process "
                f"(patterns {syst.processes}, processes {all_dc})"
            )
        if syst.effect == "shape" and syst.weight_up is not None:
            for label, src in (("weight_up", syst.weight_up),
                               ("weight_down", syst.weight_down)):
                try:
                    parse(src)
                except ExprError as e:
                    raise ValueError(
                        f"systematic '{syst.name}': invalid {label}: {e}"
                    ) from None
    # raises on unresolvable shape systematics — per category against its own
    # analysis, so a scoped systematic only needs to resolve where it acts
    for cat in fit.categories:
        shape_variations(fit, cat_cfgs[cat.name], cat.name)

    return fit, cfg, cat_cfgs
