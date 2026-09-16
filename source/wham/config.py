"""Analysis configuration: one validated YAML per analysis + sample discovery."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from wham.expr import ExprError, parse

FAMILIES = ("resolution", "datamc", "cp", "fitcp", "ffcheck", "ffclosure", "display3d")

# Columns the 3D display needs if that family is configured.
DISPLAY3D_COLUMNS = {"pt_1", "eta_1", "phi_1", "pt_2", "eta_2", "phi_2", "met_pt", "met_phi"}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class VariableCfg(_Model):
    bins: int | tuple[float, ...] | None = None  # count (needs range) or ascending edges
    range: tuple[float, float] | None = None
    column: str | None = None  # source column; defaults to the variable name
    label: str | None = None
    logy: bool = False  # render the datamc y-axis on a log scale
    kind: Literal["scalar", "angle"] = "scalar"
    relative: bool = True  # resolution: (reco-ref)/ref vs reco-ref
    # 2D discriminant unrolled to 1D: [x, y] variable names; the derived
    # variable has nx*ny unit-width bins, index = ix + nx*iy (y-major)
    unroll: tuple[str, str] | None = None

    @model_validator(mode="after")
    def _binning_valid(self) -> "VariableCfg":
        if self.unroll is not None:
            if self.bins is not None or self.range is not None or self.column is not None:
                raise ValueError(
                    "an 'unroll' variable derives its binning from the two "
                    "referenced variables; omit bins/range/column"
                )
            return self
        if self.bins is None:
            raise ValueError("a variable needs 'bins' (or 'unroll')")
        if isinstance(self.bins, int):
            if self.bins <= 0:
                raise ValueError(f"bins must be positive, got {self.bins}")
            if self.range is None:
                raise ValueError("an integer 'bins' needs an explicit 'range'")
            if not self.range[0] < self.range[1]:
                raise ValueError(f"range must be increasing, got {self.range}")
        else:
            if len(self.bins) < 2:
                raise ValueError("explicit bin edges need at least 2 values")
            if not all(a < b for a, b in zip(self.bins, self.bins[1:])):
                raise ValueError(f"bin edges must be strictly increasing, got {self.bins}")
            if self.range is not None:
                raise ValueError("'range' must be omitted when 'bins' lists explicit edges")
        return self

    def edges(self) -> list[float]:
        """Explicit bin edges (plain-binned variables only)."""
        if isinstance(self.bins, int):
            lo, hi = self.range
            return [lo + (hi - lo) * i / self.bins for i in range(self.bins + 1)]
        return [float(e) for e in self.bins]

    @field_validator("column")
    @classmethod
    def _column_is_identifier(cls, v: str | None) -> str | None:
        if v is not None and not v.isidentifier():
            raise ValueError(f"column must be a plain column name, got {v!r}")
        return v

    def span(self) -> tuple[float, float]:
        """(low, high) of the binned region, whichever way bins were given."""
        return self.range if self.range is not None else (self.bins[0], self.bins[-1])


class ProcessCfg(_Model):
    samples: list[str] = []
    color: str = "tab:gray"
    kind: Literal["mc", "data", "qcd", "ff"] = "mc"
    label: str | None = None  # legend text; falls back to the process name
    # extra per-event mask folded onto the selection for this process only,
    # e.g. a genmatch requirement isolating genuine tau_h (genPartFlav_2 == 5)
    # or excluding jet->tau_h fakes that the data-driven QCD estimate covers.
    # Skipped for any sample whose skim lacks the referenced columns (so a
    # genmatch cut on a process whose samples carry no gen info is a no-op).
    cut: str | None = None
    # For the mixed MUFFIN jet-fake estimate: identifies the MC fake component
    # used for the corresponding process fraction. Genuine pieces stay as
    # ordinary MC processes without this field.
    ff_component: Literal["Wjets", "ttbar"] | None = None

    @model_validator(mode="after")
    def _samples_match_kind(self) -> "ProcessCfg":
        if self.kind in ("qcd", "ff") and self.samples:
            raise ValueError(f"a kind={self.kind} process is derived and must not list samples")
        if self.kind not in ("qcd", "ff") and not self.samples:
            raise ValueError("process must list at least one sample pattern")
        if self.kind in ("qcd", "ff") and self.cut is not None:
            raise ValueError(f"a kind={self.kind} process is data-derived and takes no 'cut'")
        if self.kind != "mc" and self.ff_component is not None:
            raise ValueError("ff_component is only valid for kind=mc processes")
        return self


class QCDCfg(_Model):
    method: Literal["ss", "abcd", "ff"] = "ss"
    os: str = "os == 1"
    iso: str | None = None
    antiiso: str | None = None
    ff: float = 1.0  # ss method: flat SS->OS extrapolation factor
    # ff method: per-event fake-factor weight expression for the anti-iso
    # region (e.g. a BDT_FF_score_* column added by scripts/tools/muffin.py)
    ff_weight: str | None = None

    @model_validator(mode="after")
    def _method_needs_fields(self) -> "QCDCfg":
        if self.method == "abcd" and not (self.iso and self.antiiso):
            raise ValueError("qcd.method=abcd requires both 'iso' and 'antiiso' expressions")
        if self.method == "ff" and not (self.iso and self.antiiso and self.ff_weight):
            raise ValueError(
                "qcd.method=ff requires 'iso', 'antiiso' and 'ff_weight' expressions"
            )
        return self


class SampleParams(_Model):
    xs: float = Field(gt=0)
    eff: float = Field(gt=0)  # effective number of generated events
    filter_efficiency: float = 1.0


class CPPlotCfg(_Model):
    var: str
    even: str = "wt_cp_sm"
    odd: str = "wt_cp_ps"


class ComponentFillCfg(_Model):
    """Weighted template fills for one process in the fit signal region.
    Set programmatically by `wham fit`, not written by hand."""

    var: str
    process: str
    components: dict[str, str]  # component name -> weight column/expression


class VariationCfg(_Model):
    """A shape variation filled into the histogram variation axis. Set
    programmatically by `wham fit`, not written by hand.

    target=weight: weight_up/down replace the per-event weight for the listed
    (kind=mc) processes, filling <name>_up / <name>_down slices.
    target=qcd_ff: weight_up/down replace qcd.ff_weight in the anti-iso fills,
    varying the data-driven QCD estimate (<name>_up / <name>_down).
    target=columns: a single <name> slice in which the listed processes are
    refilled with the given columns multiplied by constant factors BEFORE the
    selection and observables are evaluated — cuts on scaled columns migrate
    events across their edges and scaled observables shift (one TES morph
    grid point: m_vis*sqrt(f), pt_2*f)."""

    name: str
    processes: list[str] = []  # concrete process names (weight / columns)
    target: Literal["weight", "qcd_ff", "columns"] = "weight"
    weight_up: str | None = None
    weight_down: str | None = None
    factors: dict[str, float] = {}  # columns target: column -> scale factor

    @model_validator(mode="after")
    def _fields_match_target(self) -> "VariationCfg":
        if self.target == "columns":
            if not self.factors:
                raise ValueError(f"variation '{self.name}': columns needs factors")
            bad = {c: f for c, f in self.factors.items() if f <= 0}
            if bad:
                raise ValueError(f"variation '{self.name}': factors must be > 0, got {bad}")
        else:
            if self.factors:
                raise ValueError(f"variation '{self.name}': factors are columns-only")
            if self.weight_up is None or self.weight_down is None:
                raise ValueError(
                    f"variation '{self.name}': target={self.target} needs weight_up/weight_down"
                )
        return self

    def slice_labels(self) -> list[str]:
        """Histogram variation-axis labels this variation fills."""
        if self.target == "columns":
            return [self.name]
        return [f"{self.name}_up", f"{self.name}_down"]


class Display3DCfg(_Model):
    sample: str
    n_events: int = Field(default=1, gt=0)


class FFClosureCfg(_Model):
    """Optional MUFFIN pass/fail closure plots in a determination region.

    The shape chi-square reported by this diagnostic treats the normalization
    scale as fixed and uses only diagonal statistical variances. It is a compact
    closure metric, not a formal goodness-of-fit test with full covariance.
    """

    enabled: bool = False
    process: str = "QCD"
    selection: str | None = None
    pass_: str | None = Field(default=None, alias="pass")
    fail: str | None = None
    variables: list[str] = []
    metrics: list[Literal["normalization", "shape_chi2", "max_significance"]] = [
        "normalization",
        "shape_chi2",
        "max_significance",
    ]
    weight: str | None = None

    @model_validator(mode="after")
    def _enabled_needs_fields(self) -> "FFClosureCfg":
        if self.enabled and not (self.selection and self.pass_ and self.fail and self.variables):
            raise ValueError(
                "fake_factors.closure enabled=true requires selection, pass, fail and variables"
            )
        return self

    def weight_expr(self) -> str:
        if self.weight is not None:
            return self.weight
        from wham.muffin import score_column

        return score_column(self.process)


class FFEstimateComponentsCfg(_Model):
    QCD: bool = False
    Wjets: bool = False
    ttbar: bool = False

    def enabled(self) -> tuple[str, ...]:
        return tuple(
            name for name in ("QCD", "Wjets", "ttbar")
            if bool(getattr(self, name))
        )


class FFApplicationRegionCfg(_Model):
    selection: str
    pass_: str = Field(alias="pass")
    fail: str


class FFEstimateCfg(_Model):
    enabled: bool = False
    output_process: str | None = None
    components: FFEstimateComponentsCfg = FFEstimateComponentsCfg()
    application_region: FFApplicationRegionCfg | None = None
    fake_cut: str | None = None

    @model_validator(mode="after")
    def _enabled_needs_fields(self) -> "FFEstimateCfg":
        if self.enabled and self.components.enabled() and not (
                self.output_process and self.application_region):
            raise ValueError(
                "fake_factors.estimate with enabled components requires "
                "output_process and application_region"
            )
        if self.enabled and (self.components.Wjets or self.components.ttbar) and not self.components.QCD:
            raise ValueError(
                "fake_factors.estimate.components Wjets=true or ttbar=true requires QCD=true"
            )
        return self

    def active(self) -> bool:
        return self.enabled and bool(self.components.enabled())

    def model_processes(self) -> tuple[str, ...]:
        mapping = {"QCD": "QCD", "Wjets": "Wjets", "ttbar": "ttbarMC"}
        return tuple(mapping[c] for c in self.components.enabled())


class FakeFactorsCfg(_Model):
    """Apply BDT fake-factor models at skim time (see wham/muffin.py)."""

    models: Path                      # the BDTFFModel directory
    channel: str = "mt"
    processes: list[str] = ["QCD"]
    era: str | None = None            # a trained era, e.g. Run3_2022EE
    era_label: int | None = None      # raw label override (2024 borrowing 2023BPix etc.)
    systematics: bool = False         # also write the _up/_down score columns
    closure: FFClosureCfg | None = None
    estimate: FFEstimateCfg | None = None

    @model_validator(mode="after")
    def _era_resolvable(self) -> "FakeFactorsCfg":
        from wham.muffin import ERA_LABELS

        if (self.era is None) == (self.era_label is None):
            raise ValueError("fake_factors needs exactly one of 'era' or 'era_label'")
        if self.era is not None and self.era not in ERA_LABELS:
            raise ValueError(
                f"fake_factors.era {self.era!r} not trained; known: {sorted(ERA_LABELS)}. "
                "Use era_label to borrow the closest era explicitly."
            )
        if not self.models.is_dir():
            raise ValueError(f"fake_factors.models does not exist: {self.models}")
        from wham.muffin import model_file

        if self.estimate is not None and self.estimate.active():
            self.processes = list(dict.fromkeys([*self.processes, *self.estimate.model_processes()]))
        for process in self.processes:
            try:
                model_file(self.models, self.channel, process)
            except FileNotFoundError as e:
                raise ValueError(f"fake_factors: {e}") from None
        if self.closure is not None and self.closure.enabled:
            if self.closure.process not in self.processes:
                raise ValueError(
                    "fake_factors.closure.process must be listed in fake_factors.processes"
                )
        return self

    def resolved_era_label(self) -> int:
        from wham.muffin import ERA_LABELS

        return ERA_LABELS[self.era] if self.era is not None else self.era_label


class StyleCfg(_Model):
    cms_label: str = "Private Work"  # text after "CMS": Preliminary, Simulation, ...
    era: str | None = None           # e.g. "2024" or "Run 3"; shown next to the lumi
    com: float = 13.6                # sqrt(s) in TeV


class PlotsCfg(_Model):
    resolution: list[tuple[str, str]] = []  # [reco, reference] pairs
    datamc: list[str] = []
    datamc_metrics: list[
        Literal["normalization", "shape_chi2", "max_significance"]
    ] = []
    cp: list[CPPlotCfg] = []
    # weighted fit templates in the signal region; set programmatically
    # by `wham fit`, normally not written by hand.
    fitcp: list[ComponentFillCfg] = []
    # anti-iso QCD (data − MC) before vs after the per-event fake-factor
    # weight, with a weighted/raw ratio panel (qcd.method=ff only)
    ffcheck: list[str] = []
    display3d: Display3DCfg | None = None


class AnalysisConfig(_Model):
    name: str
    lumi: float = Field(gt=0)
    data_dir: Path
    output_dir: Path | None = None
    selection: str
    trigger: str | None = None  # applied to datamc/cp fills only
    weight: str = "weight"
    processes: dict[str, ProcessCfg]
    qcd: QCDCfg = QCDCfg()
    sample_params: dict[str, SampleParams] = {}
    variables: dict[str, VariableCfg]
    plots: PlotsCfg = PlotsCfg()
    style: StyleCfg = StyleCfg()
    fake_factors: FakeFactorsCfg | None = None
    # weight-based shape variations; set programmatically by `wham fit`
    variations: list[VariationCfg] = []

    # ---- validators -------------------------------------------------

    @field_validator("data_dir")
    @classmethod
    def _dir_exists(cls, v: Path) -> Path:
        if not v.is_dir():
            raise ValueError(f"data_dir does not exist: {v}")
        return v

    @model_validator(mode="after")
    def _expressions_parse(self) -> "AnalysisConfig":
        for label, src in [
            ("selection", self.selection),
            ("trigger", self.trigger),
            ("weight", self.weight),
            ("qcd.os", self.qcd.os),
            ("qcd.iso", self.qcd.iso),
            ("qcd.antiiso", self.qcd.antiiso),
            ("qcd.ff_weight", self.qcd.ff_weight),
        ]:
            if src is None:
                continue
            try:
                parse(src)
            except ExprError as e:
                raise ValueError(f"invalid expression in '{label}': {e}") from None
        for name, proc in self.processes.items():
            if proc.cut is None:
                continue
            try:
                parse(proc.cut)
            except ExprError as e:
                raise ValueError(f"invalid 'cut' for process '{name}': {e}") from None
        if self.fake_factors is not None and self.fake_factors.closure is not None:
            cl = self.fake_factors.closure
            if cl.enabled:
                for label, src in [
                    ("fake_factors.closure.selection", cl.selection),
                    ("fake_factors.closure.pass", cl.pass_),
                    ("fake_factors.closure.fail", cl.fail),
                    ("fake_factors.closure.weight", cl.weight_expr()),
                ]:
                    try:
                        parse(src)
                    except ExprError as e:
                        raise ValueError(f"invalid expression in '{label}': {e}") from None
        if self.fake_factors is not None and self.fake_factors.estimate is not None:
            est = self.fake_factors.estimate
            if est.enabled:
                ar = est.application_region
                for label, src in [
                    ("fake_factors.estimate.application_region.selection",
                     ar.selection if ar is not None else None),
                    ("fake_factors.estimate.application_region.pass",
                     ar.pass_ if ar is not None else None),
                    ("fake_factors.estimate.application_region.fail",
                     ar.fail if ar is not None else None),
                    ("fake_factors.estimate.fake_cut", est.fake_cut),
                ]:
                    if src is None:
                        continue
                    try:
                        parse(src)
                    except ExprError as e:
                        raise ValueError(f"invalid expression in '{label}': {e}") from None
        return self

    @model_validator(mode="after")
    def _plot_vars_defined(self) -> "AnalysisConfig":
        used: dict[str, str] = {}
        for reco, ref in self.plots.resolution:
            used[reco] = used[ref] = "plots.resolution"
        for v in self.plots.datamc:
            used[v] = "plots.datamc"
        for c in self.plots.cp:
            used[c.var] = "plots.cp"
        for c in self.plots.fitcp:
            used[c.var] = "plots.fitcp"
        for v in self.plots.ffcheck:
            used[v] = "plots.ffcheck"
        if self.fake_factors is not None and self.fake_factors.closure is not None:
            cl = self.fake_factors.closure
            if cl.enabled:
                for v in cl.variables:
                    used[v] = "fake_factors.closure.variables"
        missing = {v: fam for v, fam in used.items() if v not in self.variables}
        if missing:
            listed = ", ".join(f"'{v}' ({fam})" for v, fam in sorted(missing.items()))
            raise ValueError(f"plotted variables not defined in 'variables': {listed}")
        return self

    @model_validator(mode="after")
    def _ffcheck_needs_ff_method(self) -> "AnalysisConfig":
        if self.plots.ffcheck and self.qcd.method != "ff":
            raise ValueError("plots.ffcheck requires qcd.method=ff")
        return self

    @model_validator(mode="after")
    def _unroll_refs_valid(self) -> "AnalysisConfig":
        for name, v in self.variables.items():
            if v.unroll is None:
                continue
            for ref in v.unroll:
                sub = self.variables.get(ref)
                if sub is None:
                    raise ValueError(
                        f"unroll variable '{name}' references undefined variable '{ref}'"
                    )
                if sub.unroll is not None:
                    raise ValueError(
                        f"unroll variable '{name}' references '{ref}', which is "
                        "itself unrolled — nesting is not supported"
                    )
        return self

    @model_validator(mode="after")
    def _datamc_needs_data(self) -> "AnalysisConfig":
        if self.plots.datamc or self.plots.ffcheck:
            n_data = sum(1 for p in self.processes.values() if p.kind == "data")
            if n_data != 1:
                raise ValueError(
                    "plots.datamc/ffcheck require exactly one kind=data process, "
                    f"found {n_data}"
                )
        n_qcd = sum(1 for p in self.processes.values() if p.kind == "qcd")
        if n_qcd > 1:
            raise ValueError(f"at most one kind=qcd process allowed, found {n_qcd}")
        n_ff = sum(1 for p in self.processes.values() if p.kind == "ff")
        if n_ff > 1:
            raise ValueError(f"at most one kind=ff process allowed, found {n_ff}")
        est = self.fake_factors.estimate if self.fake_factors is not None else None
        if est is not None and est.active():
            out = self.processes.get(est.output_process)
            if out is None or out.kind != "ff":
                raise ValueError(
                    "fake_factors.estimate.output_process must name a kind=ff process"
                )
            for component in ("Wjets", "ttbar"):
                if getattr(est.components, component):
                    matches = [
                        name for name, proc in self.processes.items()
                        if proc.ff_component == component
                    ]
                    if not matches:
                        raise ValueError(
                            f"fake_factors.estimate.components.{component}=true "
                            f"requires at least one process with ff_component: {component}"
                        )
        return self

    # ---- derived ----------------------------------------------------

    def resolved_output_dir(self) -> Path:
        from wham import util

        base = self.output_dir if self.output_dir is not None else Path("plots") / self.name
        if not base.is_absolute():
            base = util.repo_root() / base
        return base

    def stack_order(self) -> list[str]:
        """MC + QCD process names in YAML order (= bottom-to-top stack)."""
        return [n for n, p in self.processes.items() if p.kind != "data"]

    def data_process(self) -> str | None:
        for n, p in self.processes.items():
            if p.kind == "data":
                return n
        return None

    def qcd_process(self) -> str | None:
        for n, p in self.processes.items():
            if p.kind == "qcd":
                return n
        return None

    def column_of(self, var: str) -> str:
        """Source column for a variable (aliases resolve via 'column')."""
        vcfg = self.variables.get(var)
        return vcfg.column if (vcfg is not None and vcfg.column) else var

    def columns_of_var(self, var: str) -> set[str]:
        """All source columns a variable reads (unrolled ones read two)."""
        vcfg = self.variables.get(var)
        if vcfg is not None and vcfg.unroll is not None:
            return {self.column_of(ref) for ref in vcfg.unroll}
        return {self.column_of(var)}

    def required_columns(self) -> frozenset[str]:
        """Union of every column any configured fill could touch."""
        cols: set[str] = {"weight", "os"}
        for src in (self.selection, self.trigger, self.weight,
                    self.qcd.os, self.qcd.iso, self.qcd.antiiso, self.qcd.ff_weight):
            if src:
                cols |= parse(src).columns
        for proc in self.processes.values():
            if proc.cut:
                cols |= parse(proc.cut).columns
        for reco, ref in self.plots.resolution:
            cols |= self.columns_of_var(reco) | self.columns_of_var(ref)
        for v in (*self.plots.datamc, *self.plots.ffcheck):
            cols |= self.columns_of_var(v)
        if self.fake_factors is not None and self.fake_factors.estimate is not None:
            est = self.fake_factors.estimate
            if est.enabled and est.application_region is not None:
                ar = est.application_region
                for src in (ar.selection, ar.pass_, ar.fail, est.fake_cut):
                    if src:
                        cols |= parse(src).columns
        if self.fake_factors is not None and self.fake_factors.closure is not None:
            cl = self.fake_factors.closure
            if cl.enabled:
                for src in (cl.selection, cl.pass_, cl.fail, cl.weight_expr()):
                    cols |= parse(src).columns
                for v in cl.variables:
                    cols |= self.columns_of_var(v)
        for c in self.plots.cp:
            cols |= self.columns_of_var(c.var) | {c.even, c.odd}
        for f in self.plots.fitcp:
            cols |= self.columns_of_var(f.var)
            for weight in f.components.values():
                cols |= parse(weight).columns
        for v in self.variations:
            if v.weight_up is not None:
                cols |= parse(v.weight_up).columns | parse(v.weight_down).columns
            cols |= set(v.factors)
        if self.plots.display3d is not None:
            cols |= DISPLAY3D_COLUMNS
        if self.fake_factors is not None:
            from wham.muffin import source_columns

            cols |= source_columns(self.fake_factors)
        return frozenset(cols)


# ---- loading ---------------------------------------------------------


def load_config(path: str | Path) -> AnalysisConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")

    cfg = AnalysisConfig.model_validate(raw)

    if not cfg.sample_params:
        side = path.parent / "params.yaml"
        if side.exists():
            cfg = cfg.model_copy(update={"sample_params": _load_params_file(side)})
    return cfg


def _load_params_file(path: Path) -> dict[str, SampleParams]:
    """Side file in the existing Configurations/*/params.yaml format."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    out: dict[str, SampleParams] = {}
    for name, entry in raw.items():
        if name == "lumi" or not isinstance(entry, dict):
            continue  # analysis YAML owns lumi
        out[name] = SampleParams.model_validate(entry)
    return out


# ---- sample discovery ------------------------------------------------


@dataclass(frozen=True)
class Sample:
    name: str
    process: str
    kind: str  # mc | data
    variation: str
    path: Path
    size: int
    mtime_ns: int
    params: SampleParams | None

    @property
    def scale(self) -> float:
        """Constant factor multiplying the per-event weight (1.0 for data)."""
        return 1.0


def discover_samples(
    cfg: AnalysisConfig, *, variation: str = "nominal"
) -> tuple[list[Sample], list[str]]:
    """Match data_dir subdirectories against process sample patterns.

    Returns (samples, warnings). Raises on ambiguous matches or missing
    MC params — those are config errors, not conditions to limp past.
    """
    candidates = sorted(
        d.name for d in cfg.data_dir.iterdir()
        if d.is_dir() and (d / variation / "merged.parquet").is_file()
    )

    warnings: list[str] = []
    assignment: dict[str, list[str]] = {}
    for proc_name, proc in cfg.processes.items():
        for pattern in proc.samples:
            matched = fnmatch.filter(candidates, pattern)
            if not matched:
                warnings.append(f"pattern '{pattern}' (process '{proc_name}') matched nothing")
            for sample_name in matched:
                procs = assignment.setdefault(sample_name, [])
                if proc_name not in procs:
                    procs.append(proc_name)

    # a sample may feed several processes only when every one of them declares
    # a per-event 'cut' (a genmatch split, e.g. tt -> genuine/l-fake/jet-fake);
    # without cuts the overlap would double count events
    for sample_name, procs in sorted(assignment.items()):
        if len(procs) > 1:
            uncut = [p for p in procs if cfg.processes[p].cut is None]
            if uncut:
                raise ValueError(
                    f"sample '{sample_name}' matches processes {procs}; sharing a "
                    f"sample requires a disjoint 'cut' on every one of them "
                    f"(missing on: {uncut})"
                )

    missing_params = [
        s for s, procs in sorted(assignment.items())
        if any(cfg.processes[p].kind == "mc" for p in procs)
        and s not in cfg.sample_params
    ]
    if missing_params:
        raise ValueError(
            "MC samples without sample_params (xs/eff): " + ", ".join(missing_params)
        )

    samples: list[Sample] = []
    for sample_name, procs in sorted(assignment.items()):
        path = cfg.data_dir / sample_name / variation / "merged.parquet"
        st = path.stat()
        for proc_name in procs:
            samples.append(
                Sample(
                    name=sample_name,
                    process=proc_name,
                    kind=cfg.processes[proc_name].kind,
                    variation=variation,
                    path=path,
                    size=st.st_size,
                    mtime_ns=st.st_mtime_ns,
                    params=cfg.sample_params.get(sample_name),
                )
            )
    return samples, warnings


def sample_scale(sample: Sample, lumi: float) -> float:
    """lumi * xs * filter_efficiency / eff for MC; 1.0 for data."""
    if sample.kind == "data" or sample.params is None:
        return 1.0
    p = sample.params
    return lumi * p.xs * p.filter_efficiency / p.eff
