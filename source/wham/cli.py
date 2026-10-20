"""wham command line interface."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from wham import util
from wham.config import AnalysisConfig, Sample, discover_samples, load_config, sample_scale

console = Console()


def _resolve_config(config: str | None) -> Path:
    """Accept a path, a bare name (Configurations/<name>.yaml), or nothing
    (when exactly one analysis YAML exists)."""
    config_dir = util.repo_root() / "Configurations"

    if config is None:
        candidates = [p for p in sorted(config_dir.glob("*.yaml")) if p.name != "params.yaml"]
        if len(candidates) == 1:
            return candidates[0]
        names = ", ".join(p.stem for p in candidates) or "<none>"
        console.print(
            f"[red bold]Choose a config:[/red bold] found {len(candidates)} in "
            f"{config_dir} ({names}). Run e.g. `wham plot <name>`."
        )
        sys.exit(1)

    path = Path(config)
    if path.is_file():
        return path
    candidate = config_dir / f"{config}.yaml"
    if candidate.is_file():
        return candidate
    console.print(
        f"[red bold]Config not found:[/red bold] no file '{config}' and no "
        f"{candidate}"
    )
    sys.exit(1)


def _load(config: str | None) -> tuple[AnalysisConfig, list[Sample]]:
    config_path = _resolve_config(config)
    try:
        cfg = load_config(config_path)
    except Exception as e:
        console.print(f"[red bold]Config error:[/red bold] {e}")
        sys.exit(1)
    try:
        samples, warnings = discover_samples(cfg)
    except Exception as e:
        console.print(f"[red bold]Sample discovery error:[/red bold] {e}")
        sys.exit(1)
    for w in warnings:
        console.print(f"[yellow]Warning:[/yellow] {w}")
    if not samples:
        console.print("[red bold]No samples found.[/red bold]")
        sys.exit(1)
    return cfg, samples


def _families(cfg: AnalysisConfig, only: tuple[str, ...]) -> list[str]:
    configured = []
    if cfg.plots.resolution:
        configured.append("resolution")
    if cfg.plots.datamc:
        configured.append("datamc")
    if cfg.plots.cp:
        configured.append("cp")
    if cfg.plots.ffcheck:
        configured.append("ffcheck")
    if (cfg.fake_factors is not None and cfg.fake_factors.closure is not None
            and cfg.fake_factors.closure.enabled):
        configured.append("ffclosure")
    if cfg.plots.display3d is not None:
        configured.append("display3d")
    if not only:
        return configured
    unknown = set(only) - set(configured)
    if unknown:
        console.print(
            f"[red bold]Unknown/unconfigured families:[/red bold] {sorted(unknown)} "
            f"(configured: {configured})"
        )
        sys.exit(1)
    return [f for f in configured if f in only]


def _skim_progress(total: int):
    """Progress printer for skim builds inside plot/fit (they can take minutes)."""
    done = {"n": 0}

    def _cb(info) -> None:
        done["n"] += 1
        console.print(
            f"  [green]skimmed[/green] {info.sample} ({done['n']}/{total}): "
            f"{info.rows:,} rows -> {info.path.stat().st_size / 1e6:,.0f} MB"
        )

    return _cb


def _render_decay_mode_splits(
    cfg, samples, skims, *, families, only_vars, workers, use_cache,
    cache_only=False,
) -> None:
    """Fill/load and render every cumulative decay-mode category."""
    from wham.decay_modes import category_config, decay_mode_categories
    from wham.fill import fill_all
    from wham.render import render_families

    for category in decay_mode_categories(cfg):
        split_cfg = category_config(cfg, category)
        console.print(f"Decay-mode category: [bold]{category.name}[/bold]")
        hists = fill_all(
            split_cfg, samples, skims,
            families=families, only_vars=only_vars,
            workers=workers, use_cache=use_cache, cache_only=cache_only,
            console=console,
        )
        render_families(
            split_cfg, hists, skims, families=families,
            only_vars=only_vars, outdir=split_cfg.resolved_output_dir(), console=console,
        )


@click.group()
def main() -> None:
    """Fast H->tautau plotter: validated configs, skim + histogram caches."""


# ---------------------------------------------------------------- inspect


@main.command()
@click.argument("config_path", required=False)
@click.option("--columns", "show_columns", is_flag=True, help="Show required column availability.")
@click.option("--yields", "show_yields", is_flag=True, help="Show per-process yields from the histogram cache.")
def inspect(config_path: str, show_columns: bool, show_yields: bool) -> None:
    """Validate the config and show samples, skims and (optionally) yields."""
    from wham.muffin import signature as muffin_signature
    from wham.skim import find_skim

    cfg, samples = _load(config_path)
    required = cfg.required_columns()
    ff_sig = muffin_signature(cfg.fake_factors)

    table = Table(title=f"{cfg.name} — {len(samples)} samples")
    table.add_column("Sample")
    table.add_column("Process")
    table.add_column("Kind")
    table.add_column("Size", justify="right")
    table.add_column("Scale (lumi*xs*f/eff)", justify="right")
    table.add_column("Skim")

    for s in samples:
        info = find_skim(cfg.name, s, required, ff_sig)
        scale = sample_scale(s, cfg.lumi)
        table.add_row(
            s.name,
            s.process,
            s.kind,
            f"{s.size / 1e6:,.0f} MB",
            "—" if s.kind == "data" else f"{scale:.3e}",
            "[green]fresh[/green]" if info else "[yellow]missing[/yellow]",
        )
    console.print(table)
    console.print(
        f"Families configured: {_families(cfg, ())} | "
        f"required columns: {len(required)} | lumi: {cfg.lumi:g} /pb"
    )

    if show_columns:
        import pyarrow.parquet as pq

        schema = set(pq.read_schema(samples[0].path).names)
        col_table = Table(title=f"Required columns (vs schema of {samples[0].name})")
        col_table.add_column("Column")
        col_table.add_column("Present")
        for col in sorted(required):
            col_table.add_row(
                col, "[green]yes[/green]" if col in schema else "[red]NO[/red]"
            )
        console.print(col_table)

    if show_yields:
        from wham.histcache import load_all_cached
        from wham.fill import REGION_NOMINAL

        cached = load_all_cached(cfg, samples)
        if not cached:
            console.print("[yellow]No cached histograms — run `wham plot` first.[/yellow]")
            return
        ytable = Table(title="Yields (from histogram cache)")
        ytable.add_column("Family/Variable")
        ytable.add_column("Process")
        ytable.add_column("Region")
        ytable.add_column("Yield", justify="right")
        for (family, var), h in sorted(cached.items()):
            for proc in h.axes["process"]:
                for region in h.axes["region"]:
                    val = h[{"process": proc, "region": region, "variation": "nominal"}]
                    total = float(val.view()["value"].sum())
                    if total != 0:
                        ytable.add_row(f"{family}/{var}", proc, region, f"{total:,.1f}")
        console.print(ytable)


# ---------------------------------------------------------------- skim


@main.command()
@click.argument("config_path", required=False)
@click.option("--workers", default=6, show_default=True)
@click.option("--force", is_flag=True, help="Rebuild even if a valid skim exists.")
@click.option("--prune", is_flag=True, help="Delete superseded skims afterwards.")
def skim(config_path: str, workers: int, force: bool, prune: bool) -> None:
    """Build/refresh the local column-pruned skim cache."""
    from wham.muffin import signature as muffin_signature
    from wham.skim import ensure_skims, prune_skims

    cfg, samples = _load(config_path)

    t0 = time.perf_counter()
    done = {"n": 0}

    def _progress(info) -> None:
        done["n"] += 1
        console.print(
            f"  [green]skimmed[/green] {info.sample}: {info.rows:,} rows, "
            f"{len(info.columns)} cols -> {info.path.stat().st_size / 1e6:,.0f} MB"
        )

    skims = ensure_skims(cfg, samples, workers=workers, force=force, on_progress=_progress)
    dt = time.perf_counter() - t0

    total = sum(i.path.stat().st_size for i in skims.values())
    console.print(
        f"Skims ready: {len(skims)}/{len(samples)} samples | built {done['n']} | "
        f"total {total / 1e9:.2f} GB | {dt:.1f}s"
    )

    if prune:
        freed = prune_skims(cfg.name, samples, cfg.required_columns(),
                        muffin_signature(cfg.fake_factors))
        console.print(f"Pruned {freed / 1e6:,.0f} MB of stale skims")


# ---------------------------------------------------------------- clean


def _dir_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@main.command()
@click.argument("config_path", required=False)
@click.option("--hists", "wipe_hists", is_flag=True,
              help="Delete the histogram cache (refilled from skims in ~1 min).")
@click.option("--skims", "wipe_skims", is_flag=True,
              help="Delete the skim cache (rebuilt from EOS, ~minutes).")
@click.option("--all", "wipe_all", is_flag=True, help="Delete the whole cache directory.")
def clean(config_path: str, wipe_hists: bool, wipe_skims: bool, wipe_all: bool) -> None:
    """Clean caches. Without flags: prune superseded skims and report sizes."""
    import shutil

    root = util.cache_root()
    if wipe_all:
        wipe_hists = wipe_skims = True

    if wipe_hists or wipe_skims:
        for enabled, sub in ((wipe_skims, "skims"), (wipe_hists, "hists")):
            target = root / sub
            if not enabled:
                continue
            if not target.is_dir():
                console.print(f"  {target}: nothing to remove")
                continue
            size = _dir_size(target)
            shutil.rmtree(target)
            console.print(f"  [green]removed[/green] {target} ({size / 1e9:.2f} GB)")
        return

    from wham.muffin import signature as muffin_signature
    from wham.skim import prune_skims

    cfg, samples = _load(config_path)
    freed = prune_skims(cfg.name, samples, cfg.required_columns(),
                        muffin_signature(cfg.fake_factors))
    console.print(f"Pruned {freed / 1e6:,.0f} MB of superseded skims")
    for sub in ("skims", "hists"):
        console.print(f"  {root / sub}: {_dir_size(root / sub) / 1e9:.2f} GB")


# ---------------------------------------------------------------- plot


@main.command()
@click.argument("config_path", required=False)
@click.option("--only", multiple=True, help="Limit to plot families (repeatable).")
@click.option("--var", "only_vars", multiple=True, help="Limit to variables (repeatable).")
@click.option("--workers", default=6, show_default=True)
@click.option("--no-cache", is_flag=True, help="Rebuild skims and histograms from scratch.")
def plot(config_path: str, only: tuple[str, ...], only_vars: tuple[str, ...],
         workers: int, no_cache: bool) -> None:
    """Full pipeline: skim -> fill -> QCD -> render (cache-aware)."""
    from wham.fill import fill_all
    from wham.render import render_families
    from wham.skim import ensure_skims

    cfg, samples = _load(config_path)
    families = _families(cfg, only)

    t0 = time.perf_counter()
    skims = ensure_skims(cfg, samples, workers=workers, force=no_cache,
                         on_progress=_skim_progress(len(samples)))
    t_skim = time.perf_counter()
    console.print(f"Skims ready in {t_skim - t0:.1f}s")

    hists = fill_all(
        cfg, samples, skims,
        families=families, only_vars=only_vars or None,
        workers=workers, use_cache=not no_cache, console=console,
    )
    t_fill = time.perf_counter()
    console.print(f"Histograms ready in {t_fill - t_skim:.1f}s")

    outdir = cfg.resolved_output_dir()
    render_families(cfg, hists, skims, families=families,
                    only_vars=only_vars or None, outdir=outdir, console=console)
    _render_decay_mode_splits(
        cfg, samples, skims, families=families, only_vars=only_vars or None,
        workers=workers, use_cache=not no_cache,
    )
    console.print(
        f"[bold green]Done[/bold green] in {time.perf_counter() - t0:.1f}s -> {outdir}/"
    )


# ---------------------------------------------------------------- qcdcompare


@main.command()
@click.argument("config_path", required=False)
@click.option("--var", "only_vars", multiple=True, help="Limit to variables (repeatable).")
@click.option("--workers", default=6, show_default=True)
def qcdcompare(config_path: str, only_vars: tuple[str, ...], workers: int) -> None:
    """Compare the ABCD and BDT-FF QCD estimates in the signal region."""
    from wham.fill import fill_all
    from wham.render.common import set_style
    from wham.render.qcdcompare import render_qcdcompare
    from wham.skim import ensure_skims

    cfg, samples = _load(config_path)
    if not (cfg.qcd.iso and cfg.qcd.antiiso and cfg.qcd.ff_weight):
        console.print(
            "[red bold]qcdcompare needs qcd.iso, qcd.antiiso and qcd.ff_weight[/red bold] "
            "so both the ABCD and FF estimates are defined."
        )
        sys.exit(1)

    def variant(method: str) -> AnalysisConfig:
        return cfg.model_copy(update={"qcd": cfg.qcd.model_copy(update={"method": method})})

    t0 = time.perf_counter()
    skims = ensure_skims(cfg, samples, workers=workers,
                         on_progress=_skim_progress(len(samples)))
    # ff variant last so the datamc sidecar parquet matches the YAML's method
    hists_abcd = fill_all(variant("abcd"), samples, skims, families=["datamc"],
                          only_vars=only_vars or None, workers=workers, console=console)
    hists_ff = fill_all(variant("ff"), samples, skims, families=["datamc"],
                        only_vars=only_vars or None, workers=workers, console=console)
    console.print(f"Histograms ready in {time.perf_counter() - t0:.1f}s")

    set_style()
    outdir = cfg.resolved_output_dir()
    render_qcdcompare(cfg, hists_abcd, hists_ff, outdir, only_vars or None, console)
    console.print(
        f"[bold green]Done[/bold green] in {time.perf_counter() - t0:.1f}s -> {outdir}/qcdcompare/"
    )


# ---------------------------------------------------------------- render


@main.command()
@click.argument("config_path", required=False)
@click.option("--only", multiple=True, help="Limit to plot families (repeatable).")
@click.option("--var", "only_vars", multiple=True, help="Limit to variables (repeatable).")
def render(config_path: str, only: tuple[str, ...], only_vars: tuple[str, ...]) -> None:
    """Re-render plots from cached histograms only (seconds; no event data)."""
    from wham.fill import fill_all
    from wham.muffin import signature as muffin_signature
    from wham.render import render_families
    from wham.skim import find_skim

    cfg, samples = _load(config_path)
    families = _families(cfg, only)
    required = cfg.required_columns()
    ff_sig = muffin_signature(cfg.fake_factors)

    skims = {}
    for s in samples:
        info = find_skim(cfg.name, s, required, ff_sig)
        if info is None:
            console.print(
                f"[red bold]No skim for {s.name}[/red bold] — histogram cache cannot be "
                "validated. Run `wham plot` first."
            )
            sys.exit(1)
        skims[s.name] = info

    hists = fill_all(
        cfg, samples, skims,
        families=families, only_vars=only_vars or None,
        workers=1, use_cache=True, cache_only=True, console=console,
    )
    outdir = cfg.resolved_output_dir()
    render_families(cfg, hists, skims, families=families,
                    only_vars=only_vars or None, outdir=outdir, console=console)
    _render_decay_mode_splits(
        cfg, samples, skims, families=families, only_vars=only_vars or None,
        workers=1, use_cache=True, cache_only=True,
    )
    console.print(f"[bold green]Rendered[/bold green] -> {outdir}/")


# ---------------------------------------------------------------- fit


@main.command()
@click.argument("config", required=False)
@click.option("--workers", default=6, show_default=True)
@click.option("--no-cache", is_flag=True, help="Rebuild skims and histograms from scratch.")
@click.option("--force", is_flag=True, help="Rerun all combine stages even if fresh.")
@click.option("--datacard-only", is_flag=True, help="Stop after exporting datacard + shapes.")
@click.option("--shapes-only", is_flag=True, help="Stop after exporting shapes and optional external staging.")
@click.option("--no-render", is_flag=True, help="Skip prefit/postfit/pulls/NLL plots.")
def fit(config: str, workers: int, no_cache: bool,
        force: bool, datacard_only: bool, shapes_only: bool, no_render: bool) -> None:
    """Datacard export + Combine fit, fully driven by a fit config.

    CONFIG is a fit YAML — a path, or a bare name resolved in
    Configurations/fits/. The YAML declares everything: POIs, the yield
    model per process, categories, systematics and scans. Start from
    Configurations/fits/TEMPLATE.yaml.
    """
    from wham.combine import run_fit
    from wham.fill import build_fill_spec, fill_all, hist_keys, spec_extras
    from wham.fitconfig import (
        category_analysis,
        fit_families,
        load_fit_config,
        resolve_fit_config,
        union_analysis,
    )
    from wham.histcache import cache_key
    from wham.skim import ensure_skims

    if datacard_only and shapes_only:
        console.print("[red bold]Choose either --datacard-only or --shapes-only, not both.[/red bold]")
        sys.exit(1)

    try:
        fit_path = resolve_fit_config(config)
        fit_cfg, base_cfg, cat_cfgs = load_fit_config(fit_path)
    except Exception as e:
        console.print(f"[red bold]Fit config error:[/red bold] {e}")
        sys.exit(1)

    families = fit_families(fit_cfg)
    console.print(
        f"[bold]{fit_cfg.name}[/bold]: POIs {', '.join(fit_cfg.model.pois)} | "
        f"categories {', '.join(c.name for c in fit_cfg.categories)}"
        + (" | [crimson]Asimov[/crimson]" if fit_cfg.asimov.enabled else "")
        + (f" | [crimson]TOY A={fit_cfg.toy.asymmetry:g}[/crimson]"
           if fit_cfg.toy.asymmetry else "")
    )

    # group categories by analysis (control categories draw from their own
    # config): one sample-discovery + skim pass per analysis, then the
    # per-category fills — each cached under its own analysis name
    groups: dict[int, tuple] = {}
    for cat in fit_cfg.categories:
        acfg = cat_cfgs[cat.name]
        groups.setdefault(id(acfg), (acfg, []))[1].append(cat)

    t0 = time.perf_counter()
    hists_by_cat: dict[str, dict] = {}
    input_keys: dict[tuple[str, str, str], str] = {}
    for acfg, cats in groups.values():
        try:
            samples, warnings = discover_samples(acfg)
        except Exception as e:
            console.print(
                f"[red bold]Sample discovery error[/red bold] ({acfg.name}): {e}")
            sys.exit(1)
        for w in warnings:
            console.print(f"[yellow]Warning:[/yellow] {w}")

        # one skim pass covering this analysis's category columns
        union_cfg = union_analysis(fit_cfg, acfg, cats)
        skims = ensure_skims(union_cfg, samples, workers=workers, force=no_cache,
                             on_progress=_skim_progress(len(samples)))

        for cat in cats:
            ccfg = category_analysis(fit_cfg, acfg, cat)
            hists_by_cat[cat.name] = fill_all(
                ccfg, samples, skims, families=families, only_vars=(cat.variable,),
                workers=workers, use_cache=not no_cache, sidecars=False, console=console,
            )
            spec = build_fill_spec(ccfg, families=families, only_vars=(cat.variable,))
            extras = spec_extras(spec)
            for family, name, vcfg in hist_keys(spec):
                input_keys[(cat.name, family, name)] = cache_key(
                    ccfg, samples, skims, spec, family, name, vcfg,
                    extra=extras.get((family, name)),
                )
    console.print(f"Histograms ready in {time.perf_counter() - t0:.1f}s")

    try:
        fitdir = run_fit(fit_cfg, base_cfg, hists_by_cat, input_keys, cat_cfgs=cat_cfgs, console=console,
                         force=force, datacard_only=datacard_only, shapes_only=shapes_only)
    except Exception as e:
        console.print(f"[red bold]Fit failed:[/red bold] {e}")
        sys.exit(1)

    if datacard_only:
        console.print(f"[bold green]Datacard ready[/bold green] -> {fitdir}/")
        return
    if shapes_only:
        console.print(f"[bold green]Shapes ready[/bold green] -> {fitdir}/")
        return

    if not no_render:
        from wham.render.fit import render_fit

        render_fit(fit_cfg, base_cfg, fitdir, console, cat_cfgs=cat_cfgs)

    console.print(
        f"[bold green]Fit done[/bold green] in {time.perf_counter() - t0:.1f}s -> {fitdir}/"
    )


@main.command()
@click.argument("configs", nargs=-1, required=True)
@click.option("--var", default="pt_2", show_default=True,
              help="Variable whose category-cut window sets the x axis.")
@click.option("--label", default=None,
              help="Annotation text, e.g. the VSjet working point.")
def fitsummary(configs: tuple[str, ...], var: str, label: str | None) -> None:
    """Overlay the POIs of several finished fits on one canvas: per-category
    rate POIs vs their VAR windows (e.g. ID SFs vs pT, one series per decay
    mode) + the morph POIs (TES) per fit. Reads existing fit outputs only —
    run `wham fit` for each config first."""
    from wham.fitconfig import load_fit_config, resolve_fit_config
    from wham.render.fit import render_fit_summary

    entries = []
    for c in configs:
        try:
            fit_cfg, base_cfg, _ = load_fit_config(resolve_fit_config(c))
        except (ValueError, FileNotFoundError) as e:
            console.print(f"[red bold]Config error[/red bold] ({c}): {e}")
            sys.exit(1)
        fitdir = base_cfg.resolved_output_dir() / "fit" / fit_cfg.name
        entries.append((fit_cfg, base_cfg, fitdir))

    outdir = entries[0][1].resolved_output_dir() / "fit"
    render_fit_summary(entries, outdir, var, label, console)


@main.command()
@click.argument("config")
@click.option("--strict", is_flag=True,
              help="Turn cross-fit consistency warnings into errors.")
def export(config: str, strict: bool) -> None:
    """Write external-fitter input ROOT files (e.g. TauFW Fitter inputs) from
    the shapes.root of finished fits. CONFIG is an export YAML listing the
    fits and, per output file, the bin/process/systematic renames, sums and
    rebins (see Configurations/tau_sf/*/taufw_export.yaml). Reads fit
    outputs only — run `wham fit` for each fit first."""
    from wham.export import load_export_config, run_export

    path = Path(config)
    if not path.is_file():
        console.print(f"[red bold]Export config error:[/red bold] no file '{config}'")
        sys.exit(1)
    try:
        cfg = load_export_config(path)
    except Exception as e:
        console.print(f"[red bold]Export config error:[/red bold] {e}")
        sys.exit(1)
    try:
        run_export(cfg, path.parent.resolve(), strict=strict, console=console)
    except Exception as e:
        console.print(f"[red bold]Export failed:[/red bold] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
