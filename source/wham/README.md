# wham

Fast H→ττ analysis plotter. One validated YAML per analysis, two-level caching,
CMS-style plots via mplhep.

## Quick start

The `bin/wham` launcher handles the LCG environment and PYTHONPATH; symlink it
once into `~/.local/bin` and run from anywhere:

```bash
wham inspect                # validate config, list samples
wham plot                   # full pipeline (config optional if only one YAML exists)
wham plot mutau_2024        # bare names resolve to Configurations/<name>.yaml
wham render                 # style tweaks, seconds
wham skim --prune           # manage skim cache
wham fit <fit.yaml>         # datacard export + Combine fit (see Configurations/fits/TEMPLATE.yaml)
wham export <export.yaml>   # rename/merge finished fit shapes into external-fitter inputs (TauFW)
```

Useful flags: `--only datamc --var m_vis` (subset), `--workers 8`, `--no-cache` (full rebuild).
Without the launcher: source the LCG view, `export PYTHONPATH=$REPO/source:$PYTHONPATH`,
then `python3 -m wham.cli ...`.

## How it stays fast

1. **Skim cache** (`.cache/wham/skims/`): the input parquet files have thousands
   of tiny row groups on EOS — terrible to read. The first run rewrites just the
   ~25 referenced columns locally with large row groups (~5.5 GB total, ~7 min once).
   Skims carry no event selection, so **editing cuts never rebuilds them**; they
   invalidate automatically when the source files change (size/mtime) or when you
   reference new columns.
2. **Histogram cache** (`.cache/wham/hists/`): filled histograms keyed by
   everything that affects bin contents (cuts, weights, binning, sample list, lumi).
   Label/color/style edits hit the cache → `render` takes seconds.
   Changing the selection refills from skims (~25 s) without touching EOS.

## Config anatomy (see `Configurations/mutau_2024.yaml`)

- `selection`, `trigger`, qcd region cuts are plain expressions:
  `pt_1 > 26 & abs(eta_1) < 2.4 & n_bjets == 2`. Comparisons, `& | ~`
  (or `and/or/not`), arithmetic and `abs()` are supported; anything else is
  rejected at load time with a pointed error.
- `processes`: YAML order = stack order. `kind: data` for data, `kind: qcd`
  for the data-driven QCD slot (no samples). `samples` are glob patterns matched
  against sample directories in `data_dir`.
- Per-sample `xs`/`eff` are read from `params.yaml` next to the analysis YAML
  (or inline under `sample_params`). MC scale = `lumi * xs * filter_efficiency / eff`.
- `plots`: families `resolution` ([reco, ref] pairs), `datamc`,
  `cp` (CP-even vs CP-odd weights), `display3d`.
- `plots.datamc_blind: true` hides signal-region data, the Data/MC ratio and
  data-vs-prediction metrics only while rendering. Data remain available to
  the data-driven QCD/jet-fake estimate.
- Resolution binning: define a variable named `<reco>_from_<ref>` to control it,
  otherwise a sensible default is used ((reco−ref)/ref in [−2, 2]).

QCD estimation: `method: ss` (SS→OS with `ff`) or `method: abcd`
(per-bin transfer factor from anti-isolated regions), matching the old plotter
bin-by-bin (verified against it before its removal).

## Tests

```bash
python3 -m pytest source/wham/tests/ -q
```
