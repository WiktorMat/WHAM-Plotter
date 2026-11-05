# WHAM-plot

Plotting and fitting framework for the H→ττ (μτ<sub>h</sub>) analysis, Run 3 2024.

The package lives in `source/wham/`. It produces four plot families:

| family | output |
|---|---|
| `datamc`     | Data/MC ratio panels with data-driven QCD (SS or ABCD) and stat. uncertainty bands |
| `resolution` | (reco − ref)/ref distributions for variable pairs |
| `cp`         | CP-even vs CP-odd weighted distributions + integrated asymmetry |
| `display3d`  | 3D event display (muon/tau cones with pion sub-tracks) |

## Quick start (lxplus)

One-time setup, put the launcher on your PATH:

```bash
ln -s /eos/home-h/haawedik/WHAM-plot/bin/wham ~/.local/bin/wham
```

The launcher sources the LCG environment itself, so from any directory:

```bash
wham inspect          # validate config, list samples and cache status
wham plot             # full pipeline
wham render           # restyle from cache (~seconds)
wham fit example      # Combine fit, fully driven by a fit YAML (see Fitting)
wham qcdcompare       # ABCD vs BDT-FF QCD estimate overlay (see BDT fake factors)
wham skim             # build/refresh the skim cache
wham clean            # prune stale skims, report cache sizes
```

With a single YAML in `Configurations/` the config argument is optional;
otherwise name it: `wham plot mutau_2024` (bare name or a path both work).

Useful flags: `--only datamc --var m_vis` (subset), `--workers 8`, `--no-cache`
(full rebuild). Plots land in `plots/<analysis>/<family>/` as PNG + PDF, with a
`histograms.parquet` sidecar per family (long format: variable, process,
region, counts, sumw2, bin edges) for downstream use.

## Performance

Measured on the full 2024 dataset (33 samples, ~41 GB on EOS, 16-core lxplus node):

| scenario | time |
|---|---|
| first ever run (builds the local skim cache once) | ~8 min |
| after changing selection/binning (refill from skims) | ~48 s |
| after changing labels/colors/styles (`render`) | ~20 s, ~6 s for one plot |

The input files have thousands of tiny row groups, so reading them directly
over EOS is latency-bound. `wham` rewrites just the referenced columns into a
local skim cache (`.cache/wham/skims/`, ~5 GB) with large row groups. Skims
carry no event selection, so editing cuts never invalidates them; they rebuild
only when the source files change or new columns are referenced. On top sits a
histogram cache keyed by everything that affects bin contents, so pure style
changes never touch event data.

## Configuration

Everything lives in one pydantic-validated YAML (`Configurations/mutau_2024.yaml`);
per-sample cross-sections/event counts in `params.yaml` next to it.

```yaml
selection: "pt_1 > 26 & pt_2 > 25 & abs(eta_1) < 2.4 & n_bjets == 2"
trigger:   "trg_singlemuon == 1 | trg_mt_cross == 1"

processes:                       # YAML order = stack draw order
  QCD:  {kind: qcd, color: "tab:olive"}            # derived, no samples
  tt:   {samples: ["TTto*", "ST_tW_*"], color: "tab:purple"}
  data: {kind: data, samples: ["Muon0_*", "Muon1_*"], color: black}
# a sample may feed SEVERAL processes when every one declares a disjoint
# per-event `cut` (a genmatch split, e.g. tt -> genuine / l-fake / jet-fake)

qcd:
  method: abcd                   # or: ss (with ff)
  iso: "idDeepTau2018v2p5VSjet_2 >= 5"
  antiiso: "idDeepTau2018v2p5VSjet_2 > 1 & idDeepTau2018v2p5VSjet_2 < 5"

variables:
  m_vis:        {bins: 40, range: [0, 200], label: "$m_{vis}$ [GeV]"}
  m_vis_coarse: {column: m_vis, bins: 10, range: [0, 200]}   # same column, second binning
  m_vis_tails:  {column: m_vis, bins: [0, 50, 70, 90, 120, 200]}  # explicit edges

plots:
  datamc: [m_vis, m_vis_coarse, m_vis_tails, pt_1]
  resolution: [[pt_1, pt_2]]     # [reco, reference]

# Optional cumulative decay-mode plots.  Level 1: inclusive only; level 2:
# pi/rho/a1 versus anything; level 3: also every concrete tt combination.
decay_mode_split:
  level: 3                       # 1, 2, or 3
  channel: tt                    # mt, et, or tt
  column: decayModePNet          # alternatively: decayMode

style:                           # CMS label cosmetics (all optional)
  cms_label: "Private Work"      # or Preliminary, Simulation, ...
  era: "2024"                    # shown next to the lumi
  com: 13.6                      # sqrt(s) in TeV
```

Notes:

- Cut expressions support comparisons, `& | ~` (or `and/or/not`), arithmetic,
  `abs()`, parentheses. Anything else, including typo'd fields or undefined
  plotted variables, fails at load time with a precise error.
- Binning: `bins` is either a count (with `range`) or a list of explicit,
  strictly increasing edges (then omit `range`). To plot one column with
  several binnings, define extra variables with `column:` pointing at the
  source column — each gets its own plot and cached histogram.
- Unrolled 2D discriminants: `{unroll: [x_var, y_var]}` defines a derived
  1D variable with `nx*ny` unit bins (x within blocks of y, y-major).
  It plots like any variable (block separators + y-slice captions drawn
  automatically) and serves as a fit category observable.
- Sample patterns are globs matched against directories in `data_dir`
  (`<data_dir>/<SAMPLE>/nominal/merged.parquet`). Ambiguous matches are errors.
- MC scale = `lumi * xs * filter_efficiency / eff`; per-event `weight` column on top.
- Resolution binning: define a variable named `<reco>_from_<ref>` to customize,
  otherwise defaults apply ((reco−ref)/ref in [−2, 2]; Δ wrapped to [−π, π] for angles).
- Decay-mode splitting is cumulative.  The inclusive plots are always written
  normally.  At level 2, `mt`/`et` add `mu-pi`, `mu-rho`, ... / `e-pi`,
  `e-rho`, ... directories; `tt` adds `pi-anything`, `anything-pi`, etc.
  Level 3 additionally writes all nine `tt` pairs (for example `a1-a1` and
  `a1-rho`) below `split_by_decay_modes/`.  PNet modes 1+2 form `rho` and
  modes 10+11 form `a1`; with `decayMode`, rho is 1 and a1 is 10+11.

## BDT fake factors (muffin)

`wham/muffin.py` (adapted from higgs-dna's `add_bdtfakefactorscores.py`) applies
the XGBoost BDT fake-factor models at skim time — the EOS inputs are never
modified. A `fake_factors:` block in the analysis YAML makes the skims carry
`BDT_FF_score_<process>_sublead` columns (scores for the τ<sub>h</sub> leg),
which then behave like any other column: plot them as variables, or drive a
fake-factor QCD estimate:

```yaml
fake_factors:
  models: /eos/home-h/haawedik/shared-hagop-wiktor-data/muffin_trainings
  channel: mt
  processes: [QCD]          # also: Wjets, WjetsMC, ttbarMC (when trained)
  era_label: 0              # must match the label used in training
  # era: Run3_2023BPix      # alternative: a higgs-dna trained era by name
  # systematics: true       # also write _BkgSub/_Modelling/..._up/_down columns

qcd:
  method: ff                # OS anti-iso data weighted per event by ff_weight,
  iso: "..."                # genuine-tau MC subtracted with the same weight
  antiiso: "..."
  ff_weight: "BDT_FF_score_QCD_sublead"
```

To see what the weighting does, plot the score itself (a `variables:` entry
with `column: BDT_FF_score_QCD_sublead`, range ~[0, 0.45]) and list variables
under `plots.ffcheck:`. Each ffcheck plot overlays the anti-iso `data − MC`
shape raw and FF-weighted (the latter is exactly the QCD estimate), with a
weighted/raw ratio panel — the effective per-bin fake factor. For the score
variable itself that ratio must track the bin centers, a built-in closure check.

`wham qcdcompare` overlays the ABCD and BDT-FF QCD estimates in the signal
region (same datamc fills, two qcd.method variants, both cached) with an
FF/ABCD ratio panel, one plot per datamc variable in `plots/<name>/qcdcompare/`.

Two model layouts are recognized: `<models>/<channel>_<process>/best_model.json`
(our muffin_trainings) and `<models>/model_<channel>_<process>/model.json`
(higgs-dna's BDTFFModel); `temperature_scaling_results.json` next to the model
is picked up automatically. Skims are keyed on the model files, so retraining
triggers exactly one skim rebuild; removing the block returns to the previous
skims untouched.

Each model's saved XGBoost `feature_names` determines its inputs and their order.
QCD and Wjets can use different feature sets (for example, only Wjets may require
`ip_LengthSig`). Skim reads include the union of their inputs. Unsuffixed tau
features refer to leg 2; `_other` refers to leg 1, while explicit `_1`/`_2`
names keep their physical leg numbering. `jpt_pt` is read from `jpt_pt_2` when
available, otherwise calculated from `seeding_jpt_2` (or `jpt_2`) divided by
`pt_2` and clipped at zero. New raw global columns can be used directly by name;
new derived quantities require a recipe in `wham/muffin.py`. Missing required
inputs produce an error identifying the process and feature. Models must retain
their feature names; unnamed positional inputs cannot be inferred safely.

## Cleaning caches

```bash
wham clean            # prune superseded skims, print cache sizes
wham clean --hists    # drop the histogram cache (refilled from skims, ~1 min)
wham clean --skims    # drop the skim cache (rebuilt from EOS, ~8 min)
wham clean --all      # drop everything under .cache/wham/
```

All cached data is derived and rebuilt on demand; deleting it is always safe,
it just costs the rebuild time.

## Fitting (`wham fit`)

Fits run with standalone Combine in an Apptainer container, no CMSSW needed.
The fitter is generic: `wham fit <config>` is the whole interface, and the
fit YAML carries the entire statistical model — there are no fit "modes" in
the code. Start from the fully commented
[`Configurations/fits/TEMPLATE.yaml`](Configurations/fits/TEMPLATE.yaml);
[`example.yaml`](Configurations/fits/example.yaml) is the one runnable
example (a Z→ττ signal-strength fit on m_vis).

A fit config declares:

- **categories** — one entry per datacard bin, fitted simultaneously
  (`imax N`). Each has its own observable and an optional extra `cut`
  (e.g. a decay-mode split) on top of the analysis selection.
- **model.pois** — the parameters of interest, with `init` and `range`.
- **model.processes** — how yields depend on the POIs. A process is either
  scaled whole (`{scale: "r"}`) or split into weighted template components,
  each with its own per-event weight column and scale expression. Anything
  not listed is a plain background.
- **model.morphs** — continuous shape-morphing POIs (e.g. a τ energy
  scale): a grid of template points in `f` where the listed columns are
  scaled by `f` (`linear`) or `sqrt(f)` before the selection is evaluated
  (`scales: {m_vis: sqrt, pt_2: linear}`) — cuts on scaled columns migrate
  events across category edges, so both shape and yield vary along the
  grid. The parquet read window widens automatically; combine morphs the
  templates as a `CMSHistFunc` (requires `combine.cmssw`, see below).
- **systematics** — `lnN` (with `scaleFactor`), `rateParam` (free-floating
  normalization, one shared parameter across everything it matches) and
  `shape` in two flavors: `weight_up`/`weight_down` expressions replace
  the per-event weight of the matched MC processes (the variation also
  propagates into the QCD subtraction — or, matching the QCD process,
  replaces `qcd.ff_weight`); alternatively `scales:` + `shift:` refill the
  matched processes with columns scaled by (1 ± shift) — a constrained
  fake-τ energy scale with the same cut-migration mechanics as a morph
  (this flavor leaves the data-driven QCD template nominal). Optional
  `categories:` restricts one to specific bins. Patterns match the config
  process names (`W+jets` etc.); sanitization and component templates are
  resolved internally.
- **scans** — 1D entries give profiled −2ΔlnL curves, two-POI entries a
  −2ΔlnL heatmap with 68/95% CL contours. Windows auto-center on the best
  fit (± 10σ) unless `range`/`ranges` is given; omitted entirely → one 1D
  scan per POI.
- **asimov / toy** — observed data vs Asimov (`combine -t -1`, truth =
  POI inits overridden by `asimov.parameters`); `toy.asymmetry` modulates
  2-component processes by (1 ± A·cos x) for machinery validation (outputs
  stamped TOY).

A three-POI CP-style fit is pure configuration:

```yaml
model:
  pois:
    mu_ggH: {init: 1, range: [0, 5]}
    mu_qqH: {init: 1, range: [0, 5]}
    alpha:  {init: 0, range: [0, 1.5708]}
  processes:
    ggH:
      components:
        even: {weight: wt_cp_sm, scale: "mu_ggH * cos(alpha)^2"}
        odd:  {weight: wt_cp_ps, scale: "mu_ggH * sin(alpha)^2"}
    qqH:
      components:
        even: {weight: wt_cp_sm, scale: "mu_qqH * cos(alpha)^2"}
        odd:  {weight: wt_cp_ps, scale: "mu_qqH * sin(alpha)^2"}
```

wham turns the model into a generated Combine `PhysicsModel` (shipped to the
output as `whammodel.py`), fills per-category histograms through the same
caches as plotting (QCD via the analysis `qcd.method`), exports
`datacard.txt` + `shapes.root`, runs `text2workspace.py`, `FitDiagnostics`
and one `MultiDimFit` per scan in the container, and renders: prefit/postfit
stacks per category (datamc styling, split components drawn separately), a
pulls plot with per-POI impact columns (covariance approximation; numbers in
`fitresult.json`), and the NLL scans. Outputs go to
`plots/<analysis>/fit/<name>/`, including the effective `fitconfig.yaml`.

Every stage is keyed: editing a systematic re-exports and refits in seconds
without touching event data; rerunning with nothing changed only re-renders.
Flags: `--datacard-only`, `--force`, `--no-render`. The `pois` plot overlays
the FitDiagnostics values with the scan-profile intervals — the two must
agree, and a console warning flags a local-minimum fit when they don't.

`wham fitsummary <fit1> <fit2> ... [--var pt_2] [--label "..."]` overlays
the POIs of several finished fits on one canvas: per-category rate POIs vs
the window their category cut puts on `--var` (e.g. ID SFs vs pT, one series
per decay mode) plus the morph POIs per fit, with profile errors where scans
exist. It only reads existing fit outputs.


## Repository layout

```
Configurations/        analysis YAMLs + params.yaml; fits/ for fit configs
source/wham/           the package (see its README for internals)
source/wham/tests/     pytest suite (synthetic-parquet based)
bin/wham               launcher
scripts/tools/         merge_parquet.py (standalone parquet merger)
plots/                 outputs (gitignored)
.cache/wham/           skim + histogram caches (gitignored, safe to delete)
```

## Tests

```bash
cd source && python3 -m pytest wham/tests/ -q          # unit tests
cd source && python3 -m pytest wham/tests/ --override-ini "addopts=" -q   # + container test
```

## Notes

- Systematics: histograms already carry a `variation` axis ("nominal"); reading
  variation folders other than `nominal/` is an additive change.
- The LCG pyarrow build lacks zstd, so parquet outputs use snappy.

