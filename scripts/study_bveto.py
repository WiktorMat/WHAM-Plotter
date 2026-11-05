#!/usr/bin/env python3
"""Blinded b-veto optimisation using the standard WHAM skim/fill/QCD path.

Observed signal-region data never leave ``fill_all``: they are needed to build
the data-driven QCD estimate, but this script extracts only MC and derived
background processes and disables histogram-cache/sidecar writes.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from wham.config import AnalysisConfig, discover_samples, load_config  # noqa: E402
from wham.fill import fill_all  # noqa: E402
from wham.skim import ensure_skims  # noqa: E402

WPS = ("L", "M", "T", "XT", "XXT")
SCHEMES = ("nominal", "alternative")
CUTS = {"no_veto": 3, "max1": 2, "max0": 1}


@dataclass(frozen=True)
class StudySetup:
    signal: str
    ttbar: str
    data: str
    region: str
    backgrounds: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", nargs="?",
        default=str(REPO / "Configurations" / "mutau_2024_bveto_study.yaml"),
        help="study YAML (default: Configurations/mutau_2024_bveto_study.yaml)",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--signal", default="Higgs", help="signal process name")
    parser.add_argument("--ttbar", default="tt", help="top process used for rejection")
    parser.add_argument("--output", type=Path, help="override YAML output_dir")
    return parser.parse_args()


def _validate(cfg: AnalysisConfig, signal: str, ttbar: str) -> StudySetup:
    if not cfg.plots.datamc_blind:
        raise RuntimeError(
            "Refusing to run an unblinded b-veto study: set "
            "plots.datamc_blind: true"
        )
    if signal not in cfg.processes or cfg.processes[signal].kind != "mc":
        raise ValueError(f"signal process {signal!r} is absent or is not MC")
    if ttbar not in cfg.processes or cfg.processes[ttbar].kind != "mc":
        raise ValueError(f"top process {ttbar!r} is absent or is not MC")
    data = cfg.data_process()
    if data is None:
        raise ValueError("data-driven QCD requires a kind=data process")
    derived = [name for name, proc in cfg.processes.items() if proc.kind == "qcd"]
    if not derived:
        raise ValueError("study requires a data-driven kind=qcd process")
    if cfg.qcd.method not in ("abcd", "ff", "ss"):
        raise ValueError(f"unsupported QCD method: {cfg.qcd.method}")

    expected = {f"bveto_{wp}_{scheme}" for wp in WPS for scheme in SCHEMES}
    missing = expected - set(cfg.plots.datamc)
    if missing:
        raise ValueError("plots.datamc misses study variables: " + ", ".join(sorted(missing)))
    for scheme in SCHEMES:
        score = cfg.variables.get(f"bdt_{scheme}")
        if score is None or score.unroll is not None:
            raise ValueError(f"missing plain score variable bdt_{scheme}")
    region = "OS" if cfg.qcd.method == "ss" else "OS_iso"
    backgrounds = tuple(
        name for name in cfg.processes if name not in (signal, data)
    )
    return StudySetup(signal, ttbar, data, region, backgrounds)


def _score_edges(cfg: AnalysisConfig, scheme: str) -> np.ndarray:
    return np.asarray(cfg.variables[f"bdt_{scheme}"].edges(), dtype=float)


def _process_grid(h, process: str, region: str, n_score_bins: int) -> np.ndarray:
    """Return [score bin, b-tag category] for a non-data process."""
    values = h[{"process": process, "region": region, "variation": "nominal"}]
    flat = np.asarray(values.view()["value"], dtype=float)
    expected = n_score_bins * 3
    if flat.size != expected:
        raise RuntimeError(f"unexpected unrolled size {flat.size}; expected {expected}")
    return flat.reshape(n_score_bins, 3)


def _extract_blinded_yields(cfg, hists, setup: StudySetup):
    """Copy only MC/derived predictions; never copy or return observed data."""
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    allowed = (setup.signal, *setup.backgrounds)
    if setup.data in allowed:  # defensive invariant against future edits
        raise RuntimeError("internal blinding invariant violated")
    for scheme in SCHEMES:
        ny = len(_score_edges(cfg, scheme)) - 1
        for wp in WPS:
            h = hists[("datamc", f"bveto_{wp}_{scheme}")]
            if setup.region not in h.axes["region"]:
                raise RuntimeError(f"histogram lacks signal region {setup.region}")
            out[(scheme, wp)] = {
                process: _process_grid(h, process, setup.region, ny).copy()
                for process in allowed
            }
    return out


def _z_asimov(signal: float, background: float) -> float:
    if signal < 0 or background <= 0:
        return math.nan
    if signal == 0:
        return 0.0
    return math.sqrt(2.0 * ((signal + background) * math.log1p(signal / background) - signal))


def _metrics(signal: float, background: float) -> tuple[float, float, float]:
    if background <= 0 or not np.isfinite(background):
        return math.nan, math.nan, math.nan
    return signal / background, signal / math.sqrt(background), _z_asimov(signal, background)


def _build_rows(cfg, yields, setup: StudySetup) -> list[dict]:
    rows: list[dict] = []
    for scheme in SCHEMES:
        edges = _score_edges(cfg, scheme)
        for wp in WPS:
            grids = yields[(scheme, wp)]
            for cut, stop in CUTS.items():
                sig = grids[setup.signal][:, :stop].sum(axis=1)
                tt = grids[setup.ttbar][:, :stop].sum(axis=1)
                bkg = sum(grids[p][:, :stop] for p in setup.backgrounds).sum(axis=1)
                for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                    s, b = float(sig[i]), float(bkg[i])
                    sb, z, za = _metrics(s, b)
                    rows.append({
                        "scheme": scheme, "wp": wp, "cut": cut,
                        "bdt_low": lo, "bdt_high": hi,
                        "signal": s, "background": b, "ttbar": float(tt[i]),
                        "s_over_b": sb, "s_over_sqrt_b": z, "z_asimov": za,
                    })
    return rows


def _build_roc_rows(yields, setup: StudySetup) -> list[dict]:
    rows: list[dict] = []
    # The ROC integrates the complete [0.33, 1] score range. Its MC signal/top
    # efficiencies are independent of how that same range is subdivided, so use
    # one scheme and draw exactly the requested max1 and max0 curves.
    scheme = "nominal"
    for wp in WPS:
        grids = yields[(scheme, wp)]
        s0 = float(grids[setup.signal].sum())
        t0 = float(grids[setup.ttbar].sum())
        for cut, stop in (("max1", 2), ("max0", 1)):
            s = float(grids[setup.signal][:, :stop].sum())
            t = float(grids[setup.ttbar][:, :stop].sum())
            rows.append({
                "wp": wp, "cut": cut,
                "signal_efficiency": s / s0 if s0 > 0 else math.nan,
                "ttbar_efficiency": t / t0 if t0 > 0 else math.nan,
                "ttbar_rejection": 1.0 - t / t0 if t0 > 0 else math.nan,
            })
    return rows


def _build_summary_rows(yields, setup: StudySetup) -> list[dict]:
    rows: list[dict] = []
    for scheme in SCHEMES:
        for wp in WPS:
            grids = yields[(scheme, wp)]
            for cut, stop in CUTS.items():
                signal_bins = grids[setup.signal][:, :stop].sum(axis=1)
                background_bins = sum(
                    grids[p][:, :stop] for p in setup.backgrounds
                ).sum(axis=1)
                signal = float(signal_bins.sum())
                background = float(background_bins.sum())
                ttbar = float(grids[setup.ttbar][:, :stop].sum())
                sb, z, za = _metrics(signal, background)
                valid = background_bins > 0
                z_shape = float(np.sqrt(np.sum(
                    signal_bins[valid] ** 2 / background_bins[valid]
                ))) if np.any(valid) else math.nan
                rows.append({
                    "scheme": scheme, "wp": wp, "cut": cut,
                    "signal": signal, "background": background, "ttbar": ttbar,
                    "s_over_b": sb, "s_over_sqrt_b": z,
                    "z_asimov": za, "z_shape": z_shape,
                })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_roc(rows: list[dict], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    for cut, marker, label in (("max1", "o", "$N_b \\leq 1$"),
                               ("max0", "s", "$N_b = 0$")):
        selected = [r for r in rows if r["cut"] == cut]
        x = [r["signal_efficiency"] for r in selected]
        y = [r["ttbar_rejection"] for r in selected]
        ax.plot(x, y, marker=marker, linewidth=2, label=label)
        for row in selected:
            ax.annotate(row["wp"], (row["signal_efficiency"], row["ttbar_rejection"]),
                        xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Higgs efficiency")
    ax.set_ylabel(r"$t\bar{t}$ rejection")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Blinded b-veto ROC, BDT score in [0.33, 1]")
    _save(fig, outdir / "roc")


def _plot_summary(rows: list[dict], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    for scheme in SCHEMES:
        fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
        for cut, marker, label in (("max1", "o", "$N_b \\leq 1$"),
                                   ("max0", "s", "$N_b = 0$")):
            selected = [r for r in rows if r["scheme"] == scheme and r["cut"] == cut]
            axes[0].plot(WPS, [r["s_over_sqrt_b"] for r in selected], marker=marker,
                         linewidth=2, label=label)
            axes[1].plot(WPS, [r["z_shape"] for r in selected], marker=marker,
                         linewidth=2, label=label)
        axes[0].set_ylabel(r"$S/\sqrt{B}$ (counting)")
        axes[1].set_ylabel(r"$\sqrt{\sum_i S_i^2/B_i}$")
        axes[1].set_xlabel("b-tag working point")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(f"Blinded total expected significance ({scheme} binning)")
        fig.tight_layout()
        _save(fig, outdir / f"significance_total_{scheme}")


def _plot_significance(rows: list[dict], cfg, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    for scheme in SCHEMES:
        edges = _score_edges(cfg, scheme)
        fig, axes = plt.subplots(len(edges) - 1, 1, figsize=(9, 3.3 * (len(edges) - 1)),
                                 sharex=True, squeeze=False)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            ax = axes[i, 0]
            for cut, marker, label in (("max1", "o", "$N_b \\leq 1$"),
                                       ("max0", "s", "$N_b = 0$")):
                selected = [r for r in rows if r["scheme"] == scheme and r["cut"] == cut
                            and r["bdt_low"] == lo and r["bdt_high"] == hi]
                ax.plot(WPS, [r["s_over_sqrt_b"] for r in selected], marker=marker,
                        linewidth=2, label=label)
            ax.set_ylabel(r"$S/\sqrt{B}$")
            ax.set_title(f"BDT in [{lo:g}, {hi:g})")
            ax.grid(alpha=0.25)
            if i == 0:
                ax.legend()
        axes[-1, 0].set_xlabel("b-tag working point")
        fig.suptitle(f"Blinded expected significance ({scheme} binning)", y=1.002)
        fig.tight_layout()
        _save(fig, outdir / f"significance_by_bdt_bin_{scheme}")


def _stairs(ax, edges, values, label, **kwargs) -> None:
    ax.stairs(values, edges, label=label, linewidth=2, **kwargs)


def _plot_score_shapes(cfg, yields, setup: StudySetup, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    for scheme in SCHEMES:
        edges = _score_edges(cfg, scheme)
        for wp in WPS:
            grids = yields[(scheme, wp)]
            fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
            components = (
                ("Higgs", grids[setup.signal]),
                ("All backgrounds", sum(grids[p] for p in setup.backgrounds)),
                (r"$t\bar{t}$ + single top", grids[setup.ttbar]),
            )
            for ax, (title, grid) in zip(axes, components):
                _stairs(ax, edges, grid.sum(axis=1), "no b veto", color="black")
                _stairs(ax, edges, grid[:, :2].sum(axis=1), r"$N_b \leq 1$",
                        color="tab:blue")
                _stairs(ax, edges, grid[:, :1].sum(axis=1), r"$N_b = 0$",
                        color="tab:orange")
                ax.set_ylabel("Expected events")
                ax.set_title(title)
                ax.grid(alpha=0.25)
                ax.legend()
            axes[-1].set_xlabel("BDT score")
            fig.suptitle(f"Blinded BDT shapes, WP {wp} ({scheme} binning)", y=0.995)
            fig.tight_layout()
            _save(fig, outdir / "bdt_shapes" / f"bdt_{scheme}_{wp}")


def _save(fig, stem: Path) -> None:
    import matplotlib.pyplot as plt

    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    setup = _validate(cfg, args.signal, args.ttbar)
    samples, warnings = discover_samples(cfg)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    present = {sample.process for sample in samples}
    for required in (setup.signal, setup.ttbar, setup.data):
        if required not in present:
            raise RuntimeError(f"no discovered samples for required process {required!r}")

    print("Blinding: ON; observed SR data will not be exported or plotted.")
    print(f"Preparing one skim/fill pass for {len(samples)} sample assignments...")
    skims = ensure_skims(cfg, samples, workers=args.workers)
    hists = fill_all(
        cfg, samples, skims, families=["datamc"], workers=args.workers,
        use_cache=False, write_cache=False, sidecars=False,
    )
    yields = _extract_blinded_yields(cfg, hists, setup)
    del hists  # drop all raw process histograms, including observed data

    rows = _build_rows(cfg, yields, setup)
    roc_rows = _build_roc_rows(yields, setup)
    summary_rows = _build_summary_rows(yields, setup)
    outdir = args.output.resolve() if args.output else cfg.resolved_output_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(outdir / "bveto_metrics.csv", rows)
    _write_csv(outdir / "bveto_roc.csv", roc_rows)
    _write_csv(outdir / "bveto_summary.csv", summary_rows)
    _plot_roc(roc_rows, outdir)
    _plot_summary(summary_rows, outdir)
    _plot_significance(rows, cfg, outdir)
    _plot_score_shapes(cfg, yields, setup, outdir)
    print(f"Done: blinded study outputs written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
