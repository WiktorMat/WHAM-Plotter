"""Data/MC ratio plots: MC+QCD stack, stat band, data points, ratio panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from uncertainties.propagation import normalization_variance
from wham.config import AnalysisConfig
from wham.jetfakes import FF_PASS, active_estimate
from wham.render.ffclosure import closure_metrics, metric_lines
from wham.render.common import (
    cms_label,
    draw_unc_band,
    draw_unroll_guides,
    save,
    slice_1d,
    stack_components,
    total_mc,
    var_label,
)


def signal_region(qcd_method: str) -> str:
    return "OS" if qcd_method == "ss" else "OS_iso"


def render_datamc(
    cfg: AnalysisConfig, hists: dict, outdir: Path, only_vars=None, console=None
) -> None:
    region = FF_PASS if active_estimate(cfg) else signal_region(cfg.qcd.method)
    data_proc = cfg.data_process()

    for (family, var), h in sorted(hists.items()):
        if family != "datamc" or (only_vars and var not in only_vars):
            continue
        if data_proc is None or region not in list(h.axes["region"]):
            continue

        edges = h.axes[-1].edges
        centers = 0.5 * (edges[:-1] + edges[1:])

        data_view = slice_1d(h, data_proc, region).view()
        data = data_view["value"]
        data_unc = np.sqrt(np.maximum(data_view["variance"], 0.0))

        mc, mc_sumw2 = total_mc(h, cfg, region)
        norm_down2, norm_up2 = normalization_variance(h, cfg, region)
        mc_unc_down = np.sqrt(np.maximum(mc_sumw2, 0.0) + norm_down2)
        mc_unc_up = np.sqrt(np.maximum(mc_sumw2, 0.0) + norm_up2)

        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax = fig.add_subplot(gs[0])
        rax = fig.add_subplot(gs[1], sharex=ax)

        stack, labels, colors = stack_components(cfg, h, region)
        if stack:
            hep.histplot(stack, stack=True, histtype="fill", color=colors,
                         label=labels, edgecolor="black", linewidth=0.5, ax=ax)

        draw_unc_band(ax, edges, np.maximum(mc - mc_unc_down, 0.0), mc + mc_unc_up,
                      label="Stat. $\\oplus$ norm. unc.")
        ax.errorbar(centers, data, yerr=data_unc, fmt="o", color="black",
                    markersize=5, label="Data", zorder=5)

        ax.set_xlabel("")  # mplhep copies the hist axis name; the ratio panel owns it
        ax.set_ylabel("Events")
        vcfg = cfg.variables.get(var)
        if vcfg is not None and vcfg.logy:
            # log axis: floor just under one event, top with decades of headroom
            # so the legend clears a peak that outruns the tails by orders of mag.
            ax.set_yscale("log")
            ymax = float(max(mc.max(initial=0.0), data.max(initial=0.0)))
            ax.set_ylim(0.5, ymax * 50 if ymax > 0 else 10.0)
        else:
            ax.set_ylim(bottom=0)
        # legend reads top-of-stack first: Data, then MC top to bottom, band last
        handles, names = ax.get_legend_handles_labels()
        by_label = dict(zip(names, handles))
        order = ["Data", *reversed(labels), "Stat. $\\oplus$ norm. unc."]
        ax.legend([by_label[n] for n in order if n in by_label],
                  [n for n in order if n in by_label], fontsize=17)
        ax.tick_params(labelbottom=False)
        cms_label(ax, cfg)
        draw_unroll_guides(ax, cfg, var)
        if cfg.plots.datamc_metrics:
            metrics = closure_metrics(data, data_view["variance"], mc, mc_sumw2, edges)
            lines = metric_lines(metrics, cfg.plots.datamc_metrics)
            if lines:
                ax.text(
                    0.03,
                    0.95,
                    "\n".join(lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=13,
                    bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.85},
                )
            if console is not None:
                norm = metrics["norm_delta"]
                chi2 = metrics["shape_chi2_ndf"]
                max_abs_z = metrics["max_abs_z"]
                console.print(
                    f"  datamc/{var}: data={metrics['target_sum']:.6g}, "
                    f"mc={metrics['prediction_sum']:.6g}, "
                    f"norm={'N/A' if norm is None else f'{100.0 * float(norm):+.2f}%'}, "
                    f"shape_chi2/ndf={'N/A' if chi2 is None else f'{float(chi2):.3g}'}, "
                    f"max|z|={'N/A' if max_abs_z is None else f'{float(max_abs_z):.3g}'}, "
                    f"skipped_bins={metrics['skipped_shape_bins']}"
                )

        # ---- ratio panel
        safe_mc = np.where(mc > 0, mc, np.nan)
        ratio = data / safe_mc
        ratio_unc = data_unc / safe_mc
        rel_mc_down = np.where(mc > 0, mc_unc_down / safe_mc, 0.0)
        rel_mc_up = np.where(mc > 0, mc_unc_up / safe_mc, 0.0)

        rax.axhline(1.0, linestyle="--", color="black", linewidth=1)
        draw_unc_band(rax, edges, 1.0 - rel_mc_down, 1.0 + rel_mc_up)
        rax.errorbar(centers, ratio, yerr=ratio_unc, fmt="o", color="black", markersize=5)
        rax.set_ylim(0.5, 1.5)
        rax.set_ylabel("Data / MC")
        rax.set_xlabel(var_label(cfg, var))

        save(fig, outdir / "datamc", var, console)
