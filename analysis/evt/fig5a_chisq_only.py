from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from cache_io import load_array
from config import CACHE_DIR, FIGURE_DIR, WINDOW_SAMPLES
from plot_style import (
    FIGURE_WIDTH_IN,
    PALETTE,
    apply,
    figure_legend,
    figure_title,
    save_figure,
    style_axes,
)

ENERGY_CACHE_PATH = CACHE_DIR / "calibration_energy_normalized.npy"


def naive_chi2_energy_median() -> float:
    degrees = WINDOW_SAMPLES - 1
    return float(stats.chi2.ppf(0.5, df=degrees) / WINDOW_SAMPLES)


def naive_chi2_energy_pdf(energy: np.ndarray) -> np.ndarray:
    degrees = WINDOW_SAMPLES - 1
    median_unit_energy = naive_chi2_energy_median()
    return stats.chi2.pdf(energy * WINDOW_SAMPLES * median_unit_energy, df=degrees) * (
        WINDOW_SAMPLES * median_unit_energy
    )


def main():
    energy = load_array(ENERGY_CACHE_PATH)

    bulk_low = float(np.quantile(energy, 0.0005))
    bulk_high = float(np.quantile(energy, 0.997))
    x_bulk = np.linspace(bulk_low, bulk_high, 500)

    apply()
    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.54))
    style_axes(axes)

    axes.hist(
        energy,
        bins=140,
        range=(bulk_low, bulk_high),
        density=True,
        color=PALETTE.muted,
        alpha=0.42,
        edgecolor="none",
        zorder=1,
    )
    axes.plot(
        x_bulk,
        naive_chi2_energy_pdf(x_bulk),
        color=PALETTE.danger,
        linewidth=1.55,
        linestyle=(0, (4, 2.5)),
        solid_capstyle="round",
        zorder=3,
    )
    axes.set_ylabel("Density")
    axes.set_xlabel("Window energy (loudness²)")
    axes.set_xlim(bulk_low, bulk_high)
    axes.set_ylim(bottom=0.0)

    legend_handles = [
        Patch(facecolor=PALETTE.muted, alpha=0.55, edgecolor="none"),
        Line2D(
            [0],
            [0],
            color=PALETTE.danger,
            linewidth=1.55,
            linestyle=(0, (4, 2.5)),
        ),
    ]
    legend_labels = [
        "Measured seconds (one week)",
        "Chi-squared model (white-noise assumption)",
    ]
    figure_title(figure, "Window energy with the white-noise model")
    figure_legend(
        figure,
        legend_handles,
        legend_labels,
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
    )
    figure.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.14)
    svg_path, png_path = save_figure(figure, "fig5a_chisq_only", FIGURE_DIR)
    print(svg_path)
    print(png_path)
    print("legend: figure-level horizontal under title")


if __name__ == "__main__":
    main()
