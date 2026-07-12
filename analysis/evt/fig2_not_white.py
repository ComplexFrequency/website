from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from config import CACHE_DIR, FIGURE_DIR, SEED
from noise import generate_raw_trace
from plot_style import (
    BASE_FONT_SIZE,
    FIGURE_WIDTH_IN,
    PALETTE,
    apply,
    figure_title,
    load_or_compute,
    panel_subtitle,
    save_figure,
    style_axes,
)

RAW_SAMPLES: int = 120_000
QQ_SAMPLES: int = 50_000
QQ_DISPLAY_POINTS: int = 2_500
HISTOGRAM_BINS: int = 60
CACHE_PATH = CACHE_DIR / "fig2_not_white_raw.npy"


def compute_raw() -> np.ndarray:
    rng = np.random.default_rng(SEED + 1)
    return generate_raw_trace(RAW_SAMPLES, rng)


def sample_autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = values.astype(np.float64, copy=False) - np.mean(values)
    variance_sum = float(np.dot(centered, centered))
    coefficients = np.empty(max_lag + 1, dtype=np.float64)
    coefficients[0] = 1.0
    for lag in range(1, max_lag + 1):
        coefficients[lag] = (
            float(np.dot(centered[:-lag], centered[lag:])) / variance_sum
        )
    return coefficients


def choose_acf_max_lag(acf_200: np.ndarray) -> int:
    tail = np.abs(acf_200[100:])
    if float(np.max(tail)) < 0.04:
        return 100
    return 200


def main():
    raw = load_or_compute(CACHE_PATH, compute_raw)
    sample_mean = float(np.mean(raw))
    sample_std = float(np.std(raw, ddof=1))
    acf_200 = sample_autocorrelation(raw, 200)
    acf_max_lag = choose_acf_max_lag(acf_200)
    acf = acf_200[: acf_max_lag + 1]

    qq_values = raw[:QQ_SAMPLES]
    standardized = (qq_values - np.mean(qq_values)) / np.std(qq_values, ddof=1)
    probabilities = (np.arange(1, QQ_DISPLAY_POINTS + 1) - 0.5) / QQ_DISPLAY_POINTS
    theoretical = stats.norm.ppf(probabilities)
    empirical = np.quantile(standardized, probabilities)

    apply()
    figure, axes_list = plt.subplots(
        1,
        3,
        figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.38),
    )
    axes_pdf, axes_qq, axes_acf = axes_list
    for axes in axes_list:
        style_axes(axes)

    bin_edges = np.linspace(
        sample_mean - 4.0 * sample_std,
        sample_mean + 4.0 * sample_std,
        HISTOGRAM_BINS + 1,
    )
    axes_pdf.hist(
        raw,
        bins=bin_edges,
        density=True,
        color=PALETTE.muted,
        alpha=0.45,
        edgecolor="none",
        zorder=2,
    )
    x_grid = np.linspace(bin_edges[0], bin_edges[-1], 400)
    normal_pdf = stats.norm.pdf(x_grid, loc=sample_mean, scale=sample_std)
    axes_pdf.plot(
        x_grid,
        normal_pdf,
        color=PALETTE.accent,
        linewidth=1.7,
        zorder=3,
        solid_capstyle="round",
    )
    axes_pdf.set_xlabel("Signal value")
    axes_pdf.set_ylabel("Density")
    axes_pdf.set_xlim(bin_edges[0], bin_edges[-1])
    axes_pdf.set_ylim(0.0, float(np.max(normal_pdf)) * 1.12)
    panel_subtitle(axes_pdf, "Histogram with Gaussian fit")

    axes_qq.plot(
        theoretical,
        empirical,
        color=PALETTE.ink,
        linewidth=0.0,
        marker="o",
        markersize=2.0,
        markerfacecolor=PALETTE.ink,
        markeredgewidth=0,
        alpha=0.85,
        zorder=2,
    )
    guide = np.array([-4.0, 4.0])
    axes_qq.plot(
        guide,
        guide,
        color=PALETTE.accent,
        linewidth=1.5,
        zorder=3,
        solid_capstyle="round",
    )
    axes_qq.set_xlabel("Expected under a normal curve")
    axes_qq.set_ylabel("Sorted samples")
    axes_qq.set_xlim(-4.1, 4.1)
    axes_qq.set_ylim(-4.1, 4.1)
    axes_qq.set_aspect("equal", adjustable="box")
    axes_qq.set_xticks([-4, -2, 0, 2, 4])
    axes_qq.set_yticks([-4, -2, 0, 2, 4])
    panel_subtitle(axes_qq, "Normal QQ plot")

    lag_axis = np.arange(acf_max_lag + 1)
    markerline, stemlines, baseline = axes_acf.stem(
        lag_axis,
        acf,
        linefmt="-",
        markerfmt=" ",
        basefmt=" ",
    )
    plt.setp(stemlines, color=PALETTE.ink, linewidth=0.85)
    plt.setp(markerline, visible=False)
    axes_acf.axhline(0.0, color=PALETTE.muted, linewidth=0.7, zorder=1)
    axes_acf.set_xlabel("Lag (samples)")
    axes_acf.set_ylabel("Correlation")
    axes_acf.set_xlim(-1.0, acf_max_lag + 1.0)
    axes_acf.set_ylim(-0.22, 1.08)
    if acf_max_lag <= 100:
        axes_acf.set_xticks([0, 25, 50, 75, 100])
        panel_subtitle(axes_acf, "Autocorrelation, lags 0-100")
    else:
        axes_acf.set_xticks([0, 50, 100, 150, 200])
        panel_subtitle(axes_acf, "Autocorrelation, lags 0-200")
    axes_acf.tick_params(labelsize=BASE_FONT_SIZE - 1)

    figure_title(figure, "Distribution and autocorrelation of the background noise")
    figure.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.16, wspace=0.34)
    svg_path, png_path = save_figure(figure, "fig2_not_white", FIGURE_DIR)
    print(svg_path)
    print(png_path)
    print("legend: removed (panel subtitles only)")


if __name__ == "__main__":
    main()
