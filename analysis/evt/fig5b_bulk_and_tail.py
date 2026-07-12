from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from cache_io import load_array, save_array
from config import (
    CACHE_DIR,
    CALIBRATION_WINDOWS,
    FIGURE_DIR,
    SEED,
    WINDOW_SAMPLES,
)
from features import normalize_loudness, quiet_night_level
from fitting import fit_gamma_moments, gamma_survival
from noise import generate_loudness
from plot_style import (
    FIGURE_WIDTH_IN,
    PALETTE,
    apply,
    figure_legend,
    figure_title,
    panel_subtitle,
    save_figure,
    style_axes,
)

RAW_CACHE_PATH = CACHE_DIR / "calibration_loudness_raw.npy"
ENERGY_CACHE_PATH = CACHE_DIR / "calibration_energy_normalized.npy"

SURVIVAL_TOP: float = 1e-1
SURVIVAL_BOTTOM: float = 1.0 / 604800.0 * 0.6
HUMAN_TICKS: tuple[tuple[float, str], ...] = (
    (1.0 / 60.0, "Once a minute"),
    (1.0 / 3600.0, "Once an hour"),
    (1.0 / 86400.0, "Once a day"),
    (1.0 / 604800.0, "Once a week"),
)


def load_or_generate_raw_loudness() -> tuple[np.ndarray, float | None]:
    try:
        return load_array(RAW_CACHE_PATH), None
    except FileNotFoundError:
        pass
    print(f"generating calibration week: {CALIBRATION_WINDOWS} windows...")
    started = time.perf_counter()
    rng = np.random.default_rng(SEED + 5)
    loudness = generate_loudness(CALIBRATION_WINDOWS, rng)
    save_array(RAW_CACHE_PATH, loudness)
    wall_time = time.perf_counter() - started
    print(f"calibration wall_time_s={wall_time:.2f}")
    return loudness, wall_time


def load_normalized_energy(raw_loudness: np.ndarray) -> np.ndarray:
    try:
        return load_array(ENERGY_CACHE_PATH)
    except FileNotFoundError:
        pass
    level = quiet_night_level(raw_loudness)
    normalized = normalize_loudness(raw_loudness, level)
    energy = np.square(normalized)
    save_array(ENERGY_CACHE_PATH, energy)
    return energy


def naive_chi2_energy_median() -> float:
    degrees = WINDOW_SAMPLES - 1
    return float(stats.chi2.ppf(0.5, df=degrees) / WINDOW_SAMPLES)


def naive_chi2_energy_pdf(energy: np.ndarray) -> np.ndarray:
    degrees = WINDOW_SAMPLES - 1
    median_unit_energy = naive_chi2_energy_median()
    return stats.chi2.pdf(energy * WINDOW_SAMPLES * median_unit_energy, df=degrees) * (
        WINDOW_SAMPLES * median_unit_energy
    )


def naive_chi2_energy_sf(energy: np.ndarray) -> np.ndarray:
    degrees = WINDOW_SAMPLES - 1
    median_unit_energy = naive_chi2_energy_median()
    return stats.chi2.sf(energy * WINDOW_SAMPLES * median_unit_energy, df=degrees)


def main():
    raw_loudness, wall_time = load_or_generate_raw_loudness()
    if wall_time is None:
        print(f"calibration cache hit: {RAW_CACHE_PATH}")
    energy = load_normalized_energy(raw_loudness)
    gamma_fit = fit_gamma_moments(energy)

    sorted_energy = np.sort(energy)
    n = sorted_energy.size
    empirical_sf = (n - np.arange(n)) / n

    bulk_low = float(np.quantile(energy, 0.0005))
    bulk_high = float(np.quantile(energy, 0.997))
    tail_low = 1.02
    tail_high = float(sorted_energy[-1] * 1.03)

    x_bulk = np.linspace(bulk_low, bulk_high, 500)
    x_tail = np.linspace(tail_low, tail_high, 800)

    apply()
    figure, (axes_bulk, axes_tail) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.54),
        sharex=False,
    )
    style_axes(axes_bulk)
    style_axes(axes_tail)

    axes_bulk.hist(
        energy,
        bins=140,
        range=(bulk_low, bulk_high),
        density=True,
        color=PALETTE.muted,
        alpha=0.42,
        edgecolor="none",
        zorder=1,
    )
    axes_bulk.plot(
        x_bulk,
        stats.gamma.pdf(x_bulk, a=gamma_fit.shape, scale=gamma_fit.scale),
        color=PALETTE.accent,
        linewidth=1.9,
        solid_capstyle="round",
        zorder=3,
    )
    axes_bulk.plot(
        x_bulk,
        naive_chi2_energy_pdf(x_bulk),
        color=PALETTE.danger,
        linewidth=1.55,
        linestyle=(0, (4, 2.5)),
        solid_capstyle="round",
        zorder=3,
    )
    axes_bulk.set_ylabel("Density")
    axes_bulk.set_xlabel("Window energy (loudness²)")
    axes_bulk.set_xlim(bulk_low, bulk_high)
    axes_bulk.set_ylim(bottom=0.0)
    panel_subtitle(axes_bulk, "Full distribution (density)")

    tail_mask = (sorted_energy >= tail_low) & (empirical_sf >= SURVIVAL_BOTTOM * 0.5)
    axes_tail.scatter(
        sorted_energy[tail_mask],
        empirical_sf[tail_mask],
        s=9,
        color=PALETTE.ink,
        alpha=0.5,
        linewidths=0,
        zorder=3,
    )
    axes_tail.plot(
        x_tail,
        np.asarray(gamma_survival(x_tail, gamma_fit), dtype=np.float64),
        color=PALETTE.accent,
        linewidth=1.9,
        solid_capstyle="round",
        zorder=4,
    )
    axes_tail.plot(
        x_tail,
        naive_chi2_energy_sf(x_tail),
        color=PALETTE.danger,
        linewidth=1.55,
        linestyle=(0, (4, 2.5)),
        solid_capstyle="round",
        zorder=4,
    )
    axes_tail.set_yscale("log")
    axes_tail.set_ylabel("Chance of being this loud or louder")
    axes_tail.set_xlabel("Window energy (loudness²)")
    axes_tail.set_xlim(tail_low, tail_high)
    axes_tail.set_ylim(SURVIVAL_BOTTOM, SURVIVAL_TOP)
    panel_subtitle(axes_tail, "Right tail of the same data (survival, log scale)")

    twin = axes_tail.twinx()
    twin.set_yscale("log")
    twin.set_ylim(SURVIVAL_BOTTOM, SURVIVAL_TOP)
    twin_ticks = [value for value, _ in HUMAN_TICKS]
    twin_labels = [label for _, label in HUMAN_TICKS]
    twin.set_yticks(twin_ticks)
    twin.set_yticklabels(twin_labels)
    twin.tick_params(
        colors=PALETTE.muted,
        labelsize=8.0,
        length=3.0,
        width=0.7,
        pad=2,
    )
    twin.spines["top"].set_visible(False)
    twin.spines["left"].set_visible(False)
    twin.spines["bottom"].set_visible(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color(PALETTE.ink)
    twin.spines["right"].set_linewidth(0.8)
    twin.set_ylabel("")

    legend_handles = [
        Patch(facecolor=PALETTE.muted, alpha=0.55, edgecolor="none"),
        Line2D([0], [0], color=PALETTE.accent, linewidth=1.9),
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
        "Gamma model (correlation-corrected)",
        "Chi-squared model (white-noise assumption)",
    ]
    figure_title(figure, "Window energy: distribution and upper tail")
    figure_legend(
        figure,
        legend_handles,
        legend_labels,
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
    )
    figure.subplots_adjust(left=0.08, right=0.88, top=0.78, bottom=0.14, wspace=0.42)
    svg_path, png_path = save_figure(figure, "fig5b_bulk_and_tail", FIGURE_DIR)
    print(svg_path)
    print(png_path)
    print("legend: figure-level horizontal under title")


if __name__ == "__main__":
    main()
