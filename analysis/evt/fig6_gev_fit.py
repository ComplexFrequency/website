from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import stats

from cache_io import load_array, save_array, save_json
from config import (
    CACHE_DIR,
    FIGURE_DIR,
    SEED,
    WINDOWS_PER_HOUR,
)
from features import hourly_maxima, normalize_loudness, quiet_night_level
from fitting import (
    GammaFit,
    GevFit,
    fit_gamma_moments,
    fit_gev,
    gev_pdf,
    gev_ppf,
    gev_sf,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_style import (
    BASE_FONT_SIZE,
    FIGURE_WIDTH_IN,
    PALETTE,
    apply,
    figure_legend,
    figure_title,
    save_figure,
    style_axes,
)

RAW_CACHE_PATH = CACHE_DIR / "calibration_loudness_raw.npy"
HOURLY_CACHE_PATH = CACHE_DIR / "calibration_hourly_maxima.npy"
GEV_JSON_PATH = CACHE_DIR / "gev_fit.json"
N_BOOTSTRAP: int = 1000
CI_LEVEL: float = 0.90


def load_normalized_loudness() -> np.ndarray:
    raw = load_array(RAW_CACHE_PATH)
    level = quiet_night_level(raw)
    return normalize_loudness(raw, level)


def load_hourly_maxima(normalized_loudness: np.ndarray) -> np.ndarray:
    try:
        return load_array(HOURLY_CACHE_PATH)
    except FileNotFoundError:
        pass
    maxima = hourly_maxima(normalized_loudness, WINDOWS_PER_HOUR)
    save_array(HOURLY_CACHE_PATH, maxima)
    return maxima


def gamma_hourly_max_loudness_pdf(
    loudness: np.ndarray,
    energy_fit: GammaFit,
    n_windows: int,
) -> np.ndarray:
    energy = np.square(loudness)
    gamma_cdf = stats.gamma.cdf(energy, a=energy_fit.shape, scale=energy_fit.scale)
    gamma_pdf = stats.gamma.pdf(energy, a=energy_fit.shape, scale=energy_fit.scale)
    max_energy_pdf = n_windows * np.power(gamma_cdf, n_windows - 1) * gamma_pdf
    return max_energy_pdf * 2.0 * loudness


def gamma_hourly_max_loudness_ppf(
    probability: float,
    energy_fit: GammaFit,
    n_windows: int,
) -> float:
    per_window_cdf = probability ** (1.0 / n_windows)
    energy = stats.gamma.ppf(per_window_cdf, a=energy_fit.shape, scale=energy_fit.scale)
    return float(np.sqrt(energy))


def gamma_hourly_max_loudness_sf(
    loudness: np.ndarray | float,
    energy_fit: GammaFit,
    n_windows: int,
) -> np.ndarray | float:
    energy = np.square(loudness)
    gamma_cdf = stats.gamma.cdf(energy, a=energy_fit.shape, scale=energy_fit.scale)
    return 1.0 - np.power(gamma_cdf, n_windows)


def bootstrap_gev_intervals(
    maxima: np.ndarray,
    n_bootstrap: int,
    seed: int,
    ci_level: float,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = maxima.size
    xi_samples = np.empty(n_bootstrap, dtype=np.float64)
    mu_samples = np.empty(n_bootstrap, dtype=np.float64)
    sigma_samples = np.empty(n_bootstrap, dtype=np.float64)
    kept = 0
    attempts = 0
    max_attempts = n_bootstrap * 5
    while kept < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sample = rng.choice(maxima, size=n, replace=True)
        try:
            fit = fit_gev(sample)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        if not np.isfinite(fit.xi) or not np.isfinite(fit.loc) or not np.isfinite(fit.scale):
            continue
        if fit.scale <= 0.0:
            continue
        xi_samples[kept] = fit.xi
        mu_samples[kept] = fit.loc
        sigma_samples[kept] = fit.scale
        kept += 1
    if kept < n_bootstrap:
        raise RuntimeError(f"bootstrap only kept {kept} of {n_bootstrap} fits")
    alpha = 0.5 * (1.0 - ci_level)
    lower_q = alpha
    upper_q = 1.0 - alpha
    return {
        "xi": (
            float(np.quantile(xi_samples, lower_q)),
            float(np.quantile(xi_samples, upper_q)),
        ),
        "mu": (
            float(np.quantile(mu_samples, lower_q)),
            float(np.quantile(mu_samples, upper_q)),
        ),
        "sigma": (
            float(np.quantile(sigma_samples, lower_q)),
            float(np.quantile(sigma_samples, upper_q)),
        ),
    }


def add_qq_inset(
    axes: Axes,
    bbox: list[float],
    sorted_maxima: np.ndarray,
    theoretical: np.ndarray,
    title: str,
    xlabel: str,
    y_limits: tuple[float, float],
):
    inset = axes.inset_axes(bbox)
    style_axes(inset)
    inset.scatter(
        theoretical,
        sorted_maxima,
        s=12,
        color=PALETTE.ink,
        alpha=0.85,
        linewidths=0,
        zorder=3,
    )
    guide_low, guide_high = y_limits
    guide = np.array([guide_low, guide_high])
    inset.plot(
        guide,
        guide,
        color=PALETTE.accent,
        linewidth=1.3,
        zorder=2,
        solid_capstyle="round",
    )
    inset.set_title(
        title,
        loc="left",
        color=PALETTE.ink,
        fontsize=BASE_FONT_SIZE - 2.0,
        pad=3,
    )
    inset.set_xlabel(xlabel, fontsize=BASE_FONT_SIZE - 2.5)
    inset.set_ylabel("Hourly max", fontsize=BASE_FONT_SIZE - 2.5)
    inset.tick_params(labelsize=BASE_FONT_SIZE - 3.0, width=0.7, length=2.5)
    inset.spines["left"].set_linewidth(0.7)
    inset.spines["bottom"].set_linewidth(0.7)
    inset.set_xlim(guide_low, guide_high)
    inset.set_ylim(guide_low, guide_high)
    inset.set_aspect("equal", adjustable="box")


def add_gev_and_gamma_qq_insets(
    axes: Axes,
    maxima: np.ndarray,
    gev_fit: GevFit,
    gamma_fit: GammaFit,
    n_windows: int,
):
    sorted_maxima = np.sort(maxima)
    n = sorted_maxima.size
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    gev_theoretical = np.array([gev_ppf(float(p), gev_fit) for p in probabilities])
    gamma_theoretical = np.array(
        [gamma_hourly_max_loudness_ppf(float(p), gamma_fit, n_windows) for p in probabilities]
    )
    guide_low = float(
        min(gev_theoretical[0], gamma_theoretical[0], sorted_maxima[0])
    )
    guide_high = float(
        max(gev_theoretical[-1], gamma_theoretical[-1], sorted_maxima[-1])
    )
    pad = 0.02 * (guide_high - guide_low)
    y_limits = (guide_low - pad, guide_high + pad)
    add_qq_inset(
        axes,
        [0.62, 0.68, 0.34, 0.26],
        sorted_maxima,
        gev_theoretical,
        "GEV QQ plot",
        "GEV quantile",
        y_limits,
    )
    add_qq_inset(
        axes,
        [0.62, 0.16, 0.34, 0.26],
        sorted_maxima,
        gamma_theoretical,
        "Gamma QQ plot",
        "Gamma quantile",
        y_limits,
    )


def write_gev_json(
    path: Path,
    fit: GevFit,
    intervals: dict[str, tuple[float, float]],
    n_maxima: int,
):
    payload = {
        "mu": fit.loc,
        "sigma": fit.scale,
        "xi": fit.xi,
        "scipy_c": fit.scipy_c,
        "mu_ci90": list(intervals["mu"]),
        "sigma_ci90": list(intervals["sigma"]),
        "xi_ci90": list(intervals["xi"]),
        "n_maxima": n_maxima,
        "n_bootstrap": N_BOOTSTRAP,
        "ci_level": CI_LEVEL,
        "convention": "xi = -scipy.stats.genextreme.c",
    }
    save_json(path, payload)


def main():
    normalized = load_normalized_loudness()
    maxima = load_hourly_maxima(normalized)
    energy = np.square(normalized)
    energy_gamma = fit_gamma_moments(energy)
    gev = fit_gev(maxima)
    intervals = bootstrap_gev_intervals(
        maxima,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED + 6,
        ci_level=CI_LEVEL,
    )
    write_gev_json(GEV_JSON_PATH, gev, intervals, maxima.size)

    x_min = float(np.min(maxima) - 0.02)
    x_max = float(np.max(maxima) + 0.04)
    x_grid = np.linspace(max(x_min, 1e-6), x_max, 500)
    gev_density = np.asarray(gev_pdf(x_grid, gev), dtype=np.float64)
    gamma_max_density = gamma_hourly_max_loudness_pdf(
        x_grid,
        energy_gamma,
        WINDOWS_PER_HOUR,
    )

    apply()
    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.54))
    style_axes(axes)

    axes.hist(
        maxima,
        bins=18,
        density=True,
        color=PALETTE.muted,
        alpha=0.42,
        edgecolor=PALETTE.muted,
        linewidth=0.4,
        zorder=1,
    )
    axes.plot(
        x_grid,
        gev_density,
        color=PALETTE.accent,
        linewidth=1.9,
        solid_capstyle="round",
        zorder=3,
    )
    axes.plot(
        x_grid,
        gamma_max_density,
        color=PALETTE.danger,
        linewidth=1.55,
        linestyle=(0, (4, 2.5)),
        solid_capstyle="round",
        zorder=3,
    )
    add_gev_and_gamma_qq_insets(axes, maxima, gev, energy_gamma, WINDOWS_PER_HOUR)

    axes.set_xlabel("Hourly-max loudness (versus quiet night)")
    axes.set_ylabel("Density")
    axes.set_xlim(x_min, x_max)
    y_top = max(float(np.max(gev_density)), float(np.max(gamma_max_density))) * 1.18
    axes.set_ylim(0.0, y_top)

    figure_title(figure, "Hourly maximum loudness, one week, with fitted models")
    figure_legend(
        figure,
        [
            Patch(facecolor=PALETTE.muted, alpha=0.55, edgecolor="none"),
            Line2D([0], [0], color=PALETTE.accent, linewidth=1.9),
            Line2D(
                [0],
                [0],
                color=PALETTE.danger,
                linewidth=1.55,
                linestyle=(0, (4, 2.5)),
            ),
        ],
        [
            "Hourly maxima (one week)",
            "Fitted GEV",
            "Gamma-model prediction",
        ],
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
    )
    figure.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.14)
    svg_path, png_path = save_figure(figure, "fig6_gev_fit", FIGURE_DIR)
    print(svg_path)
    print(png_path)
    print(f"n_maxima={maxima.size}")
    print("legend: figure-level under title")

    gev_loglik = float(np.sum(np.log(np.asarray(gev_pdf(maxima, gev), dtype=np.float64))))
    gamma_loglik = float(
        np.sum(np.log(gamma_hourly_max_loudness_pdf(maxima, energy_gamma, WINDOWS_PER_HOUR)))
    )
    print(f"gev_loglik={gev_loglik:.4f}")
    print(f"gamma_loglik={gamma_loglik:.4f}")
    for level in (1.15, 1.16):
        gev_tail = float(gev_sf(level, gev))
        gamma_tail = float(
            gamma_hourly_max_loudness_sf(level, energy_gamma, WINDOWS_PER_HOUR)
        )
        print(f"P(hourly_max>{level})  gev={gev_tail:.6f}  gamma={gamma_tail:.6f}")


if __name__ == "__main__":
    main()
