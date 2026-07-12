from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from cache_io import ConfigFingerprintMismatch, load_array, load_json
from config import (
    CACHE_DIR,
    FALSE_ALARM_HORIZON_HOURS,
    FIGURE_DIR,
    HOURS_PER_YEAR,
    SEED,
    WINDOWS_PER_HOUR,
)
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
    FIGURE_WIDTH_IN,
    PALETTE,
    apply,
    figure_legend,
    figure_title,
    panel_subtitle,
    save_figure,
    style_axes,
)

RAW_CALIBRATION_PATH = CACHE_DIR / "calibration_loudness_raw.npy"
ENERGY_CACHE_PATH = CACHE_DIR / "calibration_energy_normalized.npy"
CAL_HOURLY_PATH = CACHE_DIR / "calibration_hourly_maxima.npy"
GEV_JSON_PATH = CACHE_DIR / "gev_fit.json"
VAL_HOURLY_PATH = CACHE_DIR / "validation_hourly_maxima.npy"
VAL_META_PATH = CACHE_DIR / "validation_meta.json"
SMOKE_HOURLY_PATH = CACHE_DIR / "smoke" / "validation_hourly_maxima.npy"
SMOKE_META_PATH = CACHE_DIR / "smoke" / "validation_meta.json"
TABLE_PATH = Path(__file__).resolve().parent / "validation_table.md"
N_BOOTSTRAP: int = 1000
PER_HOUR_RATES: tuple[float, ...] = (1e-1, 1e-2, 1e-3)


def load_gev(path: Path) -> GevFit:
    payload = load_json(path)
    xi = float(payload["xi"])
    return GevFit(
        xi=xi,
        loc=float(payload["mu"]),
        scale=float(payload["sigma"]),
        scipy_c=float(payload.get("scipy_c", -xi)),
    )


def load_energy_gamma() -> GammaFit:
    try:
        energy = load_array(ENERGY_CACHE_PATH)
    except (FileNotFoundError, ConfigFingerprintMismatch):
        raw = load_array(RAW_CALIBRATION_PATH)
        quiet = float(np.median(raw))
        energy = np.square(raw / quiet)
    return fit_gamma_moments(energy)


def gamma_hourly_max_survival(
    loudness: np.ndarray,
    energy_fit: GammaFit,
    n_windows: int = WINDOWS_PER_HOUR,
) -> np.ndarray:
    energy = np.square(loudness)
    cdf = stats.gamma.cdf(energy, a=energy_fit.shape, scale=energy_fit.scale)
    return 1.0 - np.power(cdf, n_windows)


def gamma_hourly_max_pdf(
    loudness: np.ndarray,
    energy_fit: GammaFit,
    n_windows: int = WINDOWS_PER_HOUR,
) -> np.ndarray:
    energy = np.square(loudness)
    cdf = stats.gamma.cdf(energy, a=energy_fit.shape, scale=energy_fit.scale)
    pdf = stats.gamma.pdf(energy, a=energy_fit.shape, scale=energy_fit.scale)
    max_energy_pdf = n_windows * np.power(cdf, n_windows - 1) * pdf
    return max_energy_pdf * 2.0 * loudness


def gamma_annual_max_pdf(
    loudness: np.ndarray,
    energy_fit: GammaFit,
    n_hours: int,
    n_windows: int = WINDOWS_PER_HOUR,
) -> np.ndarray:
    hourly_cdf = 1.0 - gamma_hourly_max_survival(loudness, energy_fit, n_windows)
    hourly_pdf = gamma_hourly_max_pdf(loudness, energy_fit, n_windows)
    return n_hours * np.power(hourly_cdf, n_hours - 1) * hourly_pdf


def gev_annual_max_pdf(loudness: np.ndarray, fit: GevFit, n_hours: int) -> np.ndarray:
    cdf = np.asarray(
        stats.genextreme.cdf(loudness, c=-fit.xi, loc=fit.loc, scale=fit.scale),
        dtype=np.float64,
    )
    pdf = np.asarray(gev_pdf(loudness, fit), dtype=np.float64)
    return n_hours * np.power(np.clip(cdf, 0.0, 1.0), n_hours - 1) * pdf


def bootstrap_gev_survival_band(
    calibration_maxima: np.ndarray,
    grid: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = calibration_maxima.size
    curves = np.empty((n_bootstrap, grid.size), dtype=np.float64)
    kept = 0
    attempts = 0
    while kept < n_bootstrap and attempts < n_bootstrap * 5:
        attempts += 1
        sample = rng.choice(calibration_maxima, size=n, replace=True)
        try:
            fit = fit_gev(sample)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        if fit.scale <= 0.0 or not np.isfinite(fit.xi):
            continue
        curves[kept] = np.asarray(gev_sf(grid, fit), dtype=np.float64)
        kept += 1
    if kept < n_bootstrap:
        raise RuntimeError(f"bootstrap kept only {kept}")
    return np.quantile(curves, 0.05, axis=0), np.quantile(curves, 0.95, axis=0)


def design_validation_thresholds(fit: GevFit) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for rate in PER_HOUR_RATES:
        key = f"p_hour_{rate:.0e}"
        thresholds[key] = gev_ppf(1.0 - rate, fit)
    u95_cdf = 0.95 ** (1.0 / FALSE_ALARM_HORIZON_HOURS)
    thresholds["u95"] = gev_ppf(u95_cdf, fit)
    return thresholds


def bootstrap_expected_hourly_false_alarms(
    calibration_maxima: np.ndarray,
    thresholds: dict[str, float],
    n_hours: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    n = calibration_maxima.size
    keys = list(thresholds.keys())
    values = np.array([thresholds[k] for k in keys], dtype=np.float64)
    samples = np.empty((n_bootstrap, values.size), dtype=np.float64)
    kept = 0
    attempts = 0
    while kept < n_bootstrap and attempts < n_bootstrap * 5:
        attempts += 1
        sample = rng.choice(calibration_maxima, size=n, replace=True)
        try:
            fit = fit_gev(sample)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        if fit.scale <= 0.0 or not np.isfinite(fit.xi):
            continue
        survival = np.asarray(gev_sf(values, fit), dtype=np.float64)
        samples[kept] = n_hours * survival
        kept += 1
    if kept < n_bootstrap:
        raise RuntimeError(f"bootstrap kept only {kept}")
    result: dict[str, tuple[float, float, float]] = {}
    for index, key in enumerate(keys):
        column = samples[:, index]
        result[key] = (
            float(np.mean(column)),
            float(np.quantile(column, 0.05)),
            float(np.quantile(column, 0.95)),
        )
    return result


def gamma_expected_hourly_false_alarms(
    thresholds: dict[str, float],
    energy_fit: GammaFit,
    n_hours: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, threshold in thresholds.items():
        survival = float(
            gamma_hourly_max_survival(np.array([threshold]), energy_fit)[0]
        )
        result[key] = n_hours * survival
    return result


def realized_hourly_counts(
    hourly_maxima: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, int]:
    return {
        key: int(np.sum(hourly_maxima > threshold))
        for key, threshold in thresholds.items()
    }


def write_validation_table(
    path: Path,
    thresholds: dict[str, float],
    gamma_hourly: dict[str, float],
    gev_hourly: dict[str, tuple[float, float, float]],
    realized_hourly: dict[str, int],
    n_hours: int,
):
    lines = [
        "# One-year false-alarm validation",
        "",
        f"Simulated hours: **{n_hours}**. "
        "Rows invert the week-fitted GEV at per-hour exceedance rates "
        r"$10^{-1}$, $10^{-2}$, $10^{-3}$, plus the decade-safe $u_{95}$ threshold. "
        "Counts are hours whose maximum loudness exceeds the threshold.",
        "",
        "| row | threshold (loudness) | gamma predicted | GEV predicted (90% bootstrap) | realized |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "p_hour_1e-01": r"$p_{\mathrm{hour}}=10^{-1}$",
        "p_hour_1e-02": r"$p_{\mathrm{hour}}=10^{-2}$",
        "p_hour_1e-03": r"$p_{\mathrm{hour}}=10^{-3}$",
        "u95": r"$u_{95}$ (10-year 95%)",
    }
    for key, threshold in thresholds.items():
        g_pred = gamma_hourly[key]
        mean, lo, hi = gev_hourly[key]
        realized = realized_hourly[key]
        lines.append(
            f"| {labels.get(key, key)} | {threshold:.4f} | "
            f"{g_pred:.3e} | {mean:.3e} [{lo:.3e}, {hi:.3e}] | {realized} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def resolve_validation_paths() -> tuple[Path, Path, bool]:
    if VAL_HOURLY_PATH.exists():
        return VAL_HOURLY_PATH, VAL_META_PATH, False
    if SMOKE_HOURLY_PATH.exists():
        return SMOKE_HOURLY_PATH, SMOKE_META_PATH, True
    raise FileNotFoundError(
        "no validation maxima found; run validate_one_year.py smoke or full year"
    )


def main():
    hourly_path, meta_path, is_smoke = resolve_validation_paths()
    validation_maxima = np.load(hourly_path)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        n_hours = int(meta.get("n_hours", validation_maxima.size))
    else:
        n_hours = int(validation_maxima.size)
        meta = {}

    central = load_gev(GEV_JSON_PATH)
    energy_gamma = load_energy_gamma()
    calibration_maxima = load_array(CAL_HOURLY_PATH)
    thresholds = design_validation_thresholds(central)
    annual_max = float(np.max(validation_maxima))

    sorted_maxima = np.sort(validation_maxima)
    empirical_sf_vals = (np.arange(sorted_maxima.size, 0, -1)) / sorted_maxima.size

    x_low = float(np.quantile(validation_maxima, 0.01))
    x_high = float(max(np.max(validation_maxima) * 1.03, annual_max * 1.02))
    grid = np.linspace(x_low, x_high, 400)
    gev_survival = np.asarray(gev_sf(grid, central), dtype=np.float64)
    gamma_survival = gamma_hourly_max_survival(grid, energy_gamma)
    band_lo, band_hi = bootstrap_gev_survival_band(
        calibration_maxima,
        grid,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED + 13,
    )

    annual_grid = np.linspace(
        float(np.min(validation_maxima)),
        float(max(annual_max * 1.08, np.quantile(validation_maxima, 0.999) * 1.05)),
        500,
    )
    gev_annual = gev_annual_max_pdf(annual_grid, central, n_hours)
    gamma_annual = gamma_annual_max_pdf(annual_grid, energy_gamma, n_hours)

    if not is_smoke:
        apply()
        figure, (axes_surv, axes_ann) = plt.subplots(
            1,
            2,
            figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.54),
        )
        style_axes(axes_surv)
        style_axes(axes_ann)

        axes_surv.fill_between(
            grid,
            band_lo,
            band_hi,
            color=PALETTE.accent,
            alpha=0.22,
            linewidth=0,
            zorder=1,
        )
        axes_surv.scatter(
            sorted_maxima,
            empirical_sf_vals,
            s=8,
            color=PALETTE.ink,
            alpha=0.45,
            linewidths=0,
            zorder=3,
        )
        axes_surv.plot(
            grid,
            gev_survival,
            color=PALETTE.accent,
            linewidth=1.8,
            solid_capstyle="round",
            zorder=4,
        )
        axes_surv.plot(
            grid,
            gamma_survival,
            color=PALETTE.danger,
            linewidth=1.5,
            linestyle=(0, (4, 2.5)),
            solid_capstyle="round",
            zorder=4,
        )
        axes_surv.set_yscale("log")
        axes_surv.set_xlabel("Hourly-max loudness (versus quiet night)")
        axes_surv.set_ylabel("Chance of being this loud or louder")
        axes_surv.set_xlim(x_low, x_high)
        axes_surv.set_ylim(1e-4, 1.05)
        panel_subtitle(axes_surv, "Hourly-maximum survival")

        axes_ann.plot(
            annual_grid,
            gev_annual,
            color=PALETTE.accent,
            linewidth=1.8,
            solid_capstyle="round",
            zorder=3,
        )
        axes_ann.plot(
            annual_grid,
            gamma_annual,
            color=PALETTE.danger,
            linewidth=1.5,
            linestyle=(0, (4, 2.5)),
            solid_capstyle="round",
            zorder=3,
        )
        axes_ann.axvline(
            annual_max,
            color=PALETTE.ink,
            linewidth=1.5,
            solid_capstyle="round",
            zorder=4,
        )
        axes_ann.set_xlabel("Annual-max loudness (versus quiet night)")
        axes_ann.set_ylabel("Density")
        axes_ann.set_xlim(annual_grid[0], annual_grid[-1])
        axes_ann.set_ylim(bottom=0.0)
        panel_subtitle(axes_ann, "Annual maximum")

        figure_title(figure, "Model predictions versus one year of simulated data")
        figure_legend(
            figure,
            [
                Line2D(
                    [0],
                    [0],
                    color=PALETTE.ink,
                    marker="o",
                    linestyle="none",
                    markersize=4,
                ),
                Line2D([0], [0], color=PALETTE.accent, linewidth=1.8),
                Patch(facecolor=PALETTE.accent, alpha=0.22, edgecolor="none"),
                Line2D(
                    [0],
                    [0],
                    color=PALETTE.danger,
                    linewidth=1.5,
                    linestyle=(0, (4, 2.5)),
                ),
                Line2D([0], [0], color=PALETTE.ink, linewidth=1.5),
            ],
            [
                "Realized hourly maxima (one year)",
                "GEV fitted on one week",
                "GEV uncertainty band",
                "Gamma model",
                "Realized annual maximum",
            ],
            bbox_to_anchor=(0.5, 0.93),
            ncol=3,
        )
        figure.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.14, wspace=0.32)
        svg_path, png_path = save_figure(figure, "fig8_validation", FIGURE_DIR)
        print(svg_path)
        print(png_path)
        print("legend: figure-level under title")
    else:
        print("smoke mode: writing validation table only (no fig8 render)")

    print(f"annual_max={annual_max:.5f}")
    print(f"n_hours={n_hours} smoke={is_smoke}")

    gev_hourly = bootstrap_expected_hourly_false_alarms(
        calibration_maxima,
        thresholds,
        n_hours=n_hours,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED + 14,
    )
    for key, threshold in thresholds.items():
        central_mean = n_hours * float(gev_sf(threshold, central))
        _, lo, hi = gev_hourly[key]
        gev_hourly[key] = (central_mean, lo, hi)

    gamma_hourly = gamma_expected_hourly_false_alarms(
        thresholds, energy_gamma, n_hours=n_hours
    )
    realized = realized_hourly_counts(validation_maxima, thresholds)
    write_validation_table(
        TABLE_PATH,
        thresholds,
        gamma_hourly,
        gev_hourly,
        realized,
        n_hours,
    )
    print(f"wrote {TABLE_PATH}")
    for key, threshold in thresholds.items():
        mean, lo, hi = gev_hourly[key]
        print(
            f"{key} t={threshold:.4f} gamma={gamma_hourly[key]:.3e} "
            f"gev={mean:.3e}[{lo:.3e},{hi:.3e}] realized={realized[key]}"
        )


if __name__ == "__main__":
    main()
