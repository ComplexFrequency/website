from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cache_io import load_array, load_json
from config import (
    CACHE_DIR,
    FALSE_ALARM_HORIZON_HOURS,
    FIGURE_DIR,
    SEED,
)
from events_segment import EVENT_LABELS, event_peak_loudness, load_event_segment
from fitting import GevFit, fit_gev, gev_cdf, gev_ppf
from plot_style import (
    BASE_FONT_SIZE,
    FIGURE_WIDTH_IN,
    PALETTE,
    apply,
    figure_title,
    save_figure,
    style_axes,
    style_legend,
)

GEV_JSON_PATH = CACHE_DIR / "gev_fit.json"
HOURLY_CACHE_PATH = CACHE_DIR / "calibration_hourly_maxima.npy"
N_BOOTSTRAP: int = 1000
RELIABILITY_MARKERS: tuple[float, ...] = (0.50, 0.90, 0.95, 0.99)


def load_central_fit(path: Path) -> GevFit:
    payload = load_json(path)
    xi = float(payload["xi"])
    return GevFit(
        xi=xi,
        loc=float(payload["mu"]),
        scale=float(payload["sigma"]),
        scipy_c=float(payload.get("scipy_c", -xi)),
    )


def zero_false_alarm_probability(
    threshold: np.ndarray | float,
    fit: GevFit,
    horizon_hours: int = FALSE_ALARM_HORIZON_HOURS,
) -> np.ndarray | float:
    cdf = np.asarray(gev_cdf(threshold, fit), dtype=np.float64)
    cdf = np.clip(cdf, 1e-300, 1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_prob = horizon_hours * np.log(cdf)
        probability = np.exp(log_prob)
    probability = np.where(np.isfinite(probability), probability, 0.0)
    if np.isscalar(threshold):
        return float(probability)
    return probability


def threshold_for_reliability(
    reliability: float,
    fit: GevFit,
    horizon_hours: int = FALSE_ALARM_HORIZON_HOURS,
) -> float:
    target_cdf = reliability ** (1.0 / horizon_hours)
    return gev_ppf(target_cdf, fit)


def bootstrap_reliability_envelope(
    maxima: np.ndarray,
    thresholds: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = maxima.size
    curves = np.empty((n_bootstrap, thresholds.size), dtype=np.float64)
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
        if fit.scale <= 0.0 or not np.isfinite(fit.xi):
            continue
        curves[kept] = zero_false_alarm_probability(thresholds, fit)
        kept += 1
    if kept < n_bootstrap:
        raise RuntimeError(f"bootstrap only kept {kept} of {n_bootstrap} fits")
    lower = np.quantile(curves, 0.05, axis=0)
    upper = np.quantile(curves, 0.95, axis=0)
    return lower, upper


def subtle_event_peak() -> float:
    segment = load_event_segment()
    subtle_index = EVENT_LABELS.index("subtle")
    return event_peak_loudness(
        segment.normalized_loudness,
        segment.event_start_seconds[subtle_index],
        segment.event_duration_seconds,
    )


def main():
    central = load_central_fit(GEV_JSON_PATH)
    maxima = load_array(HOURLY_CACHE_PATH)
    subtle_peak = subtle_event_peak()

    marker_thresholds = {
        reliability: threshold_for_reliability(reliability, central)
        for reliability in RELIABILITY_MARKERS
    }
    threshold_95 = marker_thresholds[0.95]

    x_low = 1.15
    x_high = max(subtle_peak * 1.08, marker_thresholds[0.99] * 1.08, threshold_95 * 1.12)
    thresholds = np.linspace(x_low, x_high, 400)
    central_curve = np.asarray(
        zero_false_alarm_probability(thresholds, central),
        dtype=np.float64,
    )
    band_low, band_high = bootstrap_reliability_envelope(
        maxima,
        thresholds,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED + 7,
    )

    apply()
    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.54))
    style_axes(axes)

    axes.fill_between(
        thresholds,
        band_low,
        band_high,
        color=PALETTE.accent,
        alpha=0.22,
        linewidth=0,
        zorder=1,
        label="90% uncertainty band",
    )
    axes.plot(
        thresholds,
        central_curve,
        color=PALETTE.ink,
        linewidth=1.9,
        solid_capstyle="round",
        zorder=3,
        label="Best estimate",
    )
    for reliability in RELIABILITY_MARKERS:
        x_mark = marker_thresholds[reliability]
        y_mark = float(zero_false_alarm_probability(x_mark, central))
        axes.plot(
            x_mark,
            y_mark,
            marker="o",
            markersize=5.0,
            color=PALETTE.danger,
            markeredgecolor=PALETTE.danger,
            zorder=4,
        )
        axes.annotate(
            f"{int(round(reliability * 100))}%",
            xy=(x_mark, y_mark),
            xytext=(7, -8),
            textcoords="offset points",
            color=PALETTE.danger,
            fontsize=BASE_FONT_SIZE - 1.5,
            ha="left",
            va="top",
            arrowprops={
                "arrowstyle": "-",
                "color": PALETTE.danger,
                "lw": 0.65,
                "shrinkA": 0,
                "shrinkB": 1.5,
            },
            zorder=5,
        )

    axes.set_xlabel("Threshold (loudness versus quiet night)")
    axes.set_ylabel("Chance of zero false alarms over 10 years")
    axes.set_xlim(x_low, x_high)
    y_low = -0.02
    axes.set_ylim(y_low, 1.05)
    axes.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axes.plot(
        subtle_peak,
        y_low,
        marker="^",
        markersize=7.0,
        color=PALETTE.ink,
        alpha=0.55,
        markeredgewidth=0,
        clip_on=False,
        zorder=2,
        label="Subtle event's peak loudness",
    )

    legend_handles = [
        Line2D([0], [0], color=PALETTE.ink, linewidth=1.9, label="Best estimate"),
        Patch(facecolor=PALETTE.accent, alpha=0.22, edgecolor="none", label="90% uncertainty band"),
        Line2D(
            [0],
            [0],
            color=PALETTE.ink,
            marker="^",
            markersize=7.0,
            alpha=0.55,
            linestyle="None",
            label="Subtle event's peak loudness",
        ),
    ]
    style_legend(axes, handles=legend_handles, loc="lower right")

    figure_title(figure, "Probability of zero false alarms in 10 years versus threshold")
    figure.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.14)
    svg_path, png_path = save_figure(figure, "fig7_ten_year_dial", FIGURE_DIR)
    print(svg_path)
    print(png_path)
    print(f"subtle_peak={subtle_peak:.5f}")
    print(f"threshold_95_percent={threshold_95:.5f}")


if __name__ == "__main__":
    main()
