from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory

from cache_io import load_json
from config import CACHE_DIR, FALSE_ALARM_HORIZON_HOURS, FIGURE_DIR, SAMPLE_RATE_HZ
from events_segment import (
    SEGMENT_SECONDS,
    choose_detection_threshold,
    event_peak_loudness,
    load_event_segment,
    noise_floor_max,
    step_coords,
)
from fitting import GevFit, gev_ppf
from plot_style import (
    BASE_FONT_SIZE,
    FIGURE_WIDTH_IN,
    PALETTE,
    apply,
    figure_title,
    plot_hairline_trace,
    save_figure,
    style_axes,
    style_legend,
)

GEV_JSON_PATH = CACHE_DIR / "gev_fit.json"
EVENT_DISPLAY_NAMES: tuple[str, str, str] = ("Obvious", "Subtle", "Buried")


def load_u95() -> float:
    payload = load_json(GEV_JSON_PATH)
    fit = GevFit(
        xi=float(payload["xi"]),
        loc=float(payload["mu"]),
        scale=float(payload["sigma"]),
        scipy_c=float(payload.get("scipy_c", -float(payload["xi"]))),
    )
    target_cdf = 0.95 ** (1.0 / FALSE_ALARM_HORIZON_HOURS)
    return gev_ppf(target_cdf, fit)


def shade_events(
    axes: Axes,
    event_starts: tuple[float, float, float],
    duration: float,
):
    for start in event_starts:
        axes.axvspan(
            start,
            start + duration,
            facecolor=PALETTE.accent,
            edgecolor="none",
            alpha=0.16,
            zorder=0,
        )


def label_events(
    axes: Axes,
    event_starts: tuple[float, float, float],
    duration: float,
    labels: tuple[str, str, str],
):
    label_transform = blended_transform_factory(axes.transData, axes.transAxes)
    for start, name in zip(event_starts, labels, strict=True):
        axes.text(
            start + 0.5 * duration,
            1.03,
            name,
            transform=label_transform,
            ha="center",
            va="bottom",
            color=PALETTE.ink,
            fontsize=BASE_FONT_SIZE - 1.0,
            clip_on=False,
            zorder=4,
        )


def main():
    segment = load_event_segment()
    u95 = load_u95()
    threshold = choose_detection_threshold(segment, u95=u95)
    time_seconds = np.arange(segment.trace.size, dtype=np.float64) / SAMPLE_RATE_HZ
    y_extent = float(np.max(np.abs(segment.trace))) * 1.08
    step_time, step_loudness = step_coords(segment.normalized_loudness)

    peaks = [
        event_peak_loudness(
            segment.normalized_loudness,
            start,
            segment.event_duration_seconds,
        )
        for start in segment.event_start_seconds
    ]
    noise_max = noise_floor_max(
        segment.normalized_loudness,
        segment.event_start_seconds,
        segment.event_duration_seconds,
    )

    apply()
    figure, (axes_raw, axes_loud) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.68),
        gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.14},
    )
    style_axes(axes_raw)
    style_axes(axes_loud)

    plot_hairline_trace(
        axes_raw,
        time_seconds,
        segment.trace,
        color=PALETTE.muted,
    )
    shade_events(
        axes_raw,
        segment.event_start_seconds,
        segment.event_duration_seconds,
    )
    label_events(
        axes_raw,
        segment.event_start_seconds,
        segment.event_duration_seconds,
        EVENT_DISPLAY_NAMES,
    )
    axes_raw.set_ylabel("Signal")
    axes_raw.set_ylim(-y_extent, y_extent)
    axes_raw.tick_params(labelbottom=False)

    loudness_line = axes_loud.plot(
        step_time,
        step_loudness,
        color=PALETTE.ink,
        linewidth=1.35,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=3,
        label="Loudness (1-second windows)",
    )[0]
    threshold_line = axes_loud.axhline(
        threshold,
        color=PALETTE.danger,
        linewidth=1.4,
        solid_capstyle="round",
        zorder=2,
        label="Illustrative threshold",
    )
    shade_events(
        axes_loud,
        segment.event_start_seconds,
        segment.event_duration_seconds,
    )
    axes_loud.set_xlabel("Time (seconds)")
    axes_loud.set_ylabel("Loudness (versus quiet night)")
    axes_loud.set_xlim(0.0, SEGMENT_SECONDS)
    y_top = max(
        float(np.max(segment.normalized_loudness)) * 1.12,
        threshold * 1.15,
        u95 * 1.05,
    )
    axes_loud.set_ylim(0.75, y_top)
    style_legend(
        axes_loud,
        handles=[loudness_line, threshold_line],
        loc="upper right",
    )

    figure_title(figure, "Raw signal and windowed loudness, same segment")
    figure.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.10)
    svg_path, png_path = save_figure(figure, "fig4_loudness_detector", FIGURE_DIR)
    print(svg_path)
    print(png_path)
    print(
        f"peaks obvious={peaks[0]:.4f} subtle={peaks[1]:.4f} "
        f"buried={peaks[2]:.4f} noise_max={noise_max:.4f}"
    )
    print(f"t_demo={threshold:.4f} u95={u95:.4f}")


if __name__ == "__main__":
    main()
