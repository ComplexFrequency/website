from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory

from config import FIGURE_DIR, SAMPLE_RATE_HZ
from events_segment import SEGMENT_SECONDS, load_event_segment
from plot_style import (
    BASE_FONT_SIZE,
    PALETTE,
    figure_title,
    new_figure,
    plot_hairline_trace,
    save_figure,
)

EVENT_DISPLAY_NAMES: tuple[str, str, str] = ("Obvious", "Subtle", "Buried")


def shade_and_label_events(
    axes: Axes,
    event_starts: tuple[float, float, float],
    duration: float,
    labels: tuple[str, str, str],
):
    label_transform = blended_transform_factory(axes.transData, axes.transAxes)
    for start, label in zip(event_starts, labels, strict=True):
        axes.axvspan(
            start,
            start + duration,
            facecolor=PALETTE.accent,
            edgecolor="none",
            alpha=0.16,
            zorder=0,
        )
        axes.text(
            start + 0.5 * duration,
            1.02,
            label,
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
    time_seconds = np.arange(segment.trace.size, dtype=np.float64) / SAMPLE_RATE_HZ
    y_extent = float(np.max(np.abs(segment.trace))) * 1.08

    figure, axes = new_figure(height_ratio=0.44)
    plot_hairline_trace(axes, time_seconds, segment.trace)
    shade_and_label_events(
        axes,
        segment.event_start_seconds,
        segment.event_duration_seconds,
        EVENT_DISPLAY_NAMES,
    )
    axes.set_xlabel("Time (seconds)")
    axes.set_ylabel("Signal")
    axes.set_xlim(0.0, SEGMENT_SECONDS)
    axes.set_ylim(-y_extent, y_extent)
    figure_title(figure, "Raw signal with three events of decreasing amplitude")

    figure.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.16)
    svg_path, png_path = save_figure(figure, "fig3_events_raw", FIGURE_DIR)
    print(svg_path)
    print(png_path)
    print("legend: none; event labels above bands")


if __name__ == "__main__":
    main()
