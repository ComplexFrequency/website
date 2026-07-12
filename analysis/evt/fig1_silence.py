from __future__ import annotations

import numpy as np

from config import CACHE_DIR, FIGURE_DIR, SAMPLE_RATE_HZ, SEED
from noise import generate_raw_trace
from plot_style import (
    figure_title,
    load_or_compute,
    new_figure,
    plot_hairline_trace,
    save_figure,
)

TRACE_SECONDS: float = 10.0
TRACE_SAMPLES: int = int(TRACE_SECONDS * SAMPLE_RATE_HZ)
CACHE_PATH = CACHE_DIR / "fig1_silence_trace.npy"


def compute_trace() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return generate_raw_trace(TRACE_SAMPLES, rng)


def main():
    trace = load_or_compute(CACHE_PATH, compute_trace)
    time_seconds = np.arange(trace.size, dtype=np.float64) / SAMPLE_RATE_HZ
    y_extent = float(np.max(np.abs(trace))) * 1.08

    figure, axes = new_figure(height_ratio=0.42)
    plot_hairline_trace(axes, time_seconds, trace)

    axes.set_xlabel("Time (seconds)")
    axes.set_ylabel("Signal")
    axes.set_xlim(0.0, TRACE_SECONDS)
    axes.set_ylim(-y_extent, y_extent)
    figure_title(figure, "Background noise, 10-second segment")

    figure.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.16)
    svg_path, png_path = save_figure(figure, "fig1_silence", FIGURE_DIR)
    print(svg_path)
    print(png_path)
    print("legend: none")


if __name__ == "__main__":
    main()
