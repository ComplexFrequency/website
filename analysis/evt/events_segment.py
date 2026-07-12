from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cache_io import load_or_compute_array
from config import (
    BURST_DURATION_SECONDS,
    BURST_STD_BURIED,
    BURST_STD_OBVIOUS,
    BURST_STD_SUBTLE,
    CACHE_DIR,
    SAMPLE_RATE_HZ,
    SEED,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
)
from features import normalize_loudness, quiet_night_level, window_std
from noise import add_burst, generate_raw_trace

SEGMENT_SECONDS: float = 60.0
SEGMENT_SAMPLES: int = int(SEGMENT_SECONDS * SAMPLE_RATE_HZ)
EVENT_START_SECONDS: tuple[float, float, float] = (10.0, 28.0, 46.0)
EVENT_LABELS: tuple[str, str, str] = ("obvious", "subtle", "buried")
EVENT_RATIOS: tuple[float, float, float] = (
    BURST_STD_OBVIOUS,
    BURST_STD_SUBTLE,
    BURST_STD_BURIED,
)
TRACE_CACHE_PATH = CACHE_DIR / "fig3_fig4_events_trace.npy"
LOUDNESS_CACHE_PATH = CACHE_DIR / "fig3_fig4_events_loudness.npy"


@dataclass(frozen=True)
class EventSegment:
    trace: np.ndarray
    loudness: np.ndarray
    quiet_level: float
    normalized_loudness: np.ndarray
    event_start_seconds: tuple[float, float, float]
    event_duration_seconds: float
    event_labels: tuple[str, str, str]
    event_ratios: tuple[float, float, float]
    noise_std: float


def compute_contaminated_trace(
    ratios: tuple[float, float, float] | None = None,
) -> np.ndarray:
    burst_ratios = EVENT_RATIOS if ratios is None else ratios
    rng = np.random.default_rng(SEED + 3)
    background = generate_raw_trace(SEGMENT_SAMPLES, rng)
    noise_std = float(np.std(background, ddof=0))
    duration_samples = int(BURST_DURATION_SECONDS * SAMPLE_RATE_HZ)
    contaminated = background
    for start_seconds, ratio in zip(EVENT_START_SECONDS, burst_ratios, strict=True):
        start_sample = int(start_seconds * SAMPLE_RATE_HZ)
        contaminated = add_burst(
            contaminated,
            start_sample,
            duration_samples,
            noise_std * ratio,
            rng,
        )
    return contaminated


def compute_raw_loudness() -> np.ndarray:
    trace = load_or_compute_array(TRACE_CACHE_PATH, compute_contaminated_trace)
    return window_std(trace, WINDOW_SAMPLES)


def load_event_segment() -> EventSegment:
    trace = load_or_compute_array(TRACE_CACHE_PATH, compute_contaminated_trace)
    loudness = load_or_compute_array(LOUDNESS_CACHE_PATH, compute_raw_loudness)
    event_mask = event_window_mask(
        loudness.size,
        EVENT_START_SECONDS,
        BURST_DURATION_SECONDS,
    )
    quiet_level = quiet_night_level(loudness[~event_mask])
    normalized = normalize_loudness(loudness, quiet_level)
    noise_std = float(np.std(trace, ddof=0))
    return EventSegment(
        trace=trace,
        loudness=loudness,
        quiet_level=quiet_level,
        normalized_loudness=normalized,
        event_start_seconds=EVENT_START_SECONDS,
        event_duration_seconds=BURST_DURATION_SECONDS,
        event_labels=EVENT_LABELS,
        event_ratios=EVENT_RATIOS,
        noise_std=noise_std,
    )


def measure_peaks_for_ratios(
    ratios: tuple[float, float, float],
) -> tuple[float, float, float, float, float]:
    trace = compute_contaminated_trace(ratios=ratios)
    loudness = window_std(trace, WINDOW_SAMPLES)
    event_mask = event_window_mask(
        loudness.size,
        EVENT_START_SECONDS,
        BURST_DURATION_SECONDS,
    )
    quiet_level = quiet_night_level(loudness[~event_mask])
    normalized = normalize_loudness(loudness, quiet_level)
    peaks = [
        event_peak_loudness(
            normalized,
            start,
            BURST_DURATION_SECONDS,
        )
        for start in EVENT_START_SECONDS
    ]
    noise_max = noise_floor_max(
        normalized,
        EVENT_START_SECONDS,
        BURST_DURATION_SECONDS,
    )
    return peaks[0], peaks[1], peaks[2], noise_max, quiet_level


def event_window_mask(
    n_windows: int,
    event_starts: tuple[float, float, float],
    duration_seconds: float,
) -> np.ndarray:
    mask = np.zeros(n_windows, dtype=bool)
    for start in event_starts:
        first = int(np.floor(start / WINDOW_SECONDS))
        last = int(np.ceil((start + duration_seconds) / WINDOW_SECONDS))
        mask[max(0, first) : min(n_windows, last)] = True
    return mask


def event_peak_loudness(
    normalized_loudness: np.ndarray,
    start_seconds: float,
    duration_seconds: float,
) -> float:
    first = int(np.floor(start_seconds / WINDOW_SECONDS))
    last = int(np.ceil((start_seconds + duration_seconds) / WINDOW_SECONDS))
    last = min(normalized_loudness.size, last)
    first = max(0, first)
    if first >= last:
        return float("nan")
    return float(np.max(normalized_loudness[first:last]))


def noise_floor_max(
    normalized_loudness: np.ndarray,
    event_starts: tuple[float, float, float],
    duration_seconds: float,
) -> float:
    mask = event_window_mask(normalized_loudness.size, event_starts, duration_seconds)
    quiet = normalized_loudness[~mask]
    if quiet.size == 0:
        return float(np.max(normalized_loudness))
    return float(np.max(quiet))


def choose_detection_threshold(
    segment: EventSegment,
    *,
    u95: float | None = None,
) -> float:
    peaks = [
        event_peak_loudness(
            segment.normalized_loudness,
            start,
            segment.event_duration_seconds,
        )
        for start in segment.event_start_seconds
    ]
    obvious_peak, subtle_peak, buried_peak = peaks
    noise_max = noise_floor_max(
        segment.normalized_loudness,
        segment.event_start_seconds,
        segment.event_duration_seconds,
    )
    reject_ceiling = max(noise_max, buried_peak)
    t_demo = reject_ceiling + 0.01 * max(reject_ceiling, 1.0)
    if u95 is not None:
        t_demo = min(t_demo, 0.5 * (reject_ceiling + u95))
        t_demo = max(t_demo, reject_ceiling * 1.002)
    if subtle_peak <= t_demo:
        raise ValueError(
            f"subtle peak {subtle_peak:.4f} not above demo threshold {t_demo:.4f}"
        )
    return t_demo


def step_coords(
    values: np.ndarray,
    window_seconds: float = WINDOW_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(values.size + 1, dtype=np.float64) * window_seconds
    time = np.repeat(edges, 2)[1:-1]
    level = np.repeat(values, 2)
    return time, level
