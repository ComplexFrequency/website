from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.random import Generator
from scipy import signal

from config import (
    AR_PHI,
    CHUNK_SAMPLES,
    HIGHPASS_CUTOFF_HZ,
    HIGHPASS_ORDER,
    SAMPLE_RATE_HZ,
    WINDOW_SAMPLES,
)


def ar_coefficients() -> tuple[np.ndarray, np.ndarray]:
    numerator = np.array([1.0], dtype=np.float64)
    denominator = np.array([1.0, -AR_PHI], dtype=np.float64)
    return numerator, denominator


def highpass_sos() -> np.ndarray:
    return signal.butter(
        HIGHPASS_ORDER,
        HIGHPASS_CUTOFF_HZ,
        btype="highpass",
        fs=SAMPLE_RATE_HZ,
        output="sos",
    )


def initial_filter_state() -> tuple[np.ndarray, np.ndarray]:
    numerator, denominator = ar_coefficients()
    ar_zi = signal.lfilter_zi(numerator, denominator) * 0.0
    sos = highpass_sos()
    sos_zi = signal.sosfilt_zi(sos) * 0.0
    return ar_zi, sos_zi


def filter_chunk(
    white: np.ndarray,
    ar_zi: np.ndarray,
    sos_zi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerator, denominator = ar_coefficients()
    colored, ar_zf = signal.lfilter(numerator, denominator, white, zi=ar_zi)
    sos = highpass_sos()
    filtered, sos_zf = signal.sosfilt(sos, colored, zi=sos_zi)
    return filtered.astype(np.float64, copy=False), ar_zf, sos_zf


def generate_raw_trace(
    n_samples: int,
    rng: Generator,
    *,
    burn_in_samples: int = SAMPLE_RATE_HZ * 2,
) -> np.ndarray:
    ar_zi, sos_zi = initial_filter_state()
    total_samples = n_samples + burn_in_samples
    white = rng.standard_normal(total_samples)
    filtered, _, _ = filter_chunk(white, ar_zi, sos_zi)
    return filtered[burn_in_samples:]


def generate_loudness(
    n_windows: int,
    rng: Generator,
    *,
    burn_in_windows: int = 2,
) -> np.ndarray:
    if n_windows <= 0:
        return np.empty(0, dtype=np.float64)

    ar_zi, sos_zi = initial_filter_state()
    loudness = np.empty(n_windows, dtype=np.float64)
    windows_written = 0
    windows_to_skip = burn_in_windows
    leftover = np.empty(0, dtype=np.float64)

    while windows_written < n_windows:
        windows_still_needed = n_windows - windows_written + windows_to_skip
        samples_still_needed = windows_still_needed * WINDOW_SAMPLES - leftover.size
        chunk_size = min(CHUNK_SAMPLES, max(samples_still_needed, WINDOW_SAMPLES))
        white = rng.standard_normal(chunk_size)
        filtered, ar_zi, sos_zi = filter_chunk(white, ar_zi, sos_zi)
        stream = np.concatenate([leftover, filtered]) if leftover.size else filtered
        available_windows = stream.size // WINDOW_SAMPLES
        usable = available_windows * WINDOW_SAMPLES
        leftover = stream[usable:].copy()
        if available_windows == 0:
            continue
        reshaped = stream[:usable].reshape(available_windows, WINDOW_SAMPLES)
        window_std = reshaped.std(axis=1, ddof=0)

        if windows_to_skip > 0:
            skip = min(windows_to_skip, window_std.size)
            window_std = window_std[skip:]
            windows_to_skip -= skip
            if window_std.size == 0:
                continue

        take = min(window_std.size, n_windows - windows_written)
        loudness[windows_written : windows_written + take] = window_std[:take]
        windows_written += take

    return loudness


def iter_loudness_chunks(
    n_windows: int,
    rng: Generator,
    *,
    burn_in_windows: int = 2,
) -> Iterator[np.ndarray]:
    remaining = n_windows
    ar_zi, sos_zi = initial_filter_state()
    windows_to_skip = burn_in_windows
    leftover = np.empty(0, dtype=np.float64)

    while remaining > 0:
        windows_still_needed = remaining + windows_to_skip
        samples_still_needed = windows_still_needed * WINDOW_SAMPLES - leftover.size
        chunk_size = min(CHUNK_SAMPLES, max(samples_still_needed, WINDOW_SAMPLES))
        white = rng.standard_normal(chunk_size)
        filtered, ar_zi, sos_zi = filter_chunk(white, ar_zi, sos_zi)
        stream = np.concatenate([leftover, filtered]) if leftover.size else filtered
        available_windows = stream.size // WINDOW_SAMPLES
        usable = available_windows * WINDOW_SAMPLES
        leftover = stream[usable:].copy()
        if available_windows == 0:
            continue
        reshaped = stream[:usable].reshape(available_windows, WINDOW_SAMPLES)
        window_std = reshaped.std(axis=1, ddof=0)

        if windows_to_skip > 0:
            skip = min(windows_to_skip, window_std.size)
            window_std = window_std[skip:]
            windows_to_skip -= skip
            if window_std.size == 0:
                continue

        take = min(window_std.size, remaining)
        yield window_std[:take].copy()
        remaining -= take


def add_burst(
    trace: np.ndarray,
    start_sample: int,
    duration_samples: int,
    burst_std: float,
    rng: Generator,
) -> np.ndarray:
    contaminated = trace.copy()
    end_sample = min(start_sample + duration_samples, contaminated.size)
    if start_sample >= contaminated.size or start_sample < 0:
        return contaminated
    length = end_sample - start_sample
    contaminated[start_sample:end_sample] += rng.standard_normal(length) * burst_std
    return contaminated


def verify_chunk_continuity(
    n_samples: int,
    rng: Generator,
    *,
    chunk_samples: int = 50_000,
    border_radius: int = 8,
) -> dict[str, float]:
    if n_samples < chunk_samples * 2:
        raise ValueError("n_samples must cover at least two chunks")

    white = rng.standard_normal(n_samples)
    ar_zi, sos_zi = initial_filter_state()
    continuous, _, _ = filter_chunk(white, ar_zi, sos_zi)

    ar_zi, sos_zi = initial_filter_state()
    pieces: list[np.ndarray] = []
    offset = 0
    while offset < n_samples:
        end = min(offset + chunk_samples, n_samples)
        piece, ar_zi, sos_zi = filter_chunk(white[offset:end], ar_zi, sos_zi)
        pieces.append(piece)
        offset = end
    chunked = np.concatenate(pieces)

    max_abs_diff = float(np.max(np.abs(continuous - chunked)))
    border = chunk_samples
    left = continuous[border - border_radius : border]
    right = continuous[border : border + border_radius]
    jump_continuous = float(np.abs(right[0] - left[-1]))
    jump_chunked = float(
        np.abs(chunked[border] - chunked[border - 1])
    )
    return {
        "max_abs_diff": max_abs_diff,
        "jump_continuous": jump_continuous,
        "jump_chunked": jump_chunked,
        "border_jump_diff": abs(jump_continuous - jump_chunked),
    }
