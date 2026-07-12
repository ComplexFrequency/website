from __future__ import annotations

import numpy as np

from config import WINDOWS_PER_HOUR


def window_std(samples: np.ndarray, window_samples: int) -> np.ndarray:
    n_windows = samples.size // window_samples
    if n_windows == 0:
        return np.empty(0, dtype=np.float64)
    usable = n_windows * window_samples
    reshaped = samples[:usable].reshape(n_windows, window_samples)
    return reshaped.std(axis=1, ddof=0)


def normalize_loudness(
    loudness: np.ndarray,
    quiet_night_level: float,
) -> np.ndarray:
    if quiet_night_level <= 0.0:
        raise ValueError("quiet_night_level must be positive")
    return loudness / quiet_night_level


def quiet_night_level(calibration_loudness: np.ndarray) -> float:
    return float(np.median(calibration_loudness))


def hourly_maxima(
    loudness: np.ndarray,
    windows_per_hour: int = WINDOWS_PER_HOUR,
) -> np.ndarray:
    n_hours = loudness.size // windows_per_hour
    if n_hours == 0:
        return np.empty(0, dtype=np.float64)
    usable = n_hours * windows_per_hour
    reshaped = loudness[:usable].reshape(n_hours, windows_per_hour)
    return reshaped.max(axis=1)


def exceedance_rate(values: np.ndarray, threshold: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values > threshold))


def empirical_survival(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values)
    n = sorted_values.size
    if n == 0:
        return np.ones_like(grid, dtype=np.float64)
    ranks = np.searchsorted(sorted_values, grid, side="right")
    return 1.0 - ranks / n
