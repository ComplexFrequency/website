from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cache_io import save_array, save_json
from config import (
    CACHE_DIR,
    FALSE_ALARM_HORIZON_HOURS,
    HOURS_PER_YEAR,
    SEED,
    WINDOWS_PER_HOUR,
)
from features import quiet_night_level
from fitting import GevFit, gev_ppf
from noise import iter_loudness_chunks

RAW_CALIBRATION_PATH = CACHE_DIR / "calibration_loudness_raw.npy"
GEV_JSON_PATH = CACHE_DIR / "gev_fit.json"
HOURLY_MAXIMA_PATH = CACHE_DIR / "validation_hourly_maxima.npy"
WINDOW_COUNTS_PATH = CACHE_DIR / "validation_window_exceedances.npy"
HOURLY_COUNTS_PATH = CACHE_DIR / "validation_hourly_exceedances.npy"
META_PATH = CACHE_DIR / "validation_meta.json"
RELIABILITY_LEVELS: tuple[float, ...] = (0.50, 0.90, 0.95, 0.99)
PROGRESS_EVERY_HOURS: int = 500


def load_gev_fit(path: Path) -> GevFit:
    payload = json.loads(path.read_text())
    xi = float(payload["xi"])
    return GevFit(
        xi=xi,
        loc=float(payload["mu"]),
        scale=float(payload["sigma"]),
        scipy_c=float(payload.get("scipy_c", -xi)),
    )


def candidate_thresholds(fit: GevFit) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for reliability in RELIABILITY_LEVELS:
        target_cdf = reliability ** (1.0 / FALSE_ALARM_HORIZON_HOURS)
        key = f"{int(round(reliability * 100))}pct"
        thresholds[key] = gev_ppf(target_cdf, fit)
    return thresholds


def run_validation(n_hours: int, *, seed: int) -> dict[str, object]:
    if not RAW_CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"missing calibration cache: {RAW_CALIBRATION_PATH}")
    if not GEV_JSON_PATH.exists():
        raise FileNotFoundError(f"missing GEV fit: {GEV_JSON_PATH}")

    calibration = np.load(RAW_CALIBRATION_PATH)
    quiet_level = quiet_night_level(calibration)
    fit = load_gev_fit(GEV_JSON_PATH)
    thresholds = candidate_thresholds(fit)
    threshold_values = np.array(list(thresholds.values()), dtype=np.float64)
    threshold_keys = list(thresholds.keys())

    n_windows = n_hours * WINDOWS_PER_HOUR
    rng = np.random.default_rng(seed)
    hourly_maxima = np.empty(n_hours, dtype=np.float64)
    window_exceedances = np.zeros(threshold_values.size, dtype=np.int64)
    hourly_exceedances = np.zeros(threshold_values.size, dtype=np.int64)

    hours_done = 0
    started = time.perf_counter()
    next_progress_hour = PROGRESS_EVERY_HOURS

    def log(message: str):
        print(message, flush=True)

    log(
        f"validating hours={n_hours} windows={n_windows} "
        f"quiet_level={quiet_level:.6f}"
    )
    log("thresholds: " + ", ".join(f"{k}={v:.5f}" for k, v in thresholds.items()))

    pending = np.empty(0, dtype=np.float64)
    for chunk in iter_loudness_chunks(n_windows, rng):
        normalized = chunk / quiet_level
        if pending.size:
            stream = np.concatenate([pending, normalized])
        else:
            stream = normalized
        n_complete_hours = stream.size // WINDOWS_PER_HOUR
        usable = n_complete_hours * WINDOWS_PER_HOUR
        if n_complete_hours > 0:
            hours_block = stream[:usable].reshape(n_complete_hours, WINDOWS_PER_HOUR)
            block_max = hours_block.max(axis=1)
            hourly_maxima[hours_done : hours_done + n_complete_hours] = block_max
            window_exceedances += np.sum(
                hours_block[:, :, None] > threshold_values[None, None, :],
                axis=(0, 1),
            )
            hourly_exceedances += np.sum(
                block_max[:, None] > threshold_values[None, :],
                axis=0,
            )
            hours_done += n_complete_hours
        pending = stream[usable:].copy()
        while hours_done >= next_progress_hour:
            elapsed = time.perf_counter() - started
            rate = hours_done / elapsed if elapsed > 0 else 0.0
            log(
                f"progress hours={hours_done}/{n_hours} "
                f"elapsed_s={elapsed:.1f} rate_h_per_s={rate:.3f} "
                f"window_counts={window_exceedances.tolist()} "
                f"hourly_counts={hourly_exceedances.tolist()}"
            )
            next_progress_hour += PROGRESS_EVERY_HOURS

    wall_time = time.perf_counter() - started
    if hours_done != n_hours:
        raise RuntimeError(f"expected {n_hours} hours, got {hours_done}")
    log(
        f"progress hours={hours_done}/{n_hours} "
        f"elapsed_s={wall_time:.1f} rate_h_per_s={hours_done / wall_time:.3f} "
        f"window_counts={window_exceedances.tolist()} "
        f"hourly_counts={hourly_exceedances.tolist()}"
    )
    meta = {
        "n_hours": n_hours,
        "n_windows": n_windows,
        "quiet_level": quiet_level,
        "seed": seed,
        "wall_time_s": wall_time,
        "threshold_keys": threshold_keys,
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "window_exceedances": {
            k: int(v) for k, v in zip(threshold_keys, window_exceedances, strict=True)
        },
        "hourly_exceedances": {
            k: int(v) for k, v in zip(threshold_keys, hourly_exceedances, strict=True)
        },
        "hourly_max_summary": {
            "min": float(np.min(hourly_maxima)),
            "median": float(np.median(hourly_maxima)),
            "mean": float(np.mean(hourly_maxima)),
            "max": float(np.max(hourly_maxima)),
        },
    }
    return {
        "hourly_maxima": hourly_maxima,
        "window_exceedances": window_exceedances,
        "hourly_exceedances": hourly_exceedances,
        "meta": meta,
    }


def persist_results(result: dict[str, object], *, full_year: bool):
    meta = result["meta"]
    assert isinstance(meta, dict)
    if full_year:
        save_array(HOURLY_MAXIMA_PATH, result["hourly_maxima"])
        save_array(WINDOW_COUNTS_PATH, result["window_exceedances"])
        save_array(HOURLY_COUNTS_PATH, result["hourly_exceedances"])
        save_json(META_PATH, meta)
        print(f"wrote {HOURLY_MAXIMA_PATH}", flush=True)
        print(f"wrote {WINDOW_COUNTS_PATH}", flush=True)
        print(f"wrote {HOURLY_COUNTS_PATH}", flush=True)
        print(f"wrote {META_PATH}", flush=True)
    else:
        smoke_dir = CACHE_DIR / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        save_array(smoke_dir / "validation_hourly_maxima.npy", result["hourly_maxima"])
        save_array(
            smoke_dir / "validation_window_exceedances.npy",
            result["window_exceedances"],
        )
        save_array(
            smoke_dir / "validation_hourly_exceedances.npy",
            result["hourly_exceedances"],
        )
        save_json(smoke_dir / "validation_meta.json", meta)
        print(f"wrote smoke outputs under {smoke_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-simulate noise and count threshold exceedances."
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=HOURS_PER_YEAR,
        help=f"hours of loudness to simulate (default {HOURS_PER_YEAR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED + 11,
        help="RNG seed for the validation stream",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_hours = int(args.hours)
    if n_hours <= 0:
        raise ValueError("--hours must be positive")
    full_year = n_hours >= HOURS_PER_YEAR
    result = run_validation(n_hours, seed=int(args.seed))
    persist_results(result, full_year=full_year)
    meta = result["meta"]
    assert isinstance(meta, dict)
    print(f"wall_time_s={meta['wall_time_s']:.2f}")
    print(f"hourly_max_summary={meta['hourly_max_summary']}")
    print(f"window_exceedances={meta['window_exceedances']}")
    print(f"hourly_exceedances={meta['hourly_exceedances']}")


if __name__ == "__main__":
    main()
