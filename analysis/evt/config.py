from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SEED: int = 20260711
RNG = np.random.default_rng(SEED)

SAMPLE_RATE_HZ: int = 8000
AR_PHI: float = 0.95
HIGHPASS_CUTOFF_HZ: float = 20.0
HIGHPASS_ORDER: int = 2
CHUNK_SAMPLES: int = 10_000_000

WINDOW_SECONDS: float = 1.0
WINDOW_SAMPLES: int = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)

CALIBRATION_DAYS: int = 7
SECONDS_PER_HOUR: int = 3600
WINDOWS_PER_HOUR: int = SECONDS_PER_HOUR
CALIBRATION_WINDOWS: int = CALIBRATION_DAYS * 24 * WINDOWS_PER_HOUR
CALIBRATION_HOURLY_MAXIMA: int = CALIBRATION_DAYS * 24

BURST_DURATION_SECONDS: float = 3.0
BURST_STD_OBVIOUS: float = 1.5
BURST_STD_SUBTLE: float = 0.68
BURST_STD_BURIED: float = 0.2

FALSE_ALARM_HORIZON_YEARS: float = 10.0
HOURS_PER_YEAR: int = 8766
FALSE_ALARM_HORIZON_HOURS: int = int(FALSE_ALARM_HORIZON_YEARS * HOURS_PER_YEAR)
FALSE_ALARM_BUDGET: float = 0.05

PACKAGE_DIR: Path = Path(__file__).resolve().parent
CACHE_DIR: Path = PACKAGE_DIR / "cache"
FIGURE_DIR: Path = PACKAGE_DIR.parent.parent / "assets" / "images" / "evt-thresholds"


def config_snapshot() -> dict[str, Any]:
    return {
        "SEED": SEED,
        "SAMPLE_RATE_HZ": SAMPLE_RATE_HZ,
        "AR_PHI": AR_PHI,
        "HIGHPASS_CUTOFF_HZ": HIGHPASS_CUTOFF_HZ,
        "HIGHPASS_ORDER": HIGHPASS_ORDER,
        "CHUNK_SAMPLES": CHUNK_SAMPLES,
        "WINDOW_SECONDS": WINDOW_SECONDS,
        "WINDOW_SAMPLES": WINDOW_SAMPLES,
        "CALIBRATION_DAYS": CALIBRATION_DAYS,
        "SECONDS_PER_HOUR": SECONDS_PER_HOUR,
        "WINDOWS_PER_HOUR": WINDOWS_PER_HOUR,
        "CALIBRATION_WINDOWS": CALIBRATION_WINDOWS,
        "CALIBRATION_HOURLY_MAXIMA": CALIBRATION_HOURLY_MAXIMA,
        "BURST_DURATION_SECONDS": BURST_DURATION_SECONDS,
        "BURST_STD_OBVIOUS": BURST_STD_OBVIOUS,
        "BURST_STD_SUBTLE": BURST_STD_SUBTLE,
        "BURST_STD_BURIED": BURST_STD_BURIED,
        "FALSE_ALARM_HORIZON_YEARS": FALSE_ALARM_HORIZON_YEARS,
        "HOURS_PER_YEAR": HOURS_PER_YEAR,
        "FALSE_ALARM_HORIZON_HOURS": FALSE_ALARM_HORIZON_HOURS,
        "FALSE_ALARM_BUDGET": FALSE_ALARM_BUDGET,
    }


def config_fingerprint() -> str:
    payload = json.dumps(config_snapshot(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
