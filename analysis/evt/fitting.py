from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from config import FALSE_ALARM_BUDGET, FALSE_ALARM_HORIZON_HOURS


@dataclass(frozen=True)
class GammaFit:
    shape: float
    scale: float
    mean: float
    variance: float


@dataclass(frozen=True)
class GevFit:
    xi: float
    loc: float
    scale: float
    scipy_c: float


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    per_hour_false_alarm_rate: float
    horizon_hours: int
    horizon_false_alarm_probability: float


def fit_gamma_moments(samples: np.ndarray) -> GammaFit:
    mean = float(np.mean(samples))
    variance = float(np.var(samples, ddof=1)) if samples.size > 1 else 0.0
    if mean <= 0.0 or variance <= 0.0:
        raise ValueError("gamma moment matching needs positive mean and variance")
    scale = variance / mean
    shape = mean / scale
    return GammaFit(shape=shape, scale=scale, mean=mean, variance=variance)


def gamma_survival(x: np.ndarray | float, fit: GammaFit) -> np.ndarray | float:
    return stats.gamma.sf(x, a=fit.shape, scale=fit.scale)


def gamma_ppf(probability: float, fit: GammaFit) -> float:
    return float(stats.gamma.ppf(probability, a=fit.shape, scale=fit.scale))


def naive_chi2_loudness_params(window_samples: int) -> GammaFit:
    shape = (window_samples - 1) / 2.0
    scale = np.sqrt(2.0 / (window_samples - 1))
    mean = shape * scale
    variance = shape * scale**2
    return GammaFit(shape=shape, scale=scale, mean=float(mean), variance=float(variance))


def naive_chi2_window_std_ppf(
    probability: float,
    window_samples: int,
) -> float:
    chi2_df = window_samples - 1
    chi2_quantile = stats.chi2.ppf(probability, df=chi2_df)
    return float(np.sqrt(chi2_quantile / window_samples))


def naive_chi2_hourly_max_ppf(
    probability: float,
    window_samples: int,
    windows_per_hour: int,
) -> float:
    per_window_probability = probability ** (1.0 / windows_per_hour)
    return naive_chi2_window_std_ppf(per_window_probability, window_samples)


def fit_gev(block_maxima: np.ndarray) -> GevFit:
    scipy_c, loc, scale = stats.genextreme.fit(block_maxima)
    xi = -float(scipy_c)
    return GevFit(xi=xi, loc=float(loc), scale=float(scale), scipy_c=float(scipy_c))


def gev_cdf(x: np.ndarray | float, fit: GevFit) -> np.ndarray | float:
    return stats.genextreme.cdf(x, c=-fit.xi, loc=fit.loc, scale=fit.scale)


def gev_sf(x: np.ndarray | float, fit: GevFit) -> np.ndarray | float:
    return stats.genextreme.sf(x, c=-fit.xi, loc=fit.loc, scale=fit.scale)


def gev_ppf(probability: float, fit: GevFit) -> float:
    return float(
        stats.genextreme.ppf(probability, c=-fit.xi, loc=fit.loc, scale=fit.scale)
    )


def gev_pdf(x: np.ndarray | float, fit: GevFit) -> np.ndarray | float:
    return stats.genextreme.pdf(x, c=-fit.xi, loc=fit.loc, scale=fit.scale)


def design_threshold_from_gev(
    fit: GevFit,
    *,
    horizon_hours: int = FALSE_ALARM_HORIZON_HOURS,
    false_alarm_budget: float = FALSE_ALARM_BUDGET,
) -> ThresholdResult:
    survival_per_hour = 1.0 - (1.0 - false_alarm_budget) ** (1.0 / horizon_hours)
    per_hour_cdf = 1.0 - survival_per_hour
    threshold = gev_ppf(per_hour_cdf, fit)
    horizon_false_alarm_probability = 1.0 - per_hour_cdf**horizon_hours
    return ThresholdResult(
        threshold=threshold,
        per_hour_false_alarm_rate=survival_per_hour,
        horizon_hours=horizon_hours,
        horizon_false_alarm_probability=horizon_false_alarm_probability,
    )


def design_threshold_from_gamma_hourly(
    fit: GammaFit,
    windows_per_hour: int,
    *,
    horizon_hours: int = FALSE_ALARM_HORIZON_HOURS,
    false_alarm_budget: float = FALSE_ALARM_BUDGET,
) -> ThresholdResult:
    survival_per_hour = 1.0 - (1.0 - false_alarm_budget) ** (1.0 / horizon_hours)
    per_hour_cdf = 1.0 - survival_per_hour
    hourly_max_ppf_probability = per_hour_cdf
    per_window_cdf = hourly_max_ppf_probability ** (1.0 / windows_per_hour)
    threshold = gamma_ppf(per_window_cdf, fit)
    return ThresholdResult(
        threshold=threshold,
        per_hour_false_alarm_rate=survival_per_hour,
        horizon_hours=horizon_hours,
        horizon_false_alarm_probability=1.0 - per_hour_cdf**horizon_hours,
    )


def return_level(fit: GevFit, return_period_hours: float) -> float:
    probability = 1.0 - 1.0 / return_period_hours
    return gev_ppf(probability, fit)
