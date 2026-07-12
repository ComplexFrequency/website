# One-year false-alarm validation

Simulated hours: **8766**. Rows invert the week-fitted GEV at per-hour exceedance rates $10^{-1}$, $10^{-2}$, $10^{-3}$, plus the decade-safe $u_{95}$ threshold. Counts are hours whose maximum loudness exceeds the threshold.

| row | threshold (loudness) | gamma predicted | GEV predicted (90% bootstrap) | realized |
| --- | ---: | ---: | ---: | ---: |
| $p_{\mathrm{hour}}=10^{-1}$ | 1.1387 | 3.833e+02 | 8.766e+02 [6.215e+02, 1.134e+03] | 804 |
| $p_{\mathrm{hour}}=10^{-2}$ | 1.1620 | 1.442e+01 | 8.766e+01 [2.491e+01, 1.700e+02] | 40 |
| $p_{\mathrm{hour}}=10^{-3}$ | 1.1855 | 3.298e-01 | 8.766e+00 [2.893e-01, 3.289e+01] | 1 |
| $u_{95}$ (10-year 95%) | 1.2658 | 2.803e-08 | 5.129e-03 [0.000e+00, 5.021e-01] | 0 |
