"""Distortion risk measures for prediction market fair value estimation.

Provides a family of probability distortion functions from the actuarial
science and financial economics literature, adapted for real-time
prediction market pricing.

The core idea is that prediction market prices embed a systematic risk
premium that can be modelled as a concave distortion of the underlying
physical probability:

    p_market = g(p_physical)

where g: [0, 1] -> [0, 1] is a concave, increasing distortion function
satisfying g(0) = 0 and g(1) = 1.

This module implements three single-parameter distortion families:

1. **Probit distortion** (Wang 2000):
   g(p) = Phi(Phi^{-1}(p) + lambda)

   The most analytically tractable. lambda > 0 implies systematic
   overpricing; the overpricing ratio g(p)/p is monotonically
   decreasing in p, producing the well-known favorite-longshot bias
   as a *theorem* rather than an empirical anomaly.

2. **Dual power distortion** (Denneberg 1994):
   g(p) = 1 - (1 - p)^{1/(1 + rho)}

   Distorts the survival function. Rho > 0 implies risk loading.
   More aggressive in the tails than probit distortion.

3. **Proportional hazard distortion** (Wang 1995):
   g(p) = p^{1/(1 + rho)}

   Distorts the cumulative hazard. Simplest closed-form inverse.
   Widely used in insurance premium calculation.

All distortions are parameterised by a single risk aversion parameter,
keeping the decomposition parsimonious and identifiable from data.

References:
    Wang, S. S. (2000). A class of distortion operators for pricing
        financial and insurance risks. J. Risk and Insurance 67(1), 15-36.
    Denneberg, D. (1994). Non-additive Measure and Integral. Kluwer.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Bounds to avoid infinities in probit / inverse-probit
_EPS = 1e-10
_LO = _EPS
_HI = 1.0 - _EPS


def _clamp(p: float) -> float:
    """Clamp probability to (0, 1) open interval."""
    return max(_LO, min(_HI, p))


# -- Lightweight normal CDF / PPF without scipy dependency ----------------
# Uses the rational approximation from Abramowitz & Stegun (1964) for CDF
# and Beasley-Springer-Moro for PPF.  Accuracy ~ 1e-7 absolute error,
# sufficient for trading signals (we are not publishing p-values here).


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun 26.2.17."""
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(
        -(x * x)
    )
    return 0.5 * (1.0 + sign * y)


def _norm_ppf(p: float) -> float:
    """Standard normal quantile via Beasley-Springer-Moro algorithm."""
    p = _clamp(p)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


# ── Distortion base class ────────────────────────────────────────────────


class DistortionFunction(ABC):
    """Abstract probability distortion g: [0, 1] -> [0, 1].

    A valid distortion satisfies:
    - g(0) = 0, g(1) = 1
    - g is non-decreasing
    - g is concave (for risk-averse pricing)
    """

    @abstractmethod
    def distort(self, p: float) -> float:
        """Apply distortion: p_market = g(p_physical)."""

    @abstractmethod
    def inverse(self, p_mkt: float) -> float:
        """Invert distortion: p_physical = g^{-1}(p_market)."""

    @property
    @abstractmethod
    def param(self) -> float:
        """The scalar risk-aversion parameter."""

    @property
    @abstractmethod
    def family(self) -> str:
        """Distortion family name."""

    def risk_premium(self, p_physical: float) -> float:
        """Compute the risk premium: g(p) - p."""
        return self.distort(p_physical) - p_physical

    def extract_physical(self, p_market: float) -> float:
        """Extract physical probability from observed market price."""
        return self.inverse(p_market)

    def overpricing_ratio(self, p_physical: float) -> float:
        """Compute the overpricing ratio: g(p) / p."""
        if p_physical <= _EPS:
            return 1.0
        return self.distort(p_physical) / p_physical

    def marginal_premium(self, p_physical: float, dp: float = 1e-5) -> float:
        """Numerical derivative of risk premium w.r.t. physical probability."""
        p_lo = max(_LO, p_physical - dp)
        p_hi = min(_HI, p_physical + dp)
        return (self.risk_premium(p_hi) - self.risk_premium(p_lo)) / (p_hi - p_lo)


# ── Probit distortion (Wang 2000) ────────────────────────────────────────


class ProbitDistortion(DistortionFunction):
    """Probit-space constant-shift distortion.

    g(p) = Phi(Phi^{-1}(p) + lambda)

    where Phi is the standard normal CDF.

    Properties (lambda > 0):
    - All events are overpriced: g(p) > p for all p in (0, 1)
    - Overpricing ratio g(p)/p is monotonically decreasing in p
    - This *generates* the favorite-longshot bias as a mathematical
      consequence — longshots (low p) are overpriced by a larger
      multiple than favorites (high p)
    - The distortion is equivalent to a Girsanov measure change with
      constant drift shift in probit space
    """

    def __init__(self, lam: float = 0.0) -> None:
        self._lam = lam

    @property
    def param(self) -> float:
        return self._lam

    @property
    def family(self) -> str:
        return 'probit'

    def distort(self, p: float) -> float:
        p = _clamp(p)
        return _norm_cdf(_norm_ppf(p) + self._lam)

    def inverse(self, p_mkt: float) -> float:
        p_mkt = _clamp(p_mkt)
        return _norm_cdf(_norm_ppf(p_mkt) - self._lam)

    def implied_lambda(self, p_market: float, p_physical: float) -> float:
        """Compute the implied lambda from observed market and physical prices.

        lambda = Phi^{-1}(p_market) - Phi^{-1}(p_physical)
        """
        return _norm_ppf(_clamp(p_market)) - _norm_ppf(_clamp(p_physical))


# ── Dual power distortion ────────────────────────────────────────────────


class DualPowerDistortion(DistortionFunction):
    """Dual power distortion of the survival function (Denneberg 1994).

    g(p) = 1 - (1 - p)^{1 + rho}

    For rho > 0, the exponent exceeds 1, so (1-p)^{1+rho} < (1-p)
    for all p in (0,1), yielding g(p) > p — systematic overpricing.
    More aggressive than probit in the right tail (high p).
    """

    def __init__(self, rho: float = 0.0) -> None:
        if rho < -1.0:
            raise ValueError('rho must be > -1 for dual power distortion')
        self._rho = rho

    @property
    def param(self) -> float:
        return self._rho

    @property
    def family(self) -> str:
        return 'dual_power'

    def distort(self, p: float) -> float:
        p = _clamp(p)
        exponent = 1.0 + self._rho
        return 1.0 - (1.0 - p) ** exponent

    def inverse(self, p_mkt: float) -> float:
        p_mkt = _clamp(p_mkt)
        exponent = 1.0 / (1.0 + self._rho)
        return 1.0 - (1.0 - p_mkt) ** exponent


# ── Proportional hazard distortion ───────────────────────────────────────


class ProportionalHazardDistortion(DistortionFunction):
    """Proportional hazard distortion (Wang 1995).

    g(p) = p^{1 / (1 + rho)}

    The simplest distortion family — power transform of the CDF.
    Rho > 0 implies risk loading.
    """

    def __init__(self, rho: float = 0.0) -> None:
        if rho < -1.0:
            raise ValueError('rho must be > -1 for proportional hazard distortion')
        self._rho = rho

    @property
    def param(self) -> float:
        return self._rho

    @property
    def family(self) -> str:
        return 'proportional_hazard'

    def distort(self, p: float) -> float:
        p = _clamp(p)
        exponent = 1.0 / (1.0 + self._rho)
        return p**exponent

    def inverse(self, p_mkt: float) -> float:
        p_mkt = _clamp(p_mkt)
        exponent = 1.0 + self._rho
        return p_mkt**exponent


# ── Distortion comparison utilities ──────────────────────────────────────


@dataclass(frozen=True)
class DistortionComparison:
    """Side-by-side comparison of multiple distortion families."""

    p_physical: float
    results: dict[str, float]  # family -> distorted price
    premiums: dict[str, float]  # family -> risk premium


def compare_distortions(
    p_physical: float,
    distortions: list[DistortionFunction],
) -> DistortionComparison:
    """Compare multiple distortion functions at a given physical probability."""
    results: dict[str, float] = {}
    premiums: dict[str, float] = {}
    for d in distortions:
        key = f'{d.family}({d.param:.3f})'
        results[key] = d.distort(p_physical)
        premiums[key] = d.risk_premium(p_physical)
    return DistortionComparison(
        p_physical=p_physical, results=results, premiums=premiums
    )
