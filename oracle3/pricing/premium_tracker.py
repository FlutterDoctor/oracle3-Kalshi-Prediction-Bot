"""Risk premium lifecycle tracking for prediction market contracts.

Prediction market risk premiums exhibit a characteristic decay pattern:
premiums are highest shortly after contract creation and diminish as
the contract approaches its resolution date. This pattern is consistent
with the theoretical insight that required risk compensation is
proportional to residual uncertainty — which naturally decreases
as information accrues and the time horizon shrinks.

This module tracks premium evolution over contract lifetimes and fits
decay models that enable timing-based trading strategies:

- Enter positions early when the risk premium is large (price > fair value).
- Ride the predictable premium decay as the contract matures.
- Exit before resolution when the premium has mostly evaporated.

The decay model uses a polynomial basis in normalised time-to-resolution:

    premium(tau) = gamma_1 * tau + gamma_2 * tau^2

where tau in [0, 1] is normalised time-to-resolution (tau=1 at creation,
tau=0 at resolution). The constraint premium(0) = 0 is enforced by the
basis construction (no intercept term).
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PremiumObservation:
    """A single observation of risk premium at a point in contract lifecycle."""

    ticker_symbol: str
    timestamp: float  # unix epoch
    market_price: float
    fair_value: float
    risk_premium: float  # market_price - fair_value
    tau: float  # normalized time-to-resolution [0, 1]
    lambda_param: float  # distortion parameter used


@dataclass(frozen=True)
class PremiumSnapshot:
    """Current state of premium tracking for a contract."""

    ticker_symbol: str
    current_premium: float
    initial_premium: float  # premium at earliest observation
    premium_decay_pct: float  # % of initial premium that has decayed
    half_life_fraction: float  # estimated fraction of lifetime at which premium halves
    tau_current: float  # current tau (time-to-resolution fraction)
    n_observations: int
    gamma_1: float  # linear decay coefficient
    gamma_2: float  # quadratic decay coefficient
    r_squared: float  # fit quality


@dataclass
class DecayModel:
    """Polynomial decay model for a single contract or category.

    premium(tau) = gamma_1 * tau + gamma_2 * tau^2
    """

    gamma_1: float = 0.0
    gamma_2: float = 0.0
    r_squared: float = 0.0
    n_obs: int = 0

    def predict(self, tau: float) -> float:
        """Predict premium at a given time-to-resolution fraction."""
        return self.gamma_1 * tau + self.gamma_2 * tau * tau

    @property
    def half_life_fraction(self) -> float:
        """Estimate the tau at which premium drops to half its initial value.

        Solves: gamma_1 * tau + gamma_2 * tau^2 = 0.5 * (gamma_1 + gamma_2)
        Returns NaN if no valid solution.
        """
        initial = self.gamma_1 + self.gamma_2
        if initial <= 0:
            return float('nan')
        target = 0.5 * initial

        # Solve gamma_2 * tau^2 + gamma_1 * tau - target = 0
        if abs(self.gamma_2) < 1e-10:
            # Linear case
            if abs(self.gamma_1) < 1e-10:
                return float('nan')
            sol = target / self.gamma_1
            return sol if 0 < sol < 1 else float('nan')

        discriminant = self.gamma_1**2 + 4 * self.gamma_2 * target
        if discriminant < 0:
            return float('nan')

        sqrt_d = math.sqrt(discriminant)
        t1 = (-self.gamma_1 + sqrt_d) / (2 * self.gamma_2)
        t2 = (-self.gamma_1 - sqrt_d) / (2 * self.gamma_2)

        # Pick the solution in (0, 1)
        for t in [t1, t2]:
            if 0 < t < 1:
                return t
        return float('nan')


class PremiumTracker:
    """Track risk premium evolution over contract lifecycles.

    Parameters
    ----------
    max_observations_per_contract:
        Maximum number of observations to retain per contract.
    min_observations_for_fit:
        Minimum observations required to fit a decay model.
    """

    def __init__(
        self,
        max_observations_per_contract: int = 500,
        min_observations_for_fit: int = 10,
    ) -> None:
        self._max_obs = max_observations_per_contract
        self._min_fit = min_observations_for_fit
        self._observations: dict[str, deque[PremiumObservation]] = defaultdict(
            lambda: deque(maxlen=max_observations_per_contract)
        )
        self._models: dict[str, DecayModel] = {}
        # Category-level aggregate decay models
        self._category_observations: dict[str, deque[PremiumObservation]] = defaultdict(
            lambda: deque(maxlen=2000)
        )
        self._category_models: dict[str, DecayModel] = {}

    def record(
        self,
        ticker_symbol: str,
        market_price: float,
        fair_value: float,
        tau: float,
        lambda_param: float = 0.0,
        category: str = 'default',
    ) -> None:
        """Record a premium observation for a contract.

        Parameters
        ----------
        ticker_symbol:
            Identifier for the contract.
        market_price:
            Observed market price.
        fair_value:
            Model-estimated fair value (physical probability).
        tau:
            Normalised time-to-resolution in [0, 1].
        lambda_param:
            Distortion parameter used for this estimate.
        category:
            Contract category for aggregate tracking.
        """
        obs = PremiumObservation(
            ticker_symbol=ticker_symbol,
            timestamp=time.time(),
            market_price=market_price,
            fair_value=fair_value,
            risk_premium=market_price - fair_value,
            tau=max(0.0, min(1.0, tau)),
            lambda_param=lambda_param,
        )
        self._observations[ticker_symbol].append(obs)
        self._category_observations[category].append(obs)

        # Re-fit model if enough observations
        obs_list = self._observations[ticker_symbol]
        if len(obs_list) >= self._min_fit:
            self._models[ticker_symbol] = self._fit_decay(list(obs_list))

        cat_obs = self._category_observations[category]
        if len(cat_obs) >= self._min_fit * 3:
            self._category_models[category] = self._fit_decay(list(cat_obs))

    def get_snapshot(self, ticker_symbol: str) -> PremiumSnapshot | None:
        """Get current premium tracking state for a contract."""
        obs_list = self._observations.get(ticker_symbol)
        if not obs_list or len(obs_list) == 0:
            return None

        latest = obs_list[-1]
        earliest = obs_list[0]
        model = self._models.get(ticker_symbol, DecayModel())

        decay_pct = 0.0
        if abs(earliest.risk_premium) > 1e-6:
            decay_pct = 1.0 - latest.risk_premium / earliest.risk_premium

        return PremiumSnapshot(
            ticker_symbol=ticker_symbol,
            current_premium=latest.risk_premium,
            initial_premium=earliest.risk_premium,
            premium_decay_pct=max(0.0, decay_pct),
            half_life_fraction=model.half_life_fraction,
            tau_current=latest.tau,
            n_observations=len(obs_list),
            gamma_1=model.gamma_1,
            gamma_2=model.gamma_2,
            r_squared=model.r_squared,
        )

    def get_category_model(self, category: str = 'default') -> DecayModel | None:
        """Get the aggregate decay model for a category."""
        return self._category_models.get(category)

    def predict_premium(
        self,
        ticker_symbol: str,
        tau: float,
        category: str = 'default',
    ) -> float | None:
        """Predict the expected premium at a given time-to-resolution.

        Uses contract-level model if available, falls back to category model.
        """
        model = self._models.get(ticker_symbol)
        if model is not None and model.n_obs >= self._min_fit:
            return model.predict(tau)

        cat_model = self._category_models.get(category)
        if cat_model is not None and cat_model.n_obs >= self._min_fit:
            return cat_model.predict(tau)

        return None

    def get_optimal_entry_tau(
        self,
        category: str = 'default',
        min_premium_fraction: float = 0.7,
    ) -> float | None:
        """Estimate the optimal tau for entering a premium decay trade.

        Returns the tau at which the expected premium is at least
        min_premium_fraction of the initial premium. This represents
        the latest entry point that still captures most of the decay.
        """
        model = self._category_models.get(category)
        if model is None or model.n_obs < self._min_fit:
            return None

        initial_premium = model.predict(1.0)
        if initial_premium <= 0:
            return None

        target = min_premium_fraction * initial_premium

        # Binary search for the tau where premium crosses the target
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if model.predict(mid) > target:
                hi = mid
            else:
                lo = mid

        return hi if abs(model.predict(hi) - target) < 0.01 else None

    # ── Internal ──────────────────────────────────────────────────────────

    def _fit_decay(self, observations: list[PremiumObservation]) -> DecayModel:
        """Fit polynomial decay model via weighted least squares.

        premium(tau) = gamma_1 * tau + gamma_2 * tau^2

        No intercept term (premium(0) = 0 is enforced).
        """
        n = len(observations)
        if n < self._min_fit:
            return DecayModel()

        # Build design matrix and response
        sum_t2 = 0.0
        sum_t3 = 0.0
        sum_t4 = 0.0
        sum_ty = 0.0
        sum_t2y = 0.0
        sum_y2 = 0.0
        sum_y = 0.0

        for obs in observations:
            t = obs.tau
            y = obs.risk_premium
            t2 = t * t
            sum_t2 += t2
            sum_t3 += t2 * t
            sum_t4 += t2 * t2
            sum_ty += t * y
            sum_t2y += t2 * y
            sum_y2 += y * y
            sum_y += y

        # Solve 2x2 normal equations:
        # [sum_t2  sum_t3] [gamma_1]   [sum_ty ]
        # [sum_t3  sum_t4] [gamma_2] = [sum_t2y]
        det = sum_t2 * sum_t4 - sum_t3 * sum_t3
        if abs(det) < 1e-15:
            # Degenerate — fall back to linear only
            if abs(sum_t2) < 1e-15:
                return DecayModel(n_obs=n)
            gamma_1 = sum_ty / sum_t2
            return DecayModel(gamma_1=gamma_1, n_obs=n)

        gamma_1 = (sum_t4 * sum_ty - sum_t3 * sum_t2y) / det
        gamma_2 = (sum_t2 * sum_t2y - sum_t3 * sum_ty) / det

        # R-squared
        mean_y = sum_y / n
        ss_tot = sum_y2 - n * mean_y * mean_y
        ss_res = 0.0
        for obs in observations:
            pred = gamma_1 * obs.tau + gamma_2 * obs.tau * obs.tau
            ss_res += (obs.risk_premium - pred) ** 2

        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
        r_sq = max(0.0, r_sq)

        return DecayModel(
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            r_squared=r_sq,
            n_obs=n,
        )
