"""Hybrid calibration engine: batch MLE + online EWMA.

Two calibration modes for different operational contexts:

1. **Batch MLE** — when historical resolved contracts are available,
   runs the full Wang Transform MLE from Yang (2026) for statistically
   rigorous parameter estimates with proper standard errors.

2. **Online EWMA** — during live trading, updates lambda estimates
   incrementally as contracts resolve, with exponential recency weighting
   and hierarchical shrinkage across categories.

The calibrator uses empirically-estimated priors from Yang (2026):

    Platform priors:  Polymarket=0.166, Kalshi=0.187
    Category priors:  Sports=0.070, Crypto=0.253, Politics=0.054
    Volume insight:   >$10K volume → lambda≈0 (premium competed away)

These priors serve as warm starts for MLE and as the default when no
calibration data is available, ensuring the system produces reasonable
estimates from day one.

Reference:
    Yang, Y. (2026). "Pricing Prediction Markets: Risk Premiums,
    Incomplete Markets, and a Decomposition Framework." Working Paper, UIUC.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from oracle3.pricing.distortion import _norm_ppf
from oracle3.pricing.wang_mle import (
    LAMBDA_BY_CATEGORY,
    LAMBDA_KALSHI,
    LAMBDA_POLYMARKET,
    LAMBDA_POOLED,
)

logger = logging.getLogger(__name__)

# Smoothed boundary outcomes
_Y_SMOOTH_1 = 0.999
_Y_SMOOTH_0 = 0.001

_DEFAULT_CACHE = Path(os.path.expanduser('~/.oracle3/cache/calibrator.json'))

# Platform-level priors from Yang (2026) Table 1
_PLATFORM_PRIORS: dict[str, float] = {
    'polymarket': LAMBDA_POLYMARKET,
    'kalshi': LAMBDA_KALSHI,
    'default': LAMBDA_POOLED,
}


@dataclass
class CategoryEstimate:
    """Running estimate of the distortion parameter for one category."""

    category: str
    lambda_hat: float = 0.0
    n_contracts: int = 0
    sum_lambda: float = 0.0
    sum_weight: float = 0.0
    sum_sq_lambda: float = 0.0
    last_updated: float = 0.0
    # Batch MLE results (if available)
    mle_lambda: float | None = None
    mle_se: float | None = None
    mle_n: int | None = None

    @property
    def variance(self) -> float:
        if self.n_contracts < 2:
            return float('inf')
        if self.sum_weight < 1e-12:
            return float('inf')
        return max(0.0, self.sum_sq_lambda / self.sum_weight - self.lambda_hat**2)

    @property
    def std_error(self) -> float:
        v = self.variance
        if v == float('inf') or self.n_contracts < 2:
            return float('inf')
        return math.sqrt(v / self.n_contracts)

    @property
    def best_lambda(self) -> float:
        """Best available lambda: MLE if available, else EWMA."""
        if self.mle_lambda is not None:
            return self.mle_lambda
        return self.lambda_hat


@dataclass
class CalibrationReport:
    """Summary of calibration state."""

    global_lambda: float
    global_n: int
    categories: dict[str, CategoryEstimate]
    last_updated: float
    staleness_hours: float
    has_mle: bool


class OnlineCalibrator:
    """Hybrid batch MLE + online EWMA calibration engine.

    Parameters
    ----------
    alpha:
        EWMA decay factor (default 0.05). Higher = faster adaptation.
    shrinkage_kappa:
        Hierarchical shrinkage strength (default 20).
    default_lambda:
        Fallback lambda when no data available (default: pooled estimate).
    platform:
        Platform name for platform-level priors ('polymarket' or 'kalshi').
    cache_path:
        Path for persisting calibration state.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        shrinkage_kappa: float = 20.0,
        default_lambda: float | None = None,
        platform: str = 'polymarket',
        cache_path: Path | None = None,
    ) -> None:
        self._alpha = alpha
        self._kappa = shrinkage_kappa
        self._platform = platform.lower()
        self._default_lambda = (
            default_lambda
            if default_lambda is not None
            else _PLATFORM_PRIORS.get(self._platform, LAMBDA_POOLED)
        )
        self._cache_path = cache_path or _DEFAULT_CACHE
        self._categories: dict[str, CategoryEstimate] = {}
        self._global_lambda = self._default_lambda
        self._global_n = 0
        self._load_cache()

    # ── Batch MLE calibration ─────────────────────────────────────────────

    def calibrate_batch(
        self,
        prices: object,
        outcomes: object,
        category: str = 'default',
        volumes: object | None = None,
        durations_hours: object | None = None,
        spreads: object | None = None,
    ) -> float:
        """Run batch MLE calibration from historical resolved contracts.

        This is the most statistically rigorous calibration method.
        Uses the full Wang Transform MLE from Yang (2026).

        Returns the estimated lambda.
        """
        from oracle3.pricing.wang_mle import WangMLE

        mle = WangMLE()

        # Build covariates if any are provided
        covariates = None
        cov_names = None
        if any(x is not None for x in [volumes, durations_hours, spreads]):
            covariates = mle.build_design_matrix(
                volumes=volumes,
                durations_hours=durations_hours,
                prices=prices,
                spreads=spreads,
            )
            cov_names = getattr(covariates, '_covariate_names', None)

        result = mle.fit(
            prices=prices,
            outcomes=outcomes,
            covariates=covariates,
            covariate_names=cov_names,
        )

        # Store MLE results
        est = self._categories.get(category)
        if est is None:
            est = CategoryEstimate(category=category)
            self._categories[category] = est

        est.mle_lambda = result.lambda_hat
        est.mle_se = result.se_robust[0] if result.se_robust else None
        est.mle_n = result.n_obs
        est.last_updated = time.time()

        logger.info(
            'Batch MLE calibration [%s]: lambda=%.4f (SE=%.4f, N=%d, converged=%s)',
            category,
            result.lambda_hat,
            est.mle_se or 0,
            result.n_obs,
            result.converged,
        )

        self._save_cache()
        return result.lambda_hat

    # ── Online EWMA calibration ───────────────────────────────────────────

    def update(
        self,
        outcome: int,
        final_price: float,
        category: str = 'default',
    ) -> float:
        """Update calibration with a single resolved contract.

        Returns the contract-level implied lambda.
        """
        y_smooth = _Y_SMOOTH_1 if outcome == 1 else _Y_SMOOTH_0
        final_price = max(0.001, min(0.999, final_price))
        implied_lambda = _norm_ppf(final_price) - _norm_ppf(y_smooth)

        est = self._categories.get(category)
        if est is None:
            est = CategoryEstimate(category=category)
            self._categories[category] = est

        est.n_contracts += 1
        w = self._alpha
        est.sum_weight = w + (1.0 - w) * est.sum_weight
        est.sum_lambda = w * implied_lambda + (1.0 - w) * est.sum_lambda
        est.sum_sq_lambda = (
            w * implied_lambda * implied_lambda + (1.0 - w) * est.sum_sq_lambda
        )
        est.lambda_hat = est.sum_lambda / max(est.sum_weight, 1e-12)
        est.last_updated = time.time()

        # Update global
        self._global_n += 1
        global_w = 1.0 / self._global_n
        self._global_lambda = (
            global_w * implied_lambda + (1.0 - global_w) * self._global_lambda
        )

        self._save_cache()
        return implied_lambda

    # ── Query interface ───────────────────────────────────────────────────

    def get_lambda(self, category: str = 'default') -> float:
        """Get calibrated lambda with hierarchical shrinkage.

        Priority: MLE result > EWMA estimate > category prior > platform prior.
        All shrunk toward the global estimate.
        """
        est = self._categories.get(category)

        # No data for this category
        if est is None or (est.n_contracts == 0 and est.mle_lambda is None):
            # Try category-level prior from the paper
            cat_prior = LAMBDA_BY_CATEGORY.get(category.lower())
            if cat_prior is not None:
                return cat_prior
            return self._global_lambda if self._global_n > 0 else self._default_lambda

        # Use best available estimate with shrinkage
        raw_lambda = est.best_lambda
        effective_n = est.mle_n if est.mle_lambda is not None else est.n_contracts
        effective_n = effective_n or 1

        w_cat = effective_n / (effective_n + self._kappa)
        global_lam = self._global_lambda if self._global_n > 0 else self._default_lambda
        return w_cat * raw_lambda + (1.0 - w_cat) * global_lam

    def get_confidence(self, category: str = 'default') -> float:
        """Confidence score (0-1) for the category estimate."""
        est = self._categories.get(category)
        if est is None:
            return 0.1  # prior-only

        # MLE-based confidence
        if est.mle_lambda is not None and est.mle_n is not None:
            # High confidence if SE is small relative to lambda
            if est.mle_se is not None and est.mle_se > 0 and est.mle_lambda != 0:
                t_stat = abs(est.mle_lambda / est.mle_se)
                return min(1.0, 1.0 / (1.0 + math.exp(-(t_stat - 2.0))))
            n_factor = 1.0 / (1.0 + math.exp(-(math.sqrt(est.mle_n) - 10.0)))
            return n_factor

        # EWMA-based confidence
        if est.n_contracts == 0:
            return 0.1
        n_factor = 1.0 / (1.0 + math.exp(-(math.sqrt(est.n_contracts) - 5.0)))
        var_penalty = 1.0
        if est.variance != float('inf') and est.variance > 0:
            var_penalty = 1.0 / (1.0 + est.variance)
        return min(1.0, n_factor * var_penalty)

    def get_estimate(self, category: str = 'default') -> CategoryEstimate | None:
        return self._categories.get(category)

    def report(self) -> CalibrationReport:
        last_update = max(
            (e.last_updated for e in self._categories.values()), default=0.0
        )
        staleness = (
            (time.time() - last_update) / 3600.0 if last_update > 0 else float('inf')
        )
        has_mle = any(e.mle_lambda is not None for e in self._categories.values())
        return CalibrationReport(
            global_lambda=self._global_lambda,
            global_n=self._global_n,
            categories=dict(self._categories),
            last_updated=last_update,
            staleness_hours=staleness,
            has_mle=has_mle,
        )

    def reset(self, category: str | None = None) -> None:
        if category is not None:
            self._categories.pop(category, None)
        else:
            self._categories.clear()
            self._global_lambda = self._default_lambda
            self._global_n = 0
        self._save_cache()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text())
            self._global_lambda = data.get('global_lambda', self._default_lambda)
            self._global_n = data.get('global_n', 0)
            for cat_data in data.get('categories', []):
                est = CategoryEstimate(**cat_data)
                self._categories[est.category] = est
            logger.info(
                'Loaded calibrator: %d categories, global_lambda=%.4f',
                len(self._categories),
                self._global_lambda,
            )
        except Exception:
            logger.warning('Failed to load calibrator cache, starting fresh')

    def _save_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._cache_path.with_suffix('.tmp')
            data = {
                'global_lambda': self._global_lambda,
                'global_n': self._global_n,
                'categories': [asdict(e) for e in self._categories.values()],
            }
            tmp_path.write_text(json.dumps(data, indent=2))
            tmp_path.rename(self._cache_path)
        except Exception:
            logger.warning('Failed to save calibrator cache', exc_info=True)
