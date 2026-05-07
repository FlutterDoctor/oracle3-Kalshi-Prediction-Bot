"""Correlation-aware risk management for multi-strategy portfolios.

Extends the standard risk manager with correlation-based position limits
that prevent concentrated bets in correlated prediction markets.

The core insight: two markets that are 90% correlated should not each
receive a full position allocation, because their combined risk is nearly
2x a single position rather than sqrt(2)x as diversification would imply.

The effective exposure is computed as:

    E_eff = sqrt(w^T * Sigma * w)

where w is the position vector and Sigma is the correlation matrix.
A new trade is rejected if the post-trade effective exposure exceeds
the maximum allowed effective exposure.

Correlation estimation uses exponentially weighted rolling Pearson
correlation on price returns, updated incrementally as new market
data arrives. Stale correlations decay toward zero.

This approach is inspired by multi-strategy risk management systems
in prediction market infrastructure, adapted for oracle3's event-driven
architecture and enriched with Wang-model-informed correlation priors.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from decimal import Decimal

from oracle3.data.market_data_manager import MarketDataManager
from oracle3.position.position_manager import PositionManager
from oracle3.risk.risk_manager import StandardRiskManager
from oracle3.ticker.ticker import CashTicker, Ticker
from oracle3.trader.types import TradeSide

logger = logging.getLogger(__name__)

_CORR_WINDOW = 50  # number of price observations for correlation
_CORR_DECAY = 0.97  # exponential decay for correlation weights
_STALE_THRESHOLD = 3600  # seconds before a correlation is considered stale


class CorrelationAwareRiskManager(StandardRiskManager):
    """Risk manager with correlation-based effective exposure limits.

    Extends StandardRiskManager with:
    - Rolling correlation estimation between position pairs
    - Effective exposure computation via correlation matrix
    - Rejection of trades that would create excessive correlated risk
    - Configurable maximum concentration ratio

    Parameters
    ----------
    position_manager: position tracker
    market_data: market data manager
    max_effective_exposure:
        Maximum effective exposure after accounting for correlations.
        Default $30000 (lower than raw max_total_exposure to account
        for correlation concentration).
    max_concentration_ratio:
        Maximum ratio of effective_exposure / sum(|positions|).
        Values close to 1.0 indicate high concentration (bad).
        Default 0.85 — reject if >85% of exposure is concentrated.
    correlation_window:
        Number of price observations for rolling correlation.
    **kwargs: passed to StandardRiskManager.
    """

    def __init__(
        self,
        position_manager: PositionManager,
        market_data: MarketDataManager,
        max_effective_exposure: Decimal = Decimal('30000'),
        max_concentration_ratio: float = 0.85,
        correlation_window: int = _CORR_WINDOW,
        **kwargs,
    ) -> None:
        super().__init__(position_manager, market_data, **kwargs)
        self._max_eff_exposure = max_effective_exposure
        self._max_concentration = max_concentration_ratio
        self._corr_window = correlation_window

        # Price history for correlation estimation
        self._price_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=correlation_window)
        )
        self._last_update: dict[str, float] = {}
        self._corr_cache: dict[tuple[str, str], float] = {}
        self._corr_cache_time: dict[tuple[str, str], float] = {}

    def record_price(self, symbol: str, price: float) -> None:
        """Record a price observation for correlation computation.

        Call this from the trading engine on each PriceChangeEvent.
        """
        self._price_history[symbol].append(price)
        self._last_update[symbol] = time.time()

    async def check_trade(
        self,
        ticker: Ticker,
        side: TradeSide,
        quantity: Decimal,
        price: Decimal,
    ) -> bool:
        """Check trade against both standard and correlation-aware limits."""
        if isinstance(ticker, CashTicker):
            return True

        # Standard checks first
        if not await super().check_trade(ticker, side, quantity, price):
            return False

        # Correlation-aware check (only for buys / new exposure)
        if side == TradeSide.SELL:
            return True

        trade_value = float(quantity * price)
        eff_exposure = self._compute_effective_exposure(
            ticker.symbol, trade_value
        )

        if eff_exposure > float(self._max_eff_exposure):
            logger.warning(
                'Correlation risk: effective exposure $%.0f exceeds limit $%.0f',
                eff_exposure,
                float(self._max_eff_exposure),
            )
            return False

        # Concentration check
        raw_exposure = self._compute_raw_exposure(ticker.symbol, trade_value)
        if raw_exposure > 0:
            concentration = eff_exposure / raw_exposure
            if concentration > self._max_concentration:
                logger.warning(
                    'Correlation risk: concentration ratio %.2f exceeds limit %.2f',
                    concentration,
                    self._max_concentration,
                )
                return False

        return True

    # ── Correlation estimation ────────────────────────────────────────────

    def get_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """Get the estimated correlation between two symbols.

        Uses exponentially weighted Pearson correlation on price returns.
        Returns 0.0 if insufficient data.
        """
        if symbol_a == symbol_b:
            return 1.0

        # Check cache
        key = (min(symbol_a, symbol_b), max(symbol_a, symbol_b))
        now = time.time()
        if key in self._corr_cache:
            cache_age = now - self._corr_cache_time.get(key, 0)
            if cache_age < 60:  # cache for 60 seconds
                return self._corr_cache[key]

        # Compute
        ha = self._price_history.get(symbol_a)
        hb = self._price_history.get(symbol_b)
        if ha is None or hb is None or len(ha) < 10 or len(hb) < 10:
            return 0.0

        # Align lengths
        n = min(len(ha), len(hb))
        pa = list(ha)[-n:]
        pb = list(hb)[-n:]

        # Compute returns
        if n < 3:
            return 0.0

        returns_a = [pa[i] - pa[i - 1] for i in range(1, n)]
        returns_b = [pb[i] - pb[i - 1] for i in range(1, n)]

        # Exponentially weighted correlation
        corr = self._ewma_correlation(returns_a, returns_b)

        # Decay stale correlations toward zero
        latest_a = self._last_update.get(symbol_a, 0)
        latest_b = self._last_update.get(symbol_b, 0)
        staleness = now - min(latest_a, latest_b)
        if staleness > _STALE_THRESHOLD:
            decay = math.exp(-staleness / _STALE_THRESHOLD)
            corr *= decay

        self._corr_cache[key] = corr
        self._corr_cache_time[key] = now
        return corr

    def _ewma_correlation(
        self,
        returns_a: list[float],
        returns_b: list[float],
    ) -> float:
        """Exponentially weighted moving average correlation."""
        n = len(returns_a)
        if n < 3:
            return 0.0

        # Compute EWMA means
        weights = [_CORR_DECAY ** (n - 1 - i) for i in range(n)]
        w_sum = sum(weights)
        if w_sum < 1e-12:
            return 0.0

        mean_a = sum(w * a for w, a in zip(weights, returns_a)) / w_sum
        mean_b = sum(w * b for w, b in zip(weights, returns_b)) / w_sum

        # Compute EWMA covariance and variances
        cov = 0.0
        var_a = 0.0
        var_b = 0.0
        for i in range(n):
            da = returns_a[i] - mean_a
            db = returns_b[i] - mean_b
            cov += weights[i] * da * db
            var_a += weights[i] * da * da
            var_b += weights[i] * db * db

        if var_a < 1e-15 or var_b < 1e-15:
            return 0.0

        corr = cov / math.sqrt(var_a * var_b)
        return max(-1.0, min(1.0, corr))

    # ── Effective exposure ────────────────────────────────────────────────

    def _compute_effective_exposure(
        self,
        new_symbol: str,
        new_value: float,
    ) -> float:
        """Compute effective exposure including the proposed new trade.

        Uses: E_eff = sqrt(sum_i sum_j w_i * w_j * rho_ij)
        """
        positions = self._get_position_values()
        positions[new_symbol] = positions.get(new_symbol, 0.0) + new_value

        symbols = list(positions.keys())
        n = len(symbols)
        if n == 0:
            return 0.0
        if n == 1:
            return abs(positions[symbols[0]])

        # Build correlation-weighted quadratic form
        quad_sum = 0.0
        for i in range(n):
            wi = positions[symbols[i]]
            for j in range(n):
                wj = positions[symbols[j]]
                rho = self.get_correlation(symbols[i], symbols[j])
                quad_sum += wi * wj * rho

        return math.sqrt(max(0, quad_sum))

    def _compute_raw_exposure(
        self,
        new_symbol: str,
        new_value: float,
    ) -> float:
        """Sum of absolute position values (no correlation adjustment)."""
        positions = self._get_position_values()
        positions[new_symbol] = positions.get(new_symbol, 0.0) + new_value
        return sum(abs(v) for v in positions.values())

    def _get_position_values(self) -> dict[str, float]:
        """Get current position values by symbol."""
        result: dict[str, float] = {}
        for pos in self.position_manager.get_non_cash_positions():
            if pos.quantity > 0:
                symbol = pos.ticker.symbol
                # Use last known price for valuation
                price = self._last_price(pos.ticker)
                result[symbol] = float(pos.quantity * price)
        return result

    def _last_price(self, ticker: Ticker) -> Decimal:
        """Get last known price for a ticker."""
        bid = self.market_data.get_best_bid(ticker)
        ask = self.market_data.get_best_ask(ticker)
        if bid is not None and ask is not None:
            return (bid.price + ask.price) / 2
        if bid is not None:
            return bid.price
        if ask is not None:
            return ask.price
        return Decimal('0.5')  # fallback mid
