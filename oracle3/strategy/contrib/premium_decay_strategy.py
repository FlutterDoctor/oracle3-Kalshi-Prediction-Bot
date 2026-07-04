"""Premium decay strategy — alpha from predictable risk premium lifecycle.

Exploits the empirical regularity that prediction market risk premiums
decay over contract lifetimes: premiums are highest early in a contract's
life and diminish toward zero as resolution approaches.

This pattern arises because risk compensation is proportional to residual
uncertainty, which naturally decreases as:
- More information accrues (news, polls, data releases)
- The resolution mechanism becomes more predictable
- The time horizon for bearing risk shrinks

The strategy enters positions early in a contract's lifecycle (high premium)
and rides the predictable decay toward resolution (shrinking premium).

Entry logic:
    1. Contract has sufficient remaining lifetime (tau > min_entry_tau)
    2. Current premium > threshold (market is overpricing risk)
    3. Premium tracker predicts positive expected decay
    4. Contract passes quality filters (volume, spread)

Exit logic:
    1. Premium has decayed below target fraction of entry premium
    2. Contract approaches resolution (tau < min_exit_tau)
    3. Premium reversal (premium increases rather than decays)
    4. Max hold time exceeded

Usage::

    oracle3 engine run \\
      --exchange polymarket --mode paper \\
      --strategy-ref oracle3.strategy.contrib.premium_decay_strategy:PremiumDecayStrategy \\
      --strategy-kwargs-json '{"min_premium": 0.03, "trade_size": 25}'
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from decimal import Decimal
from typing import ClassVar

from oracle3.events.events import Event, PriceChangeEvent
from oracle3.pricing.calibrator import OnlineCalibrator
from oracle3.pricing.contract_scorer import ContractScorer
from oracle3.pricing.fair_value import FairValueEstimate, FairValueEstimator
from oracle3.pricing.fees import round_trip_fee_for_ticker
from oracle3.pricing.premium_tracker import PremiumTracker
from oracle3.strategy.quant_strategy import QuantStrategy
from oracle3.ticker.ticker import Ticker
from oracle3.trader.trader import Trader
from oracle3.trader.types import TradeSide

logger = logging.getLogger(__name__)

# Position states
_FLAT = 'flat'
_DECAY_LONG = 'decay_long'  # holding NO, riding premium decay


class PremiumDecayStrategy(QuantStrategy):
    """Ride predictable risk premium decay over contract lifecycles.

    Enters short-risk (buy NO) positions when the risk premium is high
    relative to the contract's remaining lifetime, then exits as the
    premium decays toward zero approaching resolution.

    Parameters
    ----------
    min_premium:
        Minimum absolute risk premium to enter (default 0.03 = 3 cents).
    min_entry_tau:
        Earliest point in lifecycle to enter (default 0.4 = 40% remaining).
        Avoids entering too close to resolution where decay is exhausted.
    max_entry_tau:
        Latest point in lifecycle to enter (default 0.95 = 95% remaining).
        Avoids entering at creation when price discovery is noisy.
    exit_decay_fraction:
        Exit when premium decays to this fraction of entry premium
        (default 0.3 = 30% of original premium remaining).
    min_exit_tau:
        Force exit when contract approaches resolution (default 0.05).
    trade_size:
        Dollar amount per position (default 25).
    max_hold_seconds:
        Maximum hold time before forced exit (default 86400 = 24 hours).
    cooldown_seconds:
        Minimum seconds between trades on same ticker (default 300).
    premium_reversal_tolerance:
        Max allowed premium increase before signalling reversal exit
        (default 0.02 = 2 cents above entry premium).
    fee_rate:
        Kalshi fee-schedule multiplier (default 0.07, Kalshi's real
        standard rate). Fee is nonlinear in price -- computed from the
        real formula rather than a flat rate.
    category:
        Default category for calibration.
    default_lambda:
        Prior distortion parameter.
    """

    name: ClassVar[str] = 'premium_decay'
    version: ClassVar[str] = '1.0.0'
    author: ClassVar[str] = 'oracle3'

    def __init__(
        self,
        min_premium: float = 0.03,
        min_entry_tau: float = 0.40,
        max_entry_tau: float = 0.95,
        exit_decay_fraction: float = 0.30,
        min_exit_tau: float = 0.05,
        trade_size: Decimal = Decimal('25'),
        max_hold_seconds: int = 86400,
        cooldown_seconds: int = 300,
        premium_reversal_tolerance: float = 0.02,
        fee_rate: float = 0.07,
        category: str = 'default',
        default_lambda: float = 0.10,
    ) -> None:
        super().__init__()
        self._min_premium = min_premium
        self._min_entry_tau = min_entry_tau
        self._max_entry_tau = max_entry_tau
        self._exit_decay_frac = exit_decay_fraction
        self._min_exit_tau = min_exit_tau
        self._trade_size = trade_size
        self._max_hold = max_hold_seconds
        self._cooldown = cooldown_seconds
        self._reversal_tolerance = premium_reversal_tolerance
        self._fee_rate = fee_rate
        self._category = category

        # Build pricing stack
        self._calibrator = OnlineCalibrator(default_lambda=default_lambda)
        self._scorer = ContractScorer()
        self._estimator = FairValueEstimator(
            calibrator=self._calibrator, scorer=self._scorer
        )
        self._tracker = PremiumTracker()

        # State
        self._positions: dict[str, str] = defaultdict(lambda: _FLAT)
        self._entry_times: dict[str, float] = {}
        self._entry_premiums: dict[str, float] = {}
        self._entry_taus: dict[str, float] = {}
        self._last_trade: dict[str, float] = defaultdict(float)

        # Metadata: maps ticker_symbol -> estimated tau
        # In practice, tau would come from contract metadata (creation date,
        # resolution date). For now, we track elapsed fraction from first
        # observation as a proxy.
        self._first_seen: dict[str, float] = {}
        self._estimated_duration: dict[str, float] = {}

    async def process_event(self, event: Event, trader: Trader) -> None:
        """Process price events and manage premium decay positions."""
        if self.is_paused():
            return

        if not isinstance(event, PriceChangeEvent):
            return

        ticker = event.ticker
        symbol = ticker.symbol
        price = float(event.price)
        now = time.time()

        if price <= 0.01 or price >= 0.99:
            return

        # Track contract lifecycle
        if symbol not in self._first_seen:
            self._first_seen[symbol] = now
        tau = self._estimate_tau(symbol, now)

        # Compute fair value and record premium observation
        features = self._scorer.extract_features(
            ticker_symbol=symbol,
            category=self._category,
            elapsed_fraction=1.0 - tau,
        )
        estimate = self._estimator.estimate(
            market_price=price, category=self._category, features=features
        )

        # Record observation for premium tracker
        self._tracker.record(
            ticker_symbol=symbol,
            market_price=price,
            fair_value=estimate.fair_value,
            tau=tau,
            lambda_param=estimate.lambda_adjusted,
            category=self._category,
        )

        state = self._positions[symbol]

        # Manage existing positions
        if state == _DECAY_LONG:
            exit_reason = self._check_exit(symbol, estimate, tau, now)
            if exit_reason:
                await self._exit_position(
                    symbol, ticker, trader, estimate, tau, now, exit_reason
                )
            return

        # Evaluate entry
        if state == _FLAT:
            if now - self._last_trade[symbol] < self._cooldown:
                return
            if self._should_enter(symbol, ticker, estimate, tau):
                await self._enter_position(symbol, ticker, trader, estimate, tau, now)

    # ── Entry / exit logic ────────────────────────────────────────────────

    def _should_enter(
        self,
        symbol: str,
        ticker: Ticker,
        estimate: FairValueEstimate,
        tau: float,
    ) -> bool:
        """Evaluate whether to enter a premium decay trade."""
        # Check lifecycle position
        if tau < self._min_entry_tau or tau > self._max_entry_tau:
            return False

        # Check premium magnitude (must be positive = overpriced)
        if estimate.risk_premium < self._min_premium:
            return False

        # Check fee viability: expected decay must exceed round-trip fees
        expected_decay = estimate.risk_premium * (1.0 - self._exit_decay_frac)
        fee_cost = round_trip_fee_for_ticker(
            ticker, estimate.market_price, self._fee_rate
        )
        if expected_decay < fee_cost:
            return False

        # Check premium tracker prediction if available
        predicted = self._tracker.predict_premium(
            symbol, tau=self._min_exit_tau, category=self._category
        )
        if predicted is not None:
            expected_capture = estimate.risk_premium - predicted
            if expected_capture < fee_cost:
                return False

        return True

    def _check_exit(
        self,
        symbol: str,
        estimate: FairValueEstimate,
        tau: float,
        now: float,
    ) -> str:
        """Check exit conditions. Returns reason string or empty string."""
        entry_premium = self._entry_premiums.get(symbol, 0.0)

        # Premium decayed to target
        if (
            entry_premium > 0
            and estimate.risk_premium <= entry_premium * self._exit_decay_frac
        ):
            return 'decay_target_reached'

        # Approaching resolution
        if tau < self._min_exit_tau:
            return 'resolution_approaching'

        # Premium reversal
        if estimate.risk_premium > entry_premium + self._reversal_tolerance:
            return 'premium_reversal'

        # Max hold time
        entry_time = self._entry_times.get(symbol, now)
        if now - entry_time > self._max_hold:
            return 'max_hold_exceeded'

        return ''

    async def _enter_position(
        self,
        symbol: str,
        ticker: Ticker,
        trader: Trader,
        estimate: FairValueEstimate,
        tau: float,
        now: float,
    ) -> None:
        """Enter a premium decay position (buy NO = short risk)."""
        no_ticker = getattr(ticker, 'get_no_ticker', lambda: None)()
        trade_ticker = no_ticker if no_ticker is not None else ticker

        price_dec = Decimal(str(estimate.market_price))
        qty = self._trade_size / max(price_dec, Decimal('0.01'))

        result = await trader.place_order(TradeSide.BUY, trade_ticker, price_dec, qty)
        executed = result.executed
        if executed:
            self._positions[symbol] = _DECAY_LONG
            self._entry_times[symbol] = now
            self._entry_premiums[symbol] = estimate.risk_premium
            self._entry_taus[symbol] = tau
            self._last_trade[symbol] = now

        self.record_decision(
            ticker_name=symbol,
            action='BUY_NO',
            executed=executed,
            reasoning=(
                f'Premium decay entry: premium={estimate.risk_premium:.4f}, '
                f'tau={tau:.2f}, fv={estimate.fair_value:.3f}, '
                f'mkt={estimate.market_price:.3f}, lambda={estimate.lambda_adjusted:.4f}'
            ),
            confidence=estimate.confidence,
            signal_values={
                'risk_premium': estimate.risk_premium,
                'fair_value': estimate.fair_value,
                'market_price': estimate.market_price,
                'tau': tau,
                'lambda': estimate.lambda_adjusted,
            },
        )

    async def _exit_position(
        self,
        symbol: str,
        ticker: Ticker,
        trader: Trader,
        estimate: FairValueEstimate,
        tau: float,
        now: float,
        reason: str,
    ) -> None:
        """Exit a premium decay position."""
        no_ticker = getattr(ticker, 'get_no_ticker', lambda: None)()
        trade_ticker = no_ticker if no_ticker is not None else ticker

        price_dec = Decimal(str(estimate.market_price))
        qty = self._trade_size / max(price_dec, Decimal('0.01'))

        result = await trader.place_order(TradeSide.SELL, trade_ticker, price_dec, qty)
        executed = result.executed
        if executed:
            self._positions[symbol] = _FLAT
            entry_premium = self._entry_premiums.pop(symbol, 0.0)
            entry_tau = self._entry_taus.pop(symbol, 1.0)
            self._entry_times.pop(symbol, None)
            self._last_trade[symbol] = now

            premium_captured = entry_premium - estimate.risk_premium
            logger.info(
                'Premium decay exit [%s]: captured=%.4f (entry=%.4f -> exit=%.4f), '
                'tau: %.2f -> %.2f, reason=%s',
                symbol,
                premium_captured,
                entry_premium,
                estimate.risk_premium,
                entry_tau,
                tau,
                reason,
            )

        self.record_decision(
            ticker_name=symbol,
            action='CLOSE_DECAY_LONG',
            executed=executed,
            reasoning=f'Exit ({reason}): premium={estimate.risk_premium:.4f}, tau={tau:.2f}',
            confidence=estimate.confidence,
            signal_values={
                'risk_premium': estimate.risk_premium,
                'entry_premium': self._entry_premiums.get(symbol, 0.0),
                'tau': tau,
                'exit_reason': hash(reason) % 100,  # numeric encoding for signal_values
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _estimate_tau(self, symbol: str, now: float) -> float:
        """Estimate normalised time-to-resolution.

        Uses estimated duration from contract metadata if available.
        Falls back to a default 30-day assumption.
        """
        duration = self._estimated_duration.get(symbol, 30 * 86400)  # default 30 days
        first_seen = self._first_seen.get(symbol, now)
        elapsed = now - first_seen
        tau = max(0.0, 1.0 - elapsed / duration)
        return tau

    def set_contract_duration(self, symbol: str, duration_seconds: float) -> None:
        """Set the estimated duration for a contract.

        Call this when contract metadata (creation date, resolution date)
        becomes available, for more accurate tau estimation.
        """
        self._estimated_duration[symbol] = max(1.0, duration_seconds)
