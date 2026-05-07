"""Fair value divergence strategy — model-driven alpha from risk premium decomposition.

Uses the Wang Transform pricing model from Yang (2026) to decompose market
prices into physical probability and risk premium, then trades when the
observed premium deviates from the model's expectation.

The key empirical finding: prediction market prices embed a systematic
risk premium (lambda ≈ 0.17 on Polymarket, 0.19 on Kalshi). The premium
varies predictably with contract characteristics:

    lambda_i = 0.259 - 0.072*ln(1+Volume) + 0.143*ln(1+Duration)
               - 0.477*|price - 0.5|

Critical alpha insight from Yang (2026): very-high-volume markets
(>$10K) have lambda ≈ 0 — the premium is already competed away.
The alpha lives in medium-liquidity markets ($500-$10K) where
lambda = 0.25-0.35. This strategy targets that sweet spot.

Position sizing uses model-informed Kelly criterion from the
pricing Greeks, ensuring positions scale with edge quality.

Reference:
    Yang, Y. (2026). "Pricing Prediction Markets." Working Paper, UIUC.

Usage::

    oracle3 engine run \\
      --exchange polymarket --mode paper \\
      --strategy-ref oracle3.strategy.contrib.fair_value_strategy:FairValueStrategy \\
      --strategy-kwargs-json '{"min_edge": 0.01, "category": "crypto"}'
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from decimal import Decimal
from typing import ClassVar

from oracle3.events.events import Event, OrderBookEvent, PriceChangeEvent
from oracle3.pricing.calibrator import OnlineCalibrator
from oracle3.pricing.contract_scorer import ContractScorer
from oracle3.pricing.fair_value import FairValueEstimate, FairValueEstimator
from oracle3.pricing.greeks import kelly_fraction as model_kelly
from oracle3.strategy.quant_strategy import QuantStrategy
from oracle3.ticker.ticker import Ticker
from oracle3.trader.trader import Trader
from oracle3.trader.types import TradeSide

logger = logging.getLogger(__name__)

_FEE_PER_SIDE = Decimal('0.005')

_FLAT = 'flat'
_LONG_YES = 'long_yes'
_LONG_NO = 'long_no'


class FairValueStrategy(QuantStrategy):
    """Trade Wang-model fair value divergence with volume-tier targeting.

    Parameters
    ----------
    min_edge:
        Minimum net edge after fees to enter (default 0.008 = 0.8 cents).
    exit_edge:
        Exit when edge falls below this (default 0.002).
    max_kelly:
        Maximum Kelly fraction (default 0.12, conservative).
    base_trade_size:
        Base dollar amount per trade (default 25).
    max_hold_seconds:
        Maximum hold time (default 3600 = 1 hour).
    min_confidence:
        Minimum model confidence (default 0.20).
    cooldown_seconds:
        Minimum seconds between trades on same ticker (default 120).
    fee_rate:
        Per-side fee rate (default 0.005).
    category:
        Contract category for calibration (default 'default').
    platform:
        Platform for calibration priors (default 'polymarket').
    skip_very_high_volume:
        Skip >$10K volume markets where lambda≈0 (default True).
    """

    name: ClassVar[str] = 'fair_value_divergence'
    version: ClassVar[str] = '2.0.0'
    author: ClassVar[str] = 'oracle3'

    def __init__(
        self,
        min_edge: float = 0.008,
        exit_edge: float = 0.002,
        max_kelly: float = 0.12,
        base_trade_size: Decimal = Decimal('25'),
        max_hold_seconds: int = 3600,
        min_confidence: float = 0.20,
        cooldown_seconds: int = 120,
        fee_rate: Decimal = _FEE_PER_SIDE,
        category: str = 'default',
        platform: str = 'polymarket',
        skip_very_high_volume: bool = True,
    ) -> None:
        super().__init__()
        self._min_edge = min_edge
        self._exit_edge = exit_edge
        self._max_kelly = max_kelly
        self._base_size = base_trade_size
        self._max_hold = max_hold_seconds
        self._min_conf = min_confidence
        self._cooldown = cooldown_seconds
        self._fee_rate = fee_rate
        self._category = category
        self._skip_vhv = skip_very_high_volume

        self._calibrator = OnlineCalibrator(platform=platform)
        self._scorer = ContractScorer()
        self._estimator = FairValueEstimator(
            calibrator=self._calibrator,
            scorer=self._scorer,
            use_paper_coefficients=True,
        )

        self._positions: dict[str, str] = defaultdict(lambda: _FLAT)
        self._entry_times: dict[str, float] = {}
        self._entry_estimates: dict[str, FairValueEstimate] = {}
        self._last_trade: dict[str, float] = defaultdict(float)

    async def process_event(self, event: Event, trader: Trader) -> None:
        if self.is_paused():
            return
        if not isinstance(event, (PriceChangeEvent, OrderBookEvent)):
            return

        ticker = event.ticker
        symbol = ticker.symbol
        price = self._get_price(event, trader, ticker)
        if price is None or price <= 0.02 or price >= 0.98:
            return

        now = time.time()

        # Get fair value estimate with covariates
        bid = trader.market_data.get_best_bid(ticker)
        ask = trader.market_data.get_best_ask(ticker)
        spread = float(ask.price - bid.price) if bid and ask else None

        estimate = self._estimator.estimate(
            market_price=price,
            category=self._category,
            spread=spread,
        )

        state = self._positions[symbol]

        # Exit logic
        if state != _FLAT:
            if self._should_exit(symbol, estimate, now):
                await self._exit(symbol, ticker, trader, estimate, now)
                return

        # Entry logic
        if state == _FLAT:
            if now - self._last_trade[symbol] < self._cooldown:
                return
            if estimate.confidence < self._min_conf:
                return
            if self._skip_vhv and estimate.volume_tier == 'very_high':
                return
            edge = abs(estimate.risk_premium)
            net_edge = edge - 2 * float(self._fee_rate)
            if net_edge >= self._min_edge:
                await self._enter(symbol, ticker, trader, estimate, now)

    async def _enter(
        self, symbol: str, ticker: Ticker, trader: Trader,
        estimate: FairValueEstimate, now: float,
    ) -> None:
        # Use model Kelly for sizing
        kelly = model_kelly(estimate.market_price, estimate.lambda_adjusted, float(self._fee_rate))
        kelly_capped = min(abs(kelly), self._max_kelly)
        size = self._base_size * Decimal(str(max(kelly_capped, 0.01)))
        size = min(size, Decimal('100'))

        if estimate.risk_premium > 0:
            side = TradeSide.BUY
            no_ticker = getattr(ticker, 'get_no_ticker', lambda: None)()
            trade_ticker = no_ticker or ticker
            action = 'BUY_NO'
            new_state = _LONG_NO
        else:
            side = TradeSide.BUY
            trade_ticker = ticker
            action = 'BUY_YES'
            new_state = _LONG_YES

        price_dec = Decimal(str(estimate.market_price))
        qty = size / max(price_dec, Decimal('0.01'))
        result = await trader.place_order(side, trade_ticker, price_dec, qty)
        executed = result.executed

        if executed:
            self._positions[symbol] = new_state
            self._entry_times[symbol] = now
            self._entry_estimates[symbol] = estimate
            self._last_trade[symbol] = now

        self.record_decision(
            ticker_name=symbol, action=action, executed=executed,
            confidence=estimate.confidence,
            reasoning=(
                f'fv={estimate.fair_value:.3f} mkt={estimate.market_price:.3f} '
                f'prem={estimate.risk_premium:.4f} λ={estimate.lambda_adjusted:.3f} '
                f'kelly={kelly_capped:.3f} tier={estimate.volume_tier}'
            ),
            signal_values={
                'fair_value': estimate.fair_value,
                'market_price': estimate.market_price,
                'risk_premium': estimate.risk_premium,
                'lambda': estimate.lambda_adjusted,
                'kelly': kelly_capped,
                'confidence': estimate.confidence,
            },
        )

    async def _exit(
        self, symbol: str, ticker: Ticker, trader: Trader,
        estimate: FairValueEstimate, now: float,
    ) -> None:
        state = self._positions[symbol]
        if state == _LONG_NO:
            no_ticker = getattr(ticker, 'get_no_ticker', lambda: None)()
            trade_ticker = no_ticker or ticker
        else:
            trade_ticker = ticker

        price_dec = Decimal(str(estimate.market_price))
        qty = self._base_size / max(price_dec, Decimal('0.01'))
        result = await trader.place_order(TradeSide.SELL, trade_ticker, price_dec, qty)
        executed = result.executed

        if executed:
            self._positions[symbol] = _FLAT
            self._entry_times.pop(symbol, None)
            self._entry_estimates.pop(symbol, None)
            self._last_trade[symbol] = now

        self.record_decision(
            ticker_name=symbol, action=f'CLOSE_{state.upper()}', executed=executed,
            confidence=estimate.confidence,
            reasoning=f'Exit: prem={estimate.risk_premium:.4f}',
            signal_values={'risk_premium': estimate.risk_premium},
        )

    def _should_exit(self, symbol: str, estimate: FairValueEstimate, now: float) -> bool:
        if abs(estimate.risk_premium) < self._exit_edge:
            return True
        if now - self._entry_times.get(symbol, now) > self._max_hold:
            return True
        entry_est = self._entry_estimates.get(symbol)
        if entry_est:
            entry_sign = 1 if entry_est.risk_premium > 0 else -1
            curr_sign = 1 if estimate.risk_premium > 0 else -1
            if entry_sign != curr_sign:
                return True
        return False

    def _get_price(self, event: Event, trader: Trader, ticker: Ticker) -> float | None:
        if isinstance(event, PriceChangeEvent):
            return float(event.price)
        bid = trader.market_data.get_best_bid(ticker)
        ask = trader.market_data.get_best_ask(ticker)
        if bid and ask:
            return (float(bid.price) + float(ask.price)) / 2.0
        return None
