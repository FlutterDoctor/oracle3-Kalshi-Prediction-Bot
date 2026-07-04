"""Real exchange trading-fee formulas.

Kalshi's fee is nonlinear in price (near zero at the extremes, peaks at
price=0.50), so a flat fee-rate approximation systematically mis-estimates
cost in exactly the favorite-longshot region this bot targets. These
functions implement Kalshi's documented fee schedule instead of a guessed
flat rate.

Reference: fee = ceil(fee_multiplier * contracts * price * (1 - price) * 100)
cents, where price is in dollars (0-1). Standard fee_multiplier is 0.07;
a handful of Kalshi series (e.g. weekly S&P/Nasdaq range markets) use a
different multiplier, so it is exposed as an override rather than hardcoded.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import ROUND_CEILING, Decimal
from typing import TYPE_CHECKING

from oracle3.ticker.ticker import KalshiTicker

if TYPE_CHECKING:
    from oracle3.ticker.ticker import Ticker

KALSHI_STANDARD_FEE_MULTIPLIER = Decimal('0.07')

# Polymarket's CLOB currently charges ~0 maker/taker fee on most markets.
POLYMARKET_FEE_RATE = Decimal('0')


def kalshi_fee_per_contract(
    price: float, fee_multiplier: float = float(KALSHI_STANDARD_FEE_MULTIPLIER)
) -> float:
    """Kalshi's per-contract fee in dollars, before cent rounding.

    Nonlinear: `fee_multiplier * price * (1 - price)`, e.g. ~$0.0033/contract
    at price=0.05 vs ~$0.0175/contract at price=0.50.
    """
    return fee_multiplier * price * (1.0 - price)


def kalshi_round_trip_fee(
    price: float, fee_multiplier: float = float(KALSHI_STANDARD_FEE_MULTIPLIER)
) -> float:
    """Fee cost (as a fraction of $1 notional) to open AND close one contract."""
    return 2.0 * kalshi_fee_per_contract(price, fee_multiplier)


def round_trip_fee_for_ticker(
    ticker: Ticker,
    price: float,
    fee_multiplier: float = float(KALSHI_STANDARD_FEE_MULTIPLIER),
) -> float:
    """Real round-trip fee for one contract, dispatched by exchange.

    Kalshi's fee is nonlinear in price; Polymarket currently charges ~0.
    """
    if isinstance(ticker, KalshiTicker):
        return kalshi_round_trip_fee(price, fee_multiplier)
    return float(POLYMARKET_FEE_RATE) * 2


def round_trip_fee_for_legs(
    legs: Iterable[tuple[Ticker, float]],
    fee_multiplier: float = float(KALSHI_STANDARD_FEE_MULTIPLIER),
) -> float:
    """Sum of real round-trip fees across a multi-leg trade's (ticker, price) legs."""
    return sum(
        round_trip_fee_for_ticker(ticker, price, fee_multiplier)
        for ticker, price in legs
    )


def kalshi_order_fee(
    contracts: int,
    price: Decimal,
    fee_multiplier: Decimal = KALSHI_STANDARD_FEE_MULTIPLIER,
) -> Decimal:
    """Kalshi's actual billed fee for an order, rounded up to the next cent."""
    raw_dollars = fee_multiplier * Decimal(contracts) * price * (Decimal('1') - price)
    cents = (raw_dollars * Decimal('100')).to_integral_value(rounding=ROUND_CEILING)
    return cents / Decimal('100')
