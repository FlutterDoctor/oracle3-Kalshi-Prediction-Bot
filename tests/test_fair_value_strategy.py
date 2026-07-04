"""Tests for FairValueStrategy and PremiumDecayStrategy.

Uses mock traders and events to verify strategy logic without
requiring live market connections.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from oracle3.events.events import PriceChangeEvent
from oracle3.strategy.contrib.fair_value_strategy import FairValueStrategy
from oracle3.strategy.contrib.premium_decay_strategy import PremiumDecayStrategy
from oracle3.ticker.ticker import Ticker


def _make_ticker(symbol: str = 'TEST_YES') -> Ticker:
    """Create a minimal mock ticker."""
    ticker = MagicMock(spec=Ticker)
    ticker.symbol = symbol
    ticker.name = f'Test Market {symbol}'
    ticker.get_no_ticker = MagicMock(return_value=None)
    return ticker


def _make_event(ticker: Ticker, price: float = 0.55) -> PriceChangeEvent:
    """Create a price change event."""
    return PriceChangeEvent(ticker=ticker, price=Decimal(str(price)))


def _make_trader(bid: float = 0.54, ask: float = 0.56) -> MagicMock:
    """Create a mock trader with basic market data."""
    trader = MagicMock()
    trader.place_order = AsyncMock(return_value=True)

    # Mock market data
    market_data = MagicMock()
    bid_obj = MagicMock()
    bid_obj.price = Decimal(str(bid))
    bid_obj.size = Decimal('100')
    ask_obj = MagicMock()
    ask_obj.price = Decimal(str(ask))
    ask_obj.size = Decimal('100')
    market_data.get_best_bid = MagicMock(return_value=bid_obj)
    market_data.get_best_ask = MagicMock(return_value=ask_obj)
    trader.market_data = market_data

    return trader


class TestFairValueStrategy:
    """Test the fair value divergence strategy."""

    def test_initialization(self):
        s = FairValueStrategy(min_edge=0.01)
        assert s.name == 'fair_value_divergence'
        assert s._min_edge == 0.01

    def test_status_metrics_json_serializable_before_any_calibration(self):
        """Regression: OnlineCalibrator.report().staleness_hours is
        float('inf') until the calibrator has seen a resolved contract,
        and inf/nan aren't valid JSON — get_status_metrics() must normalize
        it (to None) so the dashboard's /api/games endpoint doesn't 500."""
        s = FairValueStrategy()
        metrics = s.get_status_metrics()
        assert metrics['staleness_hours'] is None
        for value in metrics.values():
            if isinstance(value, float):
                assert math.isfinite(value)
        json.dumps(metrics)  # must not raise

    @pytest.mark.asyncio
    async def test_ignores_extreme_prices(self):
        """Prices near 0 or 1 should be skipped."""
        s = FairValueStrategy()
        ticker = _make_ticker()
        trader = _make_trader()

        event_low = _make_event(ticker, price=0.005)
        await s.process_event(event_low, trader)
        trader.place_order.assert_not_called()

        event_high = _make_event(ticker, price=0.995)
        await s.process_event(event_high, trader)
        trader.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_records_decisions(self):
        """Strategy should record decisions even on HOLD."""
        s = FairValueStrategy(min_edge=100.0)  # very high threshold
        ticker = _make_ticker()
        trader = _make_trader()
        event = _make_event(ticker, price=0.55)

        await s.process_event(event, trader)
        # With very high threshold, should hold

    @pytest.mark.asyncio
    async def test_paused_does_nothing(self):
        s = FairValueStrategy()
        s.set_paused(True)
        ticker = _make_ticker()
        trader = _make_trader()
        event = _make_event(ticker, price=0.55)

        await s.process_event(event, trader)
        trader.place_order.assert_not_called()

    def test_supports_auto_tune(self):
        assert FairValueStrategy.supports_auto_tune() is True

    def test_param_schema(self):
        schema = FairValueStrategy.param_schema()
        assert 'min_edge' in schema
        assert 'max_kelly' in schema
        assert 'base_trade_size' in schema


class TestPremiumDecayStrategy:
    """Test the premium decay strategy."""

    def test_initialization(self):
        s = PremiumDecayStrategy(min_premium=0.05)
        assert s.name == 'premium_decay'
        assert s._min_premium == 0.05

    @pytest.mark.asyncio
    async def test_ignores_extreme_prices(self):
        s = PremiumDecayStrategy()
        ticker = _make_ticker()
        trader = _make_trader()

        event = _make_event(ticker, price=0.005)
        await s.process_event(event, trader)
        trader.place_order.assert_not_called()

    def test_set_contract_duration(self):
        s = PremiumDecayStrategy()
        s.set_contract_duration('TEST', 86400 * 30)
        assert s._estimated_duration['TEST'] == 86400 * 30

    @pytest.mark.asyncio
    async def test_paused_does_nothing(self):
        s = PremiumDecayStrategy()
        s.set_paused(True)
        ticker = _make_ticker()
        trader = _make_trader()
        event = _make_event(ticker, price=0.55)

        await s.process_event(event, trader)
        trader.place_order.assert_not_called()

    def test_param_schema(self):
        schema = PremiumDecayStrategy.param_schema()
        assert 'min_premium' in schema
        assert 'min_entry_tau' in schema
        assert 'trade_size' in schema
