"""Tests for PolymarketWebSocketDataSource — message handling only, no live socket."""

import asyncio
import time
from decimal import Decimal

import pytest

from oracle3.data.live.polymarket_ws_data_source import PolymarketWebSocketDataSource
from oracle3.events.events import OrderBookEvent, PriceChangeEvent
from oracle3.ticker.ticker import PolyMarketTicker


@pytest.fixture
def data_source() -> PolymarketWebSocketDataSource:
    return PolymarketWebSocketDataSource()


def _drain(ds):
    events = []
    while not ds.event_queue.empty():
        events.append(ds.event_queue.get_nowait())
    return events


def test_watch_token_registers_metadata(data_source):
    data_source.watch_token('tok1', name='Team A', market_id='m1', event_id='e1')
    assert 'tok1' in data_source._desired_assets
    assert data_source._asset_meta['tok1'] == ('Team A', 'm1', 'e1', '')


def test_watch_token_while_connected_queues_subscribe_op(data_source):
    data_source._connected = True
    data_source.watch_token('tok2', name='Team B')
    assert not data_source._control_queue.empty()
    queued = data_source._control_queue.get_nowait()
    assert queued == {'assets_ids': ['tok2'], 'operation': 'subscribe'}


def test_book_snapshot_emits_events(data_source):
    data_source.watch_token('tok1', name='Team A')
    data_source._handle_message({
        'event_type': 'book',
        'asset_id': 'tok1',
        'market': '0xabc',
        'bids': [{'price': '.48', 'size': '30'}],
        'asks': [{'price': '.52', 'size': '25'}],
    })
    events = _drain(data_source)
    assert len(events) == 2
    assert all(isinstance(e, OrderBookEvent) for e in events)
    assert all(isinstance(e.ticker, PolyMarketTicker) for e in events)


def test_price_change_buy_side_is_bid(data_source):
    data_source._handle_message({
        'event_type': 'price_change',
        'market': '0xabc',
        'price_changes': [{
            'asset_id': 'tok1',
            'price': '0.5',
            'size': '200',
            'side': 'BUY',
            'best_bid': '0.5',
            'best_ask': '1',
        }],
    })
    events = _drain(data_source)
    assert len(events) == 1
    assert events[0].side == 'bid'
    assert events[0].size == Decimal('200')


def test_price_change_sell_side_is_ask(data_source):
    data_source._handle_message({
        'event_type': 'price_change',
        'price_changes': [{
            'asset_id': 'tok1', 'price': '0.6', 'size': '50', 'side': 'SELL',
        }],
    })
    events = _drain(data_source)
    assert events[0].side == 'ask'


def test_price_change_zero_size_removes_level(data_source):
    data_source._handle_message({
        'event_type': 'price_change',
        'price_changes': [{
            'asset_id': 'tok1', 'price': '0.6', 'size': '50', 'side': 'SELL',
        }],
    })
    _drain(data_source)
    data_source._handle_message({
        'event_type': 'price_change',
        'price_changes': [{
            'asset_id': 'tok1', 'price': '0.6', 'size': '0', 'side': 'SELL',
        }],
    })
    events = _drain(data_source)
    assert len(events) == 1
    assert events[0].size == Decimal('0')
    assert events[0].size_delta == Decimal('-50')


def test_last_trade_price_emits_price_change_event(data_source):
    data_source._handle_message({
        'event_type': 'last_trade_price',
        'asset_id': 'tok1',
        'price': '0.63',
        'side': 'BUY',
        'size': '10',
    })
    events = _drain(data_source)
    assert len(events) == 1
    assert isinstance(events[0], PriceChangeEvent)
    assert events[0].price == Decimal('0.63')


def test_batched_array_message_dispatches_each_item(data_source):
    data_source._handle_message([
        {
            'event_type': 'last_trade_price',
            'asset_id': 'tok1',
            'price': '0.5',
        },
        {
            'event_type': 'last_trade_price',
            'asset_id': 'tok2',
            'price': '0.6',
        },
    ])
    events = _drain(data_source)
    assert len(events) == 2


@pytest.mark.asyncio
async def test_watchdog_triggers_reconnect_when_stale(data_source):
    class FakeWS:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    data_source.last_message_at = time.monotonic() - 200.0

    ws = FakeWS()
    await data_source._watchdog(ws, check_interval=0.01)
    assert ws.closed


@pytest.mark.asyncio
async def test_watchdog_does_not_close_when_fresh(data_source):
    class FakeWS:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    data_source.last_message_at = time.monotonic()

    ws = FakeWS()
    task = asyncio.ensure_future(data_source._watchdog(ws, check_interval=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not ws.closed


@pytest.mark.asyncio
async def test_get_next_event_timeout(data_source):
    result = await data_source.get_next_event()
    assert result is None
