"""Tests for KalshiWebSocketDataSource — message handling only, no live socket."""

from decimal import Decimal

import pytest

from oracle3.data.live.kalshi_ws_data_source import KalshiWebSocketDataSource
from oracle3.events.events import OrderBookEvent
from oracle3.ticker.ticker import KalshiTicker


@pytest.fixture
def data_source(tmp_path):
    key_path = tmp_path / 'fake_key.pem'
    key_path.write_text('not a real key')
    # Bypass the real RSA key loading — we only exercise message handling.
    ds = object.__new__(KalshiWebSocketDataSource)
    ds.ws_url = 'wss://example.invalid/trade-api/ws/v2'
    import asyncio

    ds.event_queue = asyncio.Queue(maxsize=2000)
    ds._ws_task = None
    ds._desired_tickers = set()
    ds._market_meta = {}
    ds._books = {}
    ds._book_seq = {}
    ds._sids = {}
    ds._next_cmd_id = 1
    ds._control_queue = asyncio.Queue()
    ds.connection_state = 'disconnected'
    ds.last_message_at = None
    ds.last_error = ''
    ds.reconnect_attempt = 0
    ds.next_reconnect_at = None
    return ds


def _drain(ds) -> list[OrderBookEvent]:
    events = []
    while not ds.event_queue.empty():
        events.append(ds.event_queue.get_nowait())
    return events


def test_watch_token_registers_metadata(data_source):
    data_source.watch_token('KXNBA-1', name='Lakers vs Celtics', event_ticker='KXNBA')
    assert 'KXNBA-1' in data_source._desired_tickers
    assert data_source._market_meta['KXNBA-1'] == ('Lakers vs Celtics', 'KXNBA', '')


def test_subscribed_ack_records_sid(data_source):
    data_source._handle_message({
        'type': 'subscribed',
        'msg': {'channel': 'orderbook_delta', 'sid': 7},
    })
    assert data_source._sids['orderbook_delta'] == 7


def test_snapshot_then_delta_emits_top_of_book(data_source):
    data_source.watch_token('KXNBA-1', name='Lakers vs Celtics')

    data_source._handle_message({
        'type': 'orderbook_snapshot',
        'sid': 2,
        'seq': 1,
        'msg': {
            'market_ticker': 'KXNBA-1',
            'yes_dollars_fp': [['0.4500', '300.00']],
            'no_dollars_fp': [['0.4400', '250.00']],
        },
    })
    events = _drain(data_source)
    assert len(events) == 2
    assert all(isinstance(e, OrderBookEvent) for e in events)
    assert all(isinstance(e.ticker, KalshiTicker) for e in events)
    bid_event = next(e for e in events if e.side == 'bid')
    ask_event = next(e for e in events if e.side == 'ask')
    assert bid_event.price == Decimal('0.45')
    # best_yes_ask = 1 - best_no_bid = 1 - 0.44 = 0.56
    assert ask_event.price == Decimal('0.56')

    # A valid delta (seq 1 -> 2) raising the yes bid should update top-of-book.
    data_source._handle_message({
        'type': 'orderbook_delta',
        'sid': 2,
        'seq': 2,
        'msg': {
            'market_ticker': 'KXNBA-1',
            'price_dollars': '0.47',
            'delta_fp': '50.00',
            'side': 'yes',
        },
    })
    events = _drain(data_source)
    bid_event = next(e for e in events if e.side == 'bid')
    assert bid_event.price == Decimal('0.47')


def test_delta_removes_level_when_qty_hits_zero(data_source):
    data_source._handle_message({
        'type': 'orderbook_snapshot',
        'seq': 1,
        'msg': {
            'market_ticker': 'KXNBA-1',
            'yes_dollars_fp': [['0.45', '100.00']],
            'no_dollars_fp': [['0.44', '100.00']],
        },
    })
    _drain(data_source)

    data_source._handle_message({
        'type': 'orderbook_delta',
        'seq': 2,
        'msg': {
            'market_ticker': 'KXNBA-1',
            'price_dollars': '0.45',
            'delta_fp': '-100.00',
            'side': 'yes',
        },
    })
    _drain(data_source)
    book = data_source._books['KXNBA-1']
    assert book.best_yes_bid() is None
    assert Decimal('0.45') not in book.yes


def test_seq_gap_triggers_resync(data_source):
    data_source._handle_message({
        'type': 'subscribed',
        'msg': {'channel': 'orderbook_delta', 'sid': 5},
    })
    data_source._handle_message({
        'type': 'orderbook_snapshot',
        'seq': 1,
        'msg': {
            'market_ticker': 'KXNBA-1',
            'yes_dollars_fp': [['0.45', '100.00']],
            'no_dollars_fp': [],
        },
    })
    _drain(data_source)

    # Jump straight to seq 5 (gap) instead of the expected seq 2.
    data_source._handle_message({
        'type': 'orderbook_delta',
        'seq': 5,
        'msg': {
            'market_ticker': 'KXNBA-1',
            'price_dollars': '0.46',
            'delta_fp': '10.00',
            'side': 'yes',
        },
    })
    # The gapped delta must not be applied to the local book...
    assert data_source._book_seq['KXNBA-1'] == 1
    assert Decimal('0.46') not in data_source._books['KXNBA-1'].yes
    # ...and a get_snapshot resync request must be queued.
    assert not data_source._control_queue.empty()
    queued = data_source._control_queue.get_nowait()
    assert queued['params']['action'] == 'get_snapshot'
    assert queued['params']['market_tickers'] == ['KXNBA-1']


def test_ticker_message_emits_bid_and_ask(data_source):
    data_source.watch_token('KXNBA-1', name='Lakers vs Celtics')
    data_source._handle_message({
        'type': 'ticker',
        'msg': {
            'market_ticker': 'KXNBA-1',
            'yes_bid_dollars': '0.48',
            'yes_ask_dollars': '0.52',
        },
    })
    events = _drain(data_source)
    assert len(events) == 2
    prices = {e.side: e.price for e in events}
    assert prices['bid'] == Decimal('0.48')
    assert prices['ask'] == Decimal('0.52')


def test_error_frame_records_last_error(data_source):
    data_source._handle_message({'type': 'error', 'msg': 'bad subscription'})
    assert 'bad subscription' in data_source.last_error


@pytest.mark.asyncio
async def test_get_next_event_timeout(data_source):
    result = await data_source.get_next_event()
    assert result is None


def test_watch_token_after_subscribed_queues_add_markets(data_source):
    data_source._sids['orderbook_delta'] = 3
    data_source.watch_token('KXNBA-2', name='Bulls vs Heat')
    assert not data_source._control_queue.empty()
    queued = data_source._control_queue.get_nowait()
    assert queued['params']['action'] == 'add_markets'
    assert queued['params']['market_tickers'] == ['KXNBA-2']
