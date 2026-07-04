"""Tests for the shared order-book snapshot diffing helper."""

from decimal import Decimal

from oracle3.data.live._orderbook_diff import diff_order_book_to_events
from oracle3.ticker.ticker import PolyMarketTicker


def _ticker() -> PolyMarketTicker:
    return PolyMarketTicker(symbol='tok1', name='Test', token_id='tok1')


def test_first_snapshot_emits_all_levels():
    state: dict[str, Decimal] = {}
    events = diff_order_book_to_events(
        'tok1', _ticker(),
        bid_levels=[('0.48', '30')],
        ask_levels=[('0.52', '25')],
        state=state,
    )
    assert len(events) == 2
    sides = {e.side: e.price for e in events}
    assert sides['bid'] == Decimal('0.48')
    assert sides['ask'] == Decimal('0.52')


def test_unchanged_levels_emit_nothing():
    state: dict[str, Decimal] = {}
    diff_order_book_to_events(
        'tok1', _ticker(), [('0.48', '30')], [('0.52', '25')], state,
    )
    events = diff_order_book_to_events(
        'tok1', _ticker(), [('0.48', '30')], [('0.52', '25')], state,
    )
    assert events == []


def test_changed_size_emits_delta():
    state: dict[str, Decimal] = {}
    diff_order_book_to_events(
        'tok1', _ticker(), [('0.48', '30')], [], state,
    )
    events = diff_order_book_to_events(
        'tok1', _ticker(), [('0.48', '50')], [], state,
    )
    assert len(events) == 1
    assert events[0].size == Decimal('50')
    assert events[0].size_delta == Decimal('20')


def test_disappeared_level_emits_zero_size_removal():
    state: dict[str, Decimal] = {}
    diff_order_book_to_events(
        'tok1', _ticker(), [('0.48', '30')], [], state,
    )
    events = diff_order_book_to_events(
        'tok1', _ticker(), [], [], state,
    )
    assert len(events) == 1
    assert events[0].price == Decimal('0.48')
    assert events[0].size == Decimal('0')
    assert events[0].size_delta == Decimal('-30')
    assert 'tok1:0.48:bid' not in state


def test_separate_tokens_do_not_interfere():
    state: dict[str, Decimal] = {}
    diff_order_book_to_events('tokA', _ticker(), [('0.5', '10')], [], state)
    diff_order_book_to_events('tokB', _ticker(), [('0.5', '10')], [], state)
    events = diff_order_book_to_events('tokA', _ticker(), [], [], state)
    # Only tokA's level should be reported as removed; tokB's must survive.
    assert len(events) == 1
    assert 'tokB:0.5:bid' in state
