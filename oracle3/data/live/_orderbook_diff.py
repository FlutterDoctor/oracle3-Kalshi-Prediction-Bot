"""Shared order-book-snapshot-to-event diffing.

REST-polled data sources receive a full order-book snapshot on every poll
and must diff it against the previously seen state to know what actually
changed; the same problem shows up when a websocket source's periodic
full-book message (e.g. Polymarket's ``book`` event) arrives. Extracted
here so both kinds of source share one implementation instead of two
copies of the same "diff snapshot, emit changed/removed levels" logic
(see ``LivePolyMarketDataSource._process_refresh_result``).
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from ...events.events import OrderBookEvent
from ...ticker.ticker import Ticker


def _to_decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def diff_order_book_to_events(
    token_id: str,
    ticker: Ticker,
    bid_levels: list[tuple[str, str]],
    ask_levels: list[tuple[str, str]],
    state: dict[str, Decimal],
    clear_stale: bool = True,
) -> list[OrderBookEvent]:
    """Diff a full order-book snapshot against ``state`` and emit deltas.

    ``bid_levels``/``ask_levels`` are ``(price_str, size_str)`` pairs.
    ``state`` is mutated in place, keyed ``f'{token_id}:{price_str}:{side}'``,
    so repeated calls only emit events for genuinely changed or removed
    price levels rather than re-emitting the whole book every time.
    """
    events: list[OrderBookEvent] = []

    def _process_side(levels: list[tuple[str, str]], side: str) -> set[str]:
        seen_keys: set[str] = set()
        for price_str, size_str in levels:
            price = _to_decimal(price_str)
            size = _to_decimal(size_str)
            if price is None or size is None:
                continue
            key = f'{token_id}:{price_str}:{side}'
            seen_keys.add(key)
            prev_size = state.get(key, Decimal('0'))
            state[key] = size
            if size != prev_size:
                events.append(OrderBookEvent(
                    ticker=ticker,
                    price=price,
                    size=size,
                    size_delta=size - prev_size,
                    side=side,
                ))
        return seen_keys

    seen_bid_keys = _process_side(bid_levels, 'bid')
    seen_ask_keys = _process_side(ask_levels, 'ask')

    if clear_stale:
        events.extend(_clear_stale_levels(token_id, ticker, state, seen_bid_keys, 'bid'))
        events.extend(_clear_stale_levels(token_id, ticker, state, seen_ask_keys, 'ask'))

    return events


def _clear_stale_levels(
    token_id: str,
    ticker: Ticker,
    state: dict[str, Decimal],
    seen_keys: set[str],
    side: str,
) -> list[OrderBookEvent]:
    events: list[OrderBookEvent] = []
    prefix = f'{token_id}:'
    suffix = f':{side}'
    old_keys = {k for k in state if k.startswith(prefix) and k.endswith(suffix)}
    for stale_key in old_keys - seen_keys:
        price_str = stale_key[len(prefix) : -len(suffix)]
        prev_size = state.pop(stale_key, Decimal('0'))
        if prev_size > 0:
            price = _to_decimal(price_str)
            if price is not None:
                events.append(OrderBookEvent(
                    ticker=ticker,
                    price=price,
                    size=Decimal('0'),
                    size_delta=-prev_size,
                    side=side,
                ))
    return events
