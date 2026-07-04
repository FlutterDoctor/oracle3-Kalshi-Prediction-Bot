"""Kalshi WebSocket streaming data source for real-time market data.

Connects to Kalshi's authenticated market-data websocket
(``/trade-api/ws/v2``), subscribes to ``orderbook_delta`` and ``ticker``
channels for a caller-supplied set of market tickers, and emits
``OrderBookEvent``\\ s built from either the maintained local order book
(snapshot + delta) or the cheaper top-of-book ``ticker`` frames.

Reuses the same RSA-PSS request signing ``kalshi_python``'s REST client
uses (``KalshiAuth.create_auth_headers``), just applied to the fixed
websocket handshake path instead of a REST endpoint path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

import websockets

from ...events.events import Event, OrderBookEvent
from ...ticker.ticker import KalshiTicker
from ..data_source import DataSource

logger = logging.getLogger(__name__)

_WS_URL = 'wss://api.elections.kalshi.com/trade-api/ws/v2'
_WS_SIGN_PATH = '/trade-api/ws/v2'
_CHANNELS = ('orderbook_delta', 'ticker')
_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 300.0
_TOP_OF_BOOK_SIZE = Decimal('100')


def _to_decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


@dataclass
class _LocalBook:
    """Per-market resting-order state built from snapshot + delta frames.

    Kalshi's single order book combines resting YES buy orders (``yes``)
    with resting NO buy orders (``no``); a NO bid at price ``q`` is
    equivalent to a YES ask at ``1 - q``.
    """

    yes: dict[Decimal, Decimal] = field(default_factory=dict)
    no: dict[Decimal, Decimal] = field(default_factory=dict)

    def best_yes_bid(self) -> Decimal | None:
        levels = [p for p, q in self.yes.items() if q > 0]
        return max(levels) if levels else None

    def best_yes_ask(self) -> Decimal | None:
        levels = [p for p, q in self.no.items() if q > 0]
        return (Decimal('1') - max(levels)) if levels else None

    def apply_snapshot(
        self,
        yes_levels: list[tuple[Decimal, Decimal]],
        no_levels: list[tuple[Decimal, Decimal]],
    ) -> None:
        self.yes = dict(yes_levels)
        self.no = dict(no_levels)

    def apply_delta(self, side: str, price: Decimal, delta: Decimal) -> None:
        book = self.yes if side == 'yes' else self.no
        new_qty = book.get(price, Decimal('0')) + delta
        if new_qty <= 0:
            book.pop(price, None)
        else:
            book[price] = new_qty


class KalshiWebSocketDataSource(DataSource):
    """Streams real-time Kalshi market data via the authenticated ws/v2 feed."""

    def __init__(
        self,
        api_key_id: str | None = None,
        private_key_path: str | None = None,
        ws_url: str = _WS_URL,
    ) -> None:
        key_id = api_key_id or os.environ.get('KALSHI_API_KEY_ID')
        pk_path = private_key_path or os.environ.get('KALSHI_PRIVATE_KEY_PATH')
        if not key_id or not pk_path:
            raise ValueError(
                'KalshiWebSocketDataSource requires KALSHI_API_KEY_ID and '
                'KALSHI_PRIVATE_KEY_PATH (env vars or constructor args).'
            )
        from kalshi_python.api_client import KalshiAuth

        self._auth = KalshiAuth(key_id, pk_path)
        self.ws_url = ws_url

        self.event_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=2000)
        self._ws_task: asyncio.Task | None = None

        self._desired_tickers: set[str] = set()
        self._market_meta: dict[str, tuple[str, str, str]] = {}
        self._books: dict[str, _LocalBook] = {}
        self._book_seq: dict[str, int] = {}
        self._sids: dict[str, int] = {}
        self._next_cmd_id = 1
        self._control_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Introspection state for the dashboard's live-status panel.
        self.connection_state = 'disconnected'
        self.last_message_at: float | None = None
        self.last_error = ''
        self.reconnect_attempt = 0
        self.next_reconnect_at: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_loop())

    async def stop(self) -> None:
        if self._ws_task is not None and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        self.connection_state = 'disconnected'

    async def get_next_event(self) -> Event | None:
        try:
            return await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def watch_token(
        self,
        market_ticker: str,
        name: str = '',
        event_ticker: str = '',
        series_ticker: str = '',
    ) -> None:
        """Add a market to the live subscription (called by ``SharedMarketFeed``)."""
        if not market_ticker:
            return
        self._market_meta[market_ticker] = (name, event_ticker, series_ticker)
        if market_ticker in self._desired_tickers:
            return
        self._desired_tickers.add(market_ticker)
        if self._sids:
            self._control_queue.put_nowait({
                'id': self._next_id(),
                'cmd': 'update_subscription',
                'params': {
                    'sids': list(self._sids.values()),
                    'market_tickers': [market_ticker],
                    'action': 'add_markets',
                },
            })

    def _next_id(self) -> int:
        cmd_id = self._next_cmd_id
        self._next_cmd_id += 1
        return cmd_id

    def _make_ticker(self, market_ticker: str) -> KalshiTicker:
        name, event_ticker, series_ticker = self._market_meta.get(
            market_ticker, ('', '', '')
        )
        return KalshiTicker(
            symbol=market_ticker,
            name=name,
            market_ticker=market_ticker,
            event_ticker=event_ticker,
            series_ticker=series_ticker,
        )

    def _enqueue(self, event: Event) -> None:
        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.debug('KalshiWS queue full, dropping event: %s', type(event).__name__)

    def _resync(self, market_ticker: str) -> None:
        """Request a fresh snapshot for one market after a sequence gap."""
        if not self._sids:
            return
        self._control_queue.put_nowait({
            'id': self._next_id(),
            'cmd': 'update_subscription',
            'params': {
                'sids': list(self._sids.values()),
                'market_tickers': [market_ticker],
                'action': 'get_snapshot',
            },
        })

    # ------------------------------------------------------------------
    # WebSocket loop with auto-reconnect
    # ------------------------------------------------------------------

    async def _ws_loop(self) -> None:
        backoff = _INITIAL_BACKOFF
        while True:
            self.connection_state = 'connecting'
            try:
                headers = self._auth.create_auth_headers('GET', _WS_SIGN_PATH)
                async with websockets.connect(
                    self.ws_url, additional_headers=headers
                ) as ws:
                    logger.info('KalshiWS connected to %s', self.ws_url)
                    self.connection_state = 'connected'
                    self.last_error = ''
                    self.reconnect_attempt = 0
                    self.next_reconnect_at = None
                    backoff = _INITIAL_BACKOFF
                    self._sids.clear()
                    self._books.clear()
                    self._book_seq.clear()

                    await self._subscribe_all(ws)
                    sender = asyncio.create_task(self._control_sender(ws))
                    try:
                        async for raw in ws:
                            self.last_message_at = time.monotonic()
                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError:
                                logger.debug('KalshiWS non-JSON message: %s', raw[:200])
                                continue
                            self._handle_message(msg)
                    finally:
                        sender.cancel()

            except asyncio.CancelledError:
                raise
            except Exception:
                self.connection_state = 'reconnecting'
                self.reconnect_attempt += 1
                self.next_reconnect_at = time.monotonic() + backoff
                logger.warning(
                    'KalshiWS connection lost (reconnect in %.0fs)',
                    backoff,
                    exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    async def _subscribe_all(self, ws) -> None:
        if not self._desired_tickers:
            return
        market_tickers = sorted(self._desired_tickers)
        await ws.send(json.dumps({
            'id': self._next_id(),
            'cmd': 'subscribe',
            'params': {'channels': list(_CHANNELS), 'market_tickers': market_tickers},
        }))

    async def _control_sender(self, ws) -> None:
        """Drains dynamically-queued subscription updates onto the live socket."""
        try:
            while True:
                msg = await self._control_queue.get()
                await ws.send(json.dumps(msg))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug('KalshiWS control send failed', exc_info=True)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _handle_message(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get('type', '')
        if msg_type == 'subscribed':
            self._handle_subscribed(msg)
        elif msg_type == 'orderbook_snapshot':
            self._handle_snapshot(msg)
        elif msg_type == 'orderbook_delta':
            self._handle_delta(msg)
        elif msg_type == 'ticker':
            self._handle_ticker(msg)
        elif msg_type == 'error':
            self.last_error = str(msg.get('msg', msg))
            logger.warning('KalshiWS error frame: %s', msg)
        else:
            logger.debug('KalshiWS unhandled message type: %s', msg_type)

    def _handle_subscribed(self, msg: dict[str, Any]) -> None:
        body = msg.get('msg', {})
        channel = body.get('channel', '')
        sid = body.get('sid')
        if channel and isinstance(sid, int):
            self._sids[channel] = sid

    def _handle_snapshot(self, msg: dict[str, Any]) -> None:
        body = msg.get('msg', {})
        market_ticker = body.get('market_ticker', '')
        if not market_ticker:
            return
        seq = msg.get('seq')
        yes_levels = self._parse_levels(body.get('yes_dollars_fp', []))
        no_levels = self._parse_levels(body.get('no_dollars_fp', []))
        book = self._books.setdefault(market_ticker, _LocalBook())
        book.apply_snapshot(yes_levels, no_levels)
        if isinstance(seq, int):
            self._book_seq[market_ticker] = seq
        self._emit_top_of_book(market_ticker, book)

    def _handle_delta(self, msg: dict[str, Any]) -> None:
        body = msg.get('msg', {})
        market_ticker = body.get('market_ticker', '')
        if not market_ticker:
            return
        seq = msg.get('seq')
        expected = self._book_seq.get(market_ticker)
        if isinstance(seq, int) and expected is not None and seq != expected + 1:
            logger.warning(
                'KalshiWS seq gap for %s: expected %s, got %s — resyncing',
                market_ticker,
                expected + 1,
                seq,
            )
            self._resync(market_ticker)
            return
        price = _to_decimal(body.get('price_dollars'))
        delta = _to_decimal(body.get('delta_fp'))
        side = body.get('side', '')
        if price is None or delta is None or side not in ('yes', 'no'):
            return
        book = self._books.setdefault(market_ticker, _LocalBook())
        book.apply_delta(side, price, delta)
        if isinstance(seq, int):
            self._book_seq[market_ticker] = seq
        self._emit_top_of_book(market_ticker, book)

    def _handle_ticker(self, msg: dict[str, Any]) -> None:
        body = msg.get('msg', msg)
        market_ticker = body.get('market_ticker', '')
        if not market_ticker:
            return
        ticker = self._make_ticker(market_ticker)
        yes_bid = _to_decimal(body.get('yes_bid_dollars'))
        yes_ask = _to_decimal(body.get('yes_ask_dollars'))
        if yes_bid is not None and Decimal('0') < yes_bid < Decimal('1'):
            self._enqueue(OrderBookEvent(
                ticker=ticker,
                price=yes_bid,
                size=_TOP_OF_BOOK_SIZE,
                size_delta=_TOP_OF_BOOK_SIZE,
                side='bid',
            ))
        if yes_ask is not None and Decimal('0') < yes_ask < Decimal('1'):
            self._enqueue(OrderBookEvent(
                ticker=ticker,
                price=yes_ask,
                size=_TOP_OF_BOOK_SIZE,
                size_delta=_TOP_OF_BOOK_SIZE,
                side='ask',
            ))

    def _emit_top_of_book(self, market_ticker: str, book: _LocalBook) -> None:
        ticker = self._make_ticker(market_ticker)
        bid = book.best_yes_bid()
        ask = book.best_yes_ask()
        if bid is not None and Decimal('0') < bid < Decimal('1'):
            self._enqueue(OrderBookEvent(
                ticker=ticker,
                price=bid,
                size=_TOP_OF_BOOK_SIZE,
                size_delta=_TOP_OF_BOOK_SIZE,
                side='bid',
            ))
        if ask is not None and Decimal('0') < ask < Decimal('1'):
            self._enqueue(OrderBookEvent(
                ticker=ticker,
                price=ask,
                size=_TOP_OF_BOOK_SIZE,
                size_delta=_TOP_OF_BOOK_SIZE,
                side='ask',
            ))

    @staticmethod
    def _parse_levels(raw: list[Any]) -> list[tuple[Decimal, Decimal]]:
        levels: list[tuple[Decimal, Decimal]] = []
        for entry in raw or []:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            price = _to_decimal(entry[0])
            qty = _to_decimal(entry[1])
            if price is not None and qty is not None:
                levels.append((price, qty))
        return levels
