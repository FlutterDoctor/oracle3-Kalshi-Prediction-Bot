"""Polymarket CLOB WebSocket streaming data source.

Connects to the public ``market`` channel
(``wss://ws-subscriptions-clob.polymarket.com/ws/market``) for real-time
order-book and trade data — no authentication required, but the server
drops connections that don't subscribe immediately after connecting.

Includes a data-inactivity watchdog because of a documented upstream bug
(py-clob-client issue #292): the connection can silently stop delivering
book updates while ping/pong keepalive keeps working, with no error and
no clean disconnect. The watchdog force-closes the socket (triggering the
normal reconnect path) if no message has arrived within ``_STALE_TIMEOUT``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from decimal import Decimal
from typing import Any

import websockets

from ...events.events import Event, OrderBookEvent, PriceChangeEvent
from ...ticker.ticker import PolyMarketTicker
from ..data_source import DataSource
from ._orderbook_diff import diff_order_book_to_events

logger = logging.getLogger(__name__)

_WS_URL = 'wss://ws-subscriptions-clob.polymarket.com/ws/market'
_PING_INTERVAL = 10.0
_STALE_TIMEOUT = 120.0
_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 300.0


def _to_decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except Exception:
        return None


class PolymarketWebSocketDataSource(DataSource):
    """Streams real-time Polymarket CLOB market data via the public ws feed."""

    def __init__(self, ws_url: str = _WS_URL) -> None:
        self.ws_url = ws_url
        self.event_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=2000)
        self._ws_task: asyncio.Task | None = None

        self._desired_assets: set[str] = set()
        # asset_id -> (name, market_id, event_id, no_token_id)
        self._asset_meta: dict[str, tuple[str, str, str, str]] = {}
        self._book_state: dict[str, Decimal] = {}
        self._control_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._connected = False

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
        token_id: str,
        name: str = '',
        market_id: str = '',
        event_id: str = '',
        no_token_id: str = '',
    ) -> None:
        """Add a token to the live subscription (called by ``SharedMarketFeed``)."""
        if not token_id:
            return
        self._asset_meta[token_id] = (name, market_id, event_id, no_token_id)
        if token_id in self._desired_assets:
            return
        self._desired_assets.add(token_id)
        if self._connected:
            self._control_queue.put_nowait({
                'assets_ids': [token_id],
                'operation': 'subscribe',
            })

    def _make_ticker(self, token_id: str) -> PolyMarketTicker:
        name, market_id, event_id, no_token_id = self._asset_meta.get(
            token_id, ('', '', '', '')
        )
        return PolyMarketTicker(
            symbol=token_id,
            name=name,
            token_id=token_id,
            market_id=market_id,
            event_id=event_id,
            no_token_id=no_token_id,
        )

    def _enqueue(self, event: Event) -> None:
        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.debug('PolymarketWS queue full, dropping event: %s', type(event).__name__)

    # ------------------------------------------------------------------
    # WebSocket loop with auto-reconnect + inactivity watchdog
    # ------------------------------------------------------------------

    async def _ws_loop(self) -> None:
        backoff = _INITIAL_BACKOFF
        while True:
            self.connection_state = 'connecting'
            self._connected = False
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info('PolymarketWS connected to %s', self.ws_url)
                    # Must subscribe immediately — the server drops
                    # connections that don't subscribe within a short window.
                    await self._subscribe_all(ws)
                    self.connection_state = 'connected'
                    self._connected = True
                    self.last_error = ''
                    self.reconnect_attempt = 0
                    self.next_reconnect_at = None
                    self.last_message_at = time.monotonic()
                    backoff = _INITIAL_BACKOFF

                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    watchdog_task = asyncio.create_task(self._watchdog(ws))
                    sender_task = asyncio.create_task(self._control_sender(ws))
                    try:
                        async for raw in ws:
                            self.last_message_at = time.monotonic()
                            if raw in ('PONG', 'PING'):
                                continue
                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError:
                                logger.debug(
                                    'PolymarketWS non-JSON message: %s', raw[:200]
                                )
                                continue
                            self._handle_message(msg)
                    finally:
                        self._connected = False
                        for task in (ping_task, watchdog_task, sender_task):
                            task.cancel()

            except asyncio.CancelledError:
                raise
            except Exception:
                self.connection_state = 'reconnecting'
                self.reconnect_attempt += 1
                self.next_reconnect_at = time.monotonic() + backoff
                logger.warning(
                    'PolymarketWS connection lost (reconnect in %.0fs)',
                    backoff,
                    exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    async def _subscribe_all(self, ws) -> None:
        await ws.send(json.dumps({
            'assets_ids': sorted(self._desired_assets),
            'type': 'market',
            'custom_feature_enabled': True,
        }))

    async def _ping_loop(self, ws) -> None:
        try:
            while True:
                await asyncio.sleep(_PING_INTERVAL)
                await ws.send('PING')
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug('PolymarketWS ping failed', exc_info=True)

    async def _watchdog(self, ws, check_interval: float = 10.0) -> None:
        """Force-reconnect if the socket goes quiet.

        Works around a documented upstream bug where the connection can
        silently stop delivering book updates while ping/pong keepalive
        keeps working (py-clob-client issue #292).
        """
        try:
            while True:
                await asyncio.sleep(check_interval)
                if self.last_message_at is None:
                    continue
                if time.monotonic() - self.last_message_at > _STALE_TIMEOUT:
                    logger.warning(
                        'PolymarketWS stale for >%.0fs, forcing reconnect',
                        _STALE_TIMEOUT,
                    )
                    await ws.close()
                    return
        except asyncio.CancelledError:
            raise

    async def _control_sender(self, ws) -> None:
        """Drains dynamically-queued subscription updates onto the live socket."""
        try:
            while True:
                msg = await self._control_queue.get()
                await ws.send(json.dumps(msg))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug('PolymarketWS control send failed', exc_info=True)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _handle_message(self, msg: Any) -> None:
        if isinstance(msg, list):
            for item in msg:
                if isinstance(item, dict):
                    self._dispatch(item)
            return
        if isinstance(msg, dict):
            self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        event_type = msg.get('event_type', '')
        if event_type == 'book':
            self._handle_book(msg)
        elif event_type == 'price_change':
            self._handle_price_change(msg)
        elif event_type == 'last_trade_price':
            self._handle_last_trade(msg)
        elif event_type == 'tick_size_change':
            # Stale tick sizes cause REST order rejections; not acted on
            # for paper trading, but logged for visibility.
            logger.info('PolymarketWS tick_size_change: %s', msg)
        else:
            logger.debug('PolymarketWS unhandled event_type: %s', event_type)

    def _handle_book(self, msg: dict[str, Any]) -> None:
        asset_id = msg.get('asset_id', '')
        if not asset_id:
            return
        ticker = self._make_ticker(asset_id)
        bid_levels = [
            (lvl.get('price', ''), lvl.get('size', ''))
            for lvl in msg.get('bids', []) or []
            if isinstance(lvl, dict)
        ]
        ask_levels = [
            (lvl.get('price', ''), lvl.get('size', ''))
            for lvl in msg.get('asks', []) or []
            if isinstance(lvl, dict)
        ]
        events = diff_order_book_to_events(
            asset_id, ticker, bid_levels, ask_levels, self._book_state
        )
        for event in events:
            self._enqueue(event)

    def _handle_price_change(self, msg: dict[str, Any]) -> None:
        for change in msg.get('price_changes', []) or []:
            asset_id = change.get('asset_id', '')
            price_str = change.get('price')
            size_str = change.get('size')
            side_raw = str(change.get('side', ''))
            price = _to_decimal(price_str)
            size = _to_decimal(size_str)
            if not asset_id or price is None or size is None:
                continue
            side = 'bid' if side_raw.upper() == 'BUY' else 'ask'
            ticker = self._make_ticker(asset_id)
            key = f'{asset_id}:{price_str}:{side}'
            prev_size = self._book_state.get(key, Decimal('0'))
            self._book_state[key] = size
            if size == prev_size:
                continue
            self._enqueue(OrderBookEvent(
                ticker=ticker,
                price=price,
                size=size,
                size_delta=size - prev_size,
                side=side,
            ))

    def _handle_last_trade(self, msg: dict[str, Any]) -> None:
        asset_id = msg.get('asset_id', '')
        price = _to_decimal(msg.get('price'))
        if not asset_id or price is None:
            return
        ticker = self._make_ticker(asset_id)
        self._enqueue(PriceChangeEvent(ticker=ticker, price=price))
