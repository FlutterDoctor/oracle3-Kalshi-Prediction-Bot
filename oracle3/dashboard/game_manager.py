"""Multi-game paper-trading coordinator for the web dashboard.

Each *game* the user adds runs its own :class:`TradingEngine` (its own
``PaperTrader`` / ``PositionManager`` / ``StandardRiskManager`` seeded with
that game's capital allocation) so P&L, positions and bot on/off are fully
isolated per game. A single :class:`SharedMarketFeed` polls each exchange
once and fans events out to the games that subscribed to those tickers, so
adding N games does not multiply the market-data API calls.

Paper trading only for now; ``GameSession`` is structured so a live trader
can be swapped in behind the same interface later (see ``mode``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from oracle3.core.trading_engine import TradingEngine
from oracle3.data.data_source import DataSource
from oracle3.data.market_data_manager import MarketDataManager
from oracle3.events.events import Event
from oracle3.position.position_manager import Position, PositionManager
from oracle3.risk.risk_manager import StandardRiskManager
from oracle3.strategy.contrib.fair_value_strategy import FairValueStrategy
from oracle3.ticker.ticker import CashTicker, KalshiTicker, PolyMarketTicker, Ticker
from oracle3.trader.paper_trader import PaperTrader
from oracle3.trader.types import TradeSide

logger = logging.getLogger(__name__)


def _kalshi_keys_present() -> bool:
    return bool(
        os.environ.get('KALSHI_API_KEY_ID')
        and os.environ.get('KALSHI_PRIVATE_KEY_PATH')
    )


def _cash_ticker_for(exchange: str) -> CashTicker:
    if exchange == 'kalshi':
        return CashTicker.KALSHI_USD
    return CashTicker.POLYMARKET_USDC


def _derive_bot_status(decisions: list[Any], engine_status: str) -> dict[str, str]:
    """Turn the strategy's most recent decision into a human-readable status.

    The strategy now records a HOLD decision (with reasoning) on every
    processed tick it doesn't trade on, so the latest decision doubles as a
    liveness signal — if it's recent, the bot is actively evaluating.
    """
    if engine_status in ('bot_paused', 'halted', 'needs_keys', 'resolved', 'starting'):
        return {'label': engine_status.replace('_', ' ').title(), 'detail': '', 'as_of': ''}
    if not decisions:
        return {
            'label': 'Waiting for first price update',
            'detail': 'No market data received for this game yet.',
            'as_of': '',
        }
    last = decisions[-1]
    action = last.action
    reason = last.reasoning
    if action in ('BUY_YES', 'BUY_NO'):
        label = 'Entered position' if last.executed else 'Attempted trade (rejected)'
    elif action.startswith('CLOSE'):
        label = 'Closed position' if last.executed else 'Attempted close (rejected)'
    elif reason.startswith('cooldown'):
        label = 'Waiting (cooldown)'
    elif reason.startswith('confidence'):
        label = 'Monitoring — confidence too low'
    elif reason.startswith('net edge'):
        label = 'Monitoring — no edge yet'
    elif reason.startswith('holding'):
        label = 'Holding position'
    elif reason.startswith('skipped'):
        label = 'Skipping — volume too high'
    else:
        label = 'Evaluating'
    return {'label': label, 'detail': reason, 'as_of': last.timestamp}


@dataclass
class GameMarket:
    """One tradable market within a game (a match usually has several)."""

    market_id: str
    token_id: str  # YES token (Polymarket) or market_ticker (Kalshi)
    no_token_id: str
    event_id: str
    name: str


# ---------------------------------------------------------------------------
# Shared market feed
# ---------------------------------------------------------------------------


class GameFeedView(DataSource):
    """Per-game queue-backed data source; the shared feed pushes into it."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=2000)

    async def get_next_event(self) -> Event | None:
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    def offer(self, event: Event) -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.debug('GameFeedView queue full — dropping event')


class SharedMarketFeed:
    """Owns one upstream live source per exchange and tees events to games."""

    def __init__(self) -> None:
        self._sources: dict[str, DataSource] = {}
        self._subscribers: list[tuple[set[str], GameFeedView]] = []
        self._tee_tasks: dict[str, asyncio.Task[None]] = {}

    def _build_source(self, exchange: str) -> DataSource | None:
        if exchange == 'polymarket':
            from oracle3.data.live.live_data_source import LivePolyMarketDataSource

            return LivePolyMarketDataSource(
                event_cache_file='events_cache.jsonl',
                polling_interval=60.0,
                orderbook_refresh_interval=10.0,
                reprocess_on_start=False,
            )
        if exchange == 'kalshi':
            if not _kalshi_keys_present():
                return None
            from oracle3.data.live.kalshi_data_source import LiveKalshiDataSource

            return LiveKalshiDataSource(
                event_cache_file='kalshi_events_cache.jsonl',
                polling_interval=60.0,
                reprocess_on_start=False,
            )
        return None

    async def ensure_running(self, exchange: str) -> bool:
        """Start the upstream source + tee loop for ``exchange`` if needed.

        Returns ``True`` if a live feed is available, ``False`` otherwise
        (e.g. Kalshi without API keys).
        """
        if exchange in self._tee_tasks:
            return True
        src = self._sources.get(exchange) or self._build_source(exchange)
        if src is None:
            return False
        self._sources[exchange] = src
        await src.start()
        self._tee_tasks[exchange] = asyncio.create_task(self._tee_loop(exchange, src))
        logger.info('SharedMarketFeed started upstream source for %s', exchange)
        return True

    def subscribe(self, tokens: set[str]) -> GameFeedView:
        view = GameFeedView()
        self._subscribers.append((tokens, view))
        return view

    def unsubscribe(self, view: GameFeedView) -> None:
        self._subscribers = [(t, v) for (t, v) in self._subscribers if v is not view]

    def watch(self, exchange: str, token_id: str) -> None:
        src = self._sources.get(exchange)
        watch_token = getattr(src, 'watch_token', None)
        if callable(watch_token) and token_id:
            watch_token(token_id)

    async def _tee_loop(self, exchange: str, src: DataSource) -> None:
        while True:
            try:
                event = await src.get_next_event()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug('shared feed poll error (%s)', exchange, exc_info=True)
                await asyncio.sleep(1.0)
                continue
            if event is None:
                continue
            ticker = getattr(event, 'ticker', None)
            if ticker is None:
                continue
            symbol = ticker.symbol
            for tokens, view in list(self._subscribers):
                if symbol in tokens:
                    view.offer(event)

    async def stop(self) -> None:
        for task in self._tee_tasks.values():
            task.cancel()
        for src in self._sources.values():
            try:
                await src.stop()
            except Exception:
                logger.debug('shared feed source stop error', exc_info=True)
        self._tee_tasks.clear()


# ---------------------------------------------------------------------------
# Per-game session
# ---------------------------------------------------------------------------


class GameSession:
    """One game = one isolated trading engine + paper trader."""

    def __init__(
        self,
        game_id: str,
        title: str,
        exchange: str,
        markets: list[GameMarket],
        allocation: Decimal,
        feed: SharedMarketFeed,
        mode: str = 'paper',
    ) -> None:
        self.game_id = game_id
        self.title = title
        self.exchange = exchange
        self.markets = markets
        self.allocation = allocation
        self.feed = feed
        self.mode = mode
        self.status = 'starting'
        self._task: asyncio.Task[None] | None = None
        self._equity_history: deque[float] = deque(maxlen=180)

        self.market_data = MarketDataManager(
            spread=Decimal('0.01'),
            max_history_per_ticker=None,
            max_timeline_events=None,
        )
        self.position_manager = PositionManager()
        self.position_manager.update_position(
            Position(
                ticker=_cash_ticker_for(exchange),
                quantity=allocation,
                average_cost=Decimal('0'),
                realized_pnl=Decimal('0'),
            )
        )
        self.risk_manager = StandardRiskManager(
            position_manager=self.position_manager,
            market_data=self.market_data,
            initial_capital=allocation,
        )
        # Trade sizes must fit inside a single game's allocation.
        self.risk_manager.max_single_trade_size = allocation
        self.risk_manager.max_position_size = allocation
        self.risk_manager.max_total_exposure = allocation
        self.trader = PaperTrader(
            market_data=self.market_data,
            risk_manager=self.risk_manager,
            position_manager=self.position_manager,
            min_fill_rate=Decimal('0.8'),
            max_fill_rate=Decimal('1.0'),
            commission_rate=Decimal('0.0'),
        )
        self.strategy = FairValueStrategy(platform=exchange)

        tokens = {m.token_id for m in markets if m.token_id}
        tokens |= {m.no_token_id for m in markets if m.no_token_id}
        self.feed_view = feed.subscribe(tokens)
        self.engine = TradingEngine(
            data_source=self.feed_view,
            strategy=self.strategy,
            trader=self.trader,
            initial_capital=allocation,
            continuous=True,
        )

    async def start(self) -> None:
        feed_available = await self.feed.ensure_running(self.exchange)
        if not feed_available:
            self.status = 'needs_keys'
            logger.info('Game %s has no live feed (needs API keys)', self.title)
            return
        for m in self.markets:
            self.feed.watch(self.exchange, m.token_id)
            if m.no_token_id:
                self.feed.watch(self.exchange, m.no_token_id)
        self._task = asyncio.create_task(self.engine.start())
        self.status = 'running'

    def pause_bot(self) -> None:
        """Stop the bot from acting on this game; manual orders still work."""
        self.strategy.set_paused(True)
        if self.status in ('starting', 'running'):
            self.status = 'bot_paused'

    def resume_bot(self) -> None:
        self.strategy.set_paused(False)
        if self.status == 'bot_paused':
            self.status = 'running'

    def _ticker_for(self, market_id: str, is_no: bool) -> Ticker:
        m = next(mm for mm in self.markets if mm.market_id == market_id)
        if self.exchange == 'kalshi':
            return KalshiTicker(
                symbol=m.no_token_id if is_no else m.token_id,
                name=m.name,
                market_ticker=m.token_id,
                event_ticker=m.event_id,
                is_no_side=is_no,
            )
        if is_no:
            return PolyMarketTicker(
                symbol=m.no_token_id,
                name=m.name,
                token_id=m.no_token_id,
                market_id=m.market_id,
                event_id=m.event_id,
                no_token_id=m.token_id,
            )
        return PolyMarketTicker(
            symbol=m.token_id,
            name=m.name,
            token_id=m.token_id,
            market_id=m.market_id,
            event_id=m.event_id,
            no_token_id=m.no_token_id,
        )

    async def manual_order(
        self, market_id: str, side: str, price: float, quantity: float, is_no: bool
    ) -> dict[str, Any]:
        """Place a discretionary human order on one of this game's markets."""
        ticker = self._ticker_for(market_id, is_no)
        trade_side = TradeSide.BUY if side.lower() == 'buy' else TradeSide.SELL
        result = await self.trader.place_order(
            trade_side, ticker, Decimal(str(price)), Decimal(str(quantity))
        )
        if result.order is None:
            reason = result.failure_reason.value if result.failure_reason else 'unknown'
            return {'ok': False, 'error': reason}
        return {
            'ok': True,
            'filled': str(result.order.filled_quantity),
            'status': result.order.status.value,
        }

    def get_strategy_config(self) -> dict[str, Any]:
        return self.strategy.get_thresholds()

    def set_strategy_config(self, **kwargs: Any) -> None:
        self.strategy.set_thresholds(**kwargs)

    def _market_prices(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for m in self.markets:
            ticker = self._ticker_for(m.market_id, is_no=False)
            bid = self.market_data.get_best_bid(ticker)
            ask = self.market_data.get_best_ask(ticker)
            last_prices = self.market_data.get_price_history(ticker, limit=1)
            last = last_prices[-1] if last_prices else None
            rows.append(
                {
                    'market_id': m.market_id,
                    'name': m.name,
                    'bid': str(bid.price) if bid is not None else None,
                    'ask': str(ask.price) if ask is not None else None,
                    'last': str(last) if last is not None else None,
                }
            )
        return rows

    def snapshot(self) -> dict[str, Any]:
        snap = self.engine.get_snapshot()
        self._equity_history.append(float(snap.equity))
        decisions = [
            {
                'time': d.timestamp,
                'action': d.action,
                'executed': d.executed,
                'confidence': d.confidence,
                'reasoning': d.reasoning,
                'ticker_name': d.ticker_name,
            }
            for d in self.strategy.get_decisions()[-30:]
        ]
        bot_status = _derive_bot_status(self.strategy.get_decisions(), self.status)
        positions = [
            {
                'symbol': p.ticker_symbol,
                'name': p.ticker_name,
                'qty': str(p.quantity),
                'avg_cost': str(p.average_cost),
                'current_price': str(p.current_price),
                'unrealized_pnl': str(p.unrealized_pnl),
            }
            for p in snap.positions
        ]
        trades = [
            {
                'time': t.time,
                'side': t.side,
                'name': t.ticker_name,
                'price': str(t.price),
                'qty': str(t.quantity),
                'status': t.status,
            }
            for t in snap.recent_trades
        ]
        total_pnl = snap.total_pnl
        return {
            'game_id': self.game_id,
            'title': self.title,
            'exchange': self.exchange,
            'status': self.status,
            'bot_active': not self.strategy.is_paused(),
            'allocation': str(self.allocation),
            'equity': str(snap.equity),
            'cash': str(snap.cash),
            'realized_pnl': str(snap.realized_pnl),
            'unrealized_pnl': str(snap.unrealized_pnl),
            'total_pnl': str(total_pnl),
            'return_pct': (
                float(total_pnl / self.allocation * 100) if self.allocation > 0 else 0.0
            ),
            'positions': positions,
            'trades': trades[-20:],
            'equity_curve': list(self._equity_history),
            'markets': [
                {'market_id': m.market_id, 'name': m.name, 'token_id': m.token_id}
                for m in self.markets
            ],
            'market_prices': self._market_prices(),
            'decisions': decisions,
            'bot_status': bot_status,
            'strategy_config': self.get_strategy_config(),
        }

    async def stop(self) -> None:
        try:
            await self.engine.stop()
        except Exception:
            logger.debug('engine stop error for %s', self.title, exc_info=True)
        if self._task is not None:
            self._task.cancel()
        self.feed.unsubscribe(self.feed_view)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class GameManager:
    """Owns all live games + the shared market feed."""

    def __init__(self, default_allocation: Decimal = Decimal('1000')) -> None:
        self.default_allocation = default_allocation
        self.feed = SharedMarketFeed()
        self.games: dict[str, GameSession] = {}
        self._read_only = False
        self._resolution_task: asyncio.Task[None] | None = None

    async def add_game(
        self,
        title: str,
        exchange: str,
        markets: list[GameMarket],
        allocation: Decimal | None = None,
    ) -> GameSession:
        game_id = uuid.uuid4().hex[:8]
        session = GameSession(
            game_id=game_id,
            title=title,
            exchange=exchange,
            markets=markets,
            allocation=allocation or self.default_allocation,
            feed=self.feed,
        )
        self.games[game_id] = session
        await session.start()
        return session

    async def remove_game(self, game_id: str) -> bool:
        session = self.games.pop(game_id, None)
        if session is None:
            return False
        await session.stop()
        return True

    def pause_bot(self, game_id: str) -> bool:
        session = self.games.get(game_id)
        if session is None:
            return False
        session.pause_bot()
        return True

    def resume_bot(self, game_id: str) -> bool:
        session = self.games.get(game_id)
        if session is None:
            return False
        session.resume_bot()
        return True

    async def manual_order(
        self,
        game_id: str,
        market_id: str,
        side: str,
        price: float,
        quantity: float,
        is_no: bool = False,
    ) -> dict[str, Any]:
        session = self.games.get(game_id)
        if session is None:
            return {'ok': False, 'error': 'game not found'}
        return await session.manual_order(market_id, side, price, quantity, is_no)

    def get_strategy_config(self, game_id: str) -> dict[str, Any] | None:
        session = self.games.get(game_id)
        if session is None:
            return None
        return session.get_strategy_config()

    def set_strategy_config(self, game_id: str, **kwargs: Any) -> bool:
        session = self.games.get(game_id)
        if session is None:
            return False
        session.set_strategy_config(**kwargs)
        return True

    def emergency_stop(self) -> None:
        """Halt all bot activity and manual trading across every game."""
        self._read_only = True
        for session in self.games.values():
            session.strategy.set_paused(True)
            session.trader.set_read_only(True)
            if session.status not in ('resolved', 'needs_keys'):
                session.status = 'halted'

    def aggregate_state(self) -> dict[str, Any]:
        games = [s.snapshot() for s in self.games.values()]
        total_alloc = sum((Decimal(g['allocation']) for g in games), Decimal('0'))
        total_equity = sum((Decimal(g['equity']) for g in games), Decimal('0'))
        total_pnl = sum((Decimal(g['total_pnl']) for g in games), Decimal('0'))
        return {
            'read_only': self._read_only,
            'summary': {
                'num_games': len(games),
                'total_allocated': str(total_alloc),
                'total_equity': str(total_equity),
                'total_pnl': str(total_pnl),
                'return_pct': (
                    float(total_pnl / total_alloc * 100) if total_alloc > 0 else 0.0
                ),
            },
            'games': games,
        }

    # --- resolution polling ------------------------------------------------

    def start_resolution_polling(self, interval: float = 30.0) -> None:
        if self._resolution_task is None:
            self._resolution_task = asyncio.create_task(self._resolution_loop(interval))

    async def _resolution_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            for session in list(self.games.values()):
                if session.status in ('resolved', 'needs_keys'):
                    continue
                try:
                    await self._check_resolution(session)
                except Exception:
                    logger.debug('resolution check failed', exc_info=True)

    async def _check_resolution(self, session: GameSession) -> None:
        if session.exchange != 'polymarket':
            return
        from oracle3.cli.market_commands import _polymarket_market_info

        all_closed = True
        for m in session.markets:
            info = await _polymarket_market_info(m.market_id)
            if info is None or not info.get('closed', False):
                all_closed = False
        if all_closed and session.markets:
            # The match is over — stop the bot so it can't trade a settled
            # market. Positions stay marked at their last price; P&L is final.
            session.strategy.set_paused(True)
            session.status = 'resolved'
            logger.info('Game %s resolved', session.title)

    async def stop(self) -> None:
        if self._resolution_task is not None:
            self._resolution_task.cancel()
        for session in list(self.games.values()):
            await session.stop()
        await self.feed.stop()
