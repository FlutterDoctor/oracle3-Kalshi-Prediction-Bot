"""FastAPI WebSocket server for the Oracle3 web dashboard.

Runs in-process alongside the TradingEngine, providing real-time state
via WebSocket and a single-page HTML dashboard.

NOTE: This module intentionally does NOT use ``from __future__ import
annotations`` because FastAPI relies on runtime annotation evaluation
for dependency injection (e.g. WebSocket parameter type resolution).
"""

import asyncio
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oracle3.core.trading_engine import TradingEngine
    from oracle3.dashboard.game_manager import GameManager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / 'static'

# Wallet address for Solscan links
SOLANA_WALLET = '7RQ3YL4cLNbQbwAUHBP6GzdRbG6NRng8qBcHbiDrf8Ae'

_RISK_LIMIT_KEYS = (
    'max_single_trade_size',
    'max_position_size',
    'max_total_exposure',
    'max_drawdown_pct',
    'daily_loss_limit',
    'max_positions',
)


def _apply_risk_limits(rm: Any, body: dict[str, Any]) -> None:
    """Update a StandardRiskManager's limits in place from a config payload.

    Raises ``InvalidOperation``/``ValueError``/``TypeError`` on bad input;
    the caller turns that into an API error response.
    """
    from decimal import Decimal

    if 'max_single_trade_size' in body:
        rm.max_single_trade_size = Decimal(str(body['max_single_trade_size']))
    if 'max_position_size' in body:
        rm.max_position_size = Decimal(str(body['max_position_size']))
    if 'max_total_exposure' in body:
        rm.max_total_exposure = Decimal(str(body['max_total_exposure']))
    if 'max_drawdown_pct' in body:
        rm.max_drawdown_pct = Decimal(str(body['max_drawdown_pct']))
    if 'daily_loss_limit' in body:
        val = body['daily_loss_limit']
        rm.daily_loss_limit = Decimal(str(val)) if val not in (None, '') else None
    if 'max_positions' in body:
        rm.max_positions = int(body['max_positions'])


def _serialize_snapshot(engine: 'TradingEngine') -> dict[str, Any]:  # noqa: C901
    """Build a JSON-safe state dict from the engine snapshot.

    Re-uses the same data as ControlServer._cmd_get_state() but reads
    directly from the engine's get_snapshot() method for cleaner access.
    """
    snap = engine.get_snapshot()

    # Positions
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

    # Order books — only include markets with two-sided liquidity
    order_books = []
    for ob in snap.orderbooks:
        best_bid = float(ob.bids[0][0]) if ob.bids else 0.0
        best_ask = float(ob.asks[0][0]) if ob.asks else 0.0
        if best_bid <= 0 or best_ask <= 0:
            continue
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2
        order_books.append(
            {
                'symbol': ob.ticker_symbol,
                'bid': f'{best_bid:.4f}',
                'ask': f'{best_ask:.4f}',
                'spread': f'{spread:.4f}',
                'mid_pct': f'{mid * 100:.0f}',
            }
        )

    # Recent trades
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

    # AI decisions from strategy
    decisions: list[dict[str, Any]] = []
    strategy = getattr(engine, 'strategy', None)
    if strategy is not None:
        try:
            raw_decisions = list(strategy.get_decisions())
            for d in raw_decisions[-30:]:
                decisions.append(
                    {
                        'timestamp': d.timestamp,
                        'action': d.action,
                        'ticker_name': (d.ticker_name or '')[:40],
                        'confidence': float(getattr(d, 'confidence', 0.0) or 0.0),
                        'reasoning': (getattr(d, 'reasoning', '') or '')[:80],
                        'executed': bool(d.executed),
                    }
                )
        except Exception:
            logger.debug('Failed to serialize decisions', exc_info=True)

    # Activity log
    activity_log = list(getattr(engine, '_activity_log', []))

    # News
    news = list(getattr(engine, '_news', []))

    # Performance stats from analyzer
    performance: dict[str, Any] = {}
    analyzer = getattr(engine, '_perf', None) or getattr(engine, 'analyzer', None)
    if analyzer is not None:
        try:
            stats = analyzer.get_stats()
            performance = {
                'total_trades': stats.total_trades,
                'winning_trades': stats.winning_trades,
                'losing_trades': stats.losing_trades,
                'win_rate': str(stats.win_rate),
                'average_profit': str(stats.average_profit),
                'average_loss': str(stats.average_loss),
                'max_drawdown': str(stats.max_drawdown),
                'sharpe_ratio': str(stats.sharpe_ratio),
                'profit_factor': str(stats.profit_factor),
                'total_pnl': str(stats.total_pnl),
                'max_consecutive_wins': stats.max_consecutive_wins,
                'max_consecutive_losses': stats.max_consecutive_losses,
            }
        except Exception:
            logger.debug('Failed to serialize performance stats', exc_info=True)

    # Equity curve from analyzer (reuse resolved reference)
    equity_curve: list[str] = []
    if analyzer is not None:
        try:
            curve = analyzer.get_equity_curve()
            equity_curve = [str(pt.equity) for pt in curve]
        except Exception:
            logger.debug('Failed to serialize equity curve', exc_info=True)

    # Initial capital for return % calculation
    initial_capital = str(getattr(engine, '_initial_capital', '10000'))

    # Truncated wallet for display (e.g. "7RQ3...f8Ae")
    wallet_short = (
        f'{SOLANA_WALLET[:4]}...{SOLANA_WALLET[-4:]}'
        if len(SOLANA_WALLET) >= 8
        else SOLANA_WALLET
    )

    # Arbitrage opportunities (Feature 1)
    arbitrage_opportunities: list[dict[str, Any]] = []
    strategy = getattr(engine, 'strategy', None)
    if strategy is not None:
        try:
            find_arb = getattr(strategy, 'find_arbitrage_opportunities', None)
            if callable(find_arb):
                arbitrage_opportunities = find_arb()
        except Exception:
            logger.debug('Failed to get arbitrage opportunities', exc_info=True)

    # Risk status (Feature 2)
    risk_status: dict[str, Any] = {}
    trader = getattr(engine, 'trader', None)
    if trader is not None:
        rm = getattr(trader, 'risk_manager', None)
        if rm is not None:
            get_status = getattr(rm, 'get_risk_status', None)
            if callable(get_status):
                try:
                    risk_status = get_status()
                except Exception:
                    logger.debug('Failed to get risk status', exc_info=True)

    # On-chain signals (Feature 3)
    onchain_signals: list[dict[str, Any]] = []
    ds = getattr(engine, 'data_source', None)
    if ds is not None:
        # Check composite data source children too
        sources = [ds] + list(getattr(ds, 'sources', []))
        for src in sources:
            get_signals = getattr(src, 'get_onchain_signals', None)
            if callable(get_signals):
                try:
                    onchain_signals = get_signals(limit=10)
                except Exception:
                    logger.debug('Failed to get on-chain signals', exc_info=True)
                break

    # Reputation (Feature 5)
    reputation: dict[str, Any] = {}
    rep_mgr = getattr(engine, '_reputation_manager', None)
    if rep_mgr is not None:
        try:
            reputation = rep_mgr.get_my_reputation()
        except Exception:
            logger.debug('Failed to get reputation', exc_info=True)

    # Multi-agent pipeline status (Feature 6)
    pipeline_status: dict[str, Any] = {}
    if strategy is not None:
        coordinator = getattr(strategy, 'coordinator', None)
        if coordinator is not None:
            get_pipeline = getattr(coordinator, 'get_pipeline_status', None)
            if callable(get_pipeline):
                try:
                    pipeline_status = get_pipeline()
                except Exception:
                    logger.debug('Failed to get pipeline status', exc_info=True)

    # MEV Protection status (Feature 4)
    mev_status: dict[str, Any] = {}
    if trader is not None:
        jito = getattr(trader, '_jito_submitter', None)
        if jito is not None:
            get_mev = getattr(jito, 'get_mev_protection_status', None)
            if callable(get_mev):
                try:
                    mev_status = get_mev()
                except Exception:
                    logger.debug('Failed to get MEV status', exc_info=True)

    # Flash Loan Arbitrage stats (Feature 7)
    flash_loan_stats: dict[str, Any] = {}
    fl = getattr(engine, '_flash_loan', None)
    if fl is not None:
        try:
            flash_loan_stats = getattr(fl, 'stats', {}) or {}
        except Exception:
            logger.debug('Failed to get flash loan stats', exc_info=True)
    if not flash_loan_stats and strategy is not None:
        fl_handler = getattr(strategy, 'flash_loan_handler', None)
        if fl_handler is not None:
            try:
                flash_loan_stats = getattr(fl_handler, 'stats', {}) or {}
            except Exception:
                logger.debug(
                    'Failed to get flash loan stats from strategy', exc_info=True
                )

    # Atomic Multi-Leg Trader stats (Feature 8)
    atomic_trader_stats: dict[str, Any] = {}
    at = getattr(engine, '_atomic_trader', None)
    if at is not None:
        try:
            atomic_trader_stats = getattr(at, 'stats', {}) or {}
        except Exception:
            logger.debug('Failed to get atomic trader stats', exc_info=True)

    return {
        'timestamp': datetime.now().isoformat(),
        'running': snap.engine_running,
        'paused': getattr(engine, '_data_paused', False),
        'uptime': snap.uptime,
        'event_count': snap.event_count,
        'initial_capital': initial_capital,
        'network': 'Solana Mainnet',
        'portfolio': {
            'equity': str(snap.equity),
            'cash': str(snap.cash),
            'realized_pnl': str(snap.realized_pnl),
            'unrealized_pnl': str(snap.unrealized_pnl),
            'total_pnl': str(snap.total_pnl),
            'exposure_pct': snap.exposure_pct,
        },
        'positions': positions,
        'order_books': order_books,
        'decisions': decisions,
        'trades': trades,
        'performance': performance,
        'equity_curve': equity_curve,
        'activity_log': activity_log[-50:],
        'news': news[-20:],
        'wallet': SOLANA_WALLET,
        'wallet_short': wallet_short,
        'arbitrage_opportunities': arbitrage_opportunities,
        'risk_status': risk_status,
        'onchain_signals': onchain_signals,
        'reputation': reputation,
        'pipeline_status': pipeline_status,
        'mev_status': mev_status,
        'flash_loan_stats': flash_loan_stats,
        'atomic_trader_stats': atomic_trader_stats,
    }


def _build_dashboard_app(engine: 'TradingEngine'):  # noqa: C901
    """Build the FastAPI app with WebSocket and REST endpoints."""
    try:
        from fastapi import FastAPI, Request, WebSocket
        from fastapi.responses import FileResponse, JSONResponse
    except ImportError as exc:
        raise RuntimeError(
            'FastAPI not installed. Install with: pip install fastapi uvicorn'
        ) from exc

    app = FastAPI(title='Oracle3 Dashboard', version='1.0.0')

    # Track active WebSocket connections for broadcasting
    ws_clients: set[Any] = set()

    @app.get('/')
    async def index():
        """Serve the dashboard HTML."""
        return FileResponse(STATIC_DIR / 'index.html', media_type='text/html')

    @app.get('/live')
    async def live_dashboard():
        """Serve the live trading dashboard HTML."""
        return FileResponse(STATIC_DIR / 'live.html', media_type='text/html')

    @app.get('/api/state')
    async def get_state():
        """One-shot full state snapshot."""
        return JSONResponse(_serialize_snapshot(engine))

    @app.post('/api/command/{cmd}')
    async def send_command(cmd: str):
        """Execute a control command (pause/resume/stop)."""
        if cmd == 'pause':
            engine._data_paused = True
            strategy = getattr(engine, 'strategy', None)
            trader = getattr(engine, 'trader', None)
            if strategy is not None:
                strategy.set_paused(True)
            if trader is not None:
                trader.set_read_only(True)
            return JSONResponse({'ok': True, 'status': 'paused'})
        elif cmd == 'resume':
            engine._data_paused = False
            strategy = getattr(engine, 'strategy', None)
            trader = getattr(engine, 'trader', None)
            if strategy is not None:
                strategy.set_paused(False)
            if trader is not None:
                trader.set_read_only(False)
            return JSONResponse({'ok': True, 'status': 'running'})
        elif cmd == 'stop':
            asyncio.ensure_future(engine.stop())
            return JSONResponse({'ok': True, 'status': 'stopping'})
        else:
            return JSONResponse(
                {'ok': False, 'error': f'Unknown command: {cmd}'}, status_code=400
            )

    @app.get('/api/config')
    async def get_config():
        """Current ticker filter + risk limits (for populating the Controls panel)."""
        from oracle3.core.trading_engine import _resolve_standard_risk_manager

        rm = _resolve_standard_risk_manager(engine.trader.risk_manager)
        limits: dict[str, Any] = {}
        if rm is not None:
            limits = {
                'max_single_trade_size': str(rm.max_single_trade_size),
                'max_position_size': str(rm.max_position_size),
                'max_total_exposure': str(rm.max_total_exposure),
                'max_drawdown_pct': str(rm.max_drawdown_pct),
                'daily_loss_limit': (
                    str(rm.daily_loss_limit)
                    if rm.daily_loss_limit is not None
                    else None
                ),
                'max_positions': rm.max_positions,
            }
        return JSONResponse(
            {
                'risk_manager_active': rm is not None,
                'risk_limits': limits,
                'ticker_filter': engine.get_ticker_filter(),
            }
        )

    @app.post('/api/config')
    async def set_config(request: Request):
        """Update the ticker filter and/or risk limits live, no restart needed."""
        from decimal import InvalidOperation

        from oracle3.core.trading_engine import _resolve_standard_risk_manager

        body = await request.json()
        errors: list[str] = []

        if 'ticker_filter' in body:
            raw = body['ticker_filter']
            if isinstance(raw, str):
                raw = raw.split(',')
            if raw is None or raw == []:
                engine.set_ticker_filter(None)
            elif isinstance(raw, list):
                engine.set_ticker_filter(raw)
            else:
                errors.append('ticker_filter must be a comma-separated string or list')

        if any(k in body for k in _RISK_LIMIT_KEYS):
            rm = _resolve_standard_risk_manager(engine.trader.risk_manager)
            if rm is None:
                errors.append('No adjustable risk manager for this session')
            else:
                try:
                    _apply_risk_limits(rm, body)
                except (InvalidOperation, ValueError, TypeError) as exc:
                    errors.append(f'Invalid risk limit value: {exc}')

        if errors:
            return JSONResponse({'ok': False, 'errors': errors}, status_code=400)
        return JSONResponse({'ok': True})

    @app.get('/api/markets')
    async def get_markets():
        """Return active market tickers from the engine's order books."""
        from oracle3.ticker.ticker import CashTicker as CT

        md = getattr(engine, 'market_data', None)
        if md is None:
            return JSONResponse({'markets': []})
        tickers = []
        for ticker in list(md.order_books.keys()):
            if isinstance(ticker, CT):
                continue
            tickers.append(
                {
                    'symbol': ticker.symbol,
                    'name': getattr(ticker, 'name', '') or ticker.symbol,
                }
            )
        return JSONResponse({'markets': tickers})

    @app.websocket('/ws')
    async def websocket_endpoint(websocket: WebSocket):
        from starlette.websockets import WebSocketDisconnect

        await websocket.accept()
        ws_clients.add(websocket)
        try:
            while True:
                try:
                    state = _serialize_snapshot(engine)
                    await websocket.send_json(state)
                except Exception:
                    logger.debug('WebSocket send failed', exc_info=True)
                    break
                await asyncio.sleep(2.0)
        except WebSocketDisconnect:
            pass
        finally:
            ws_clients.discard(websocket)

    return app


def _group_polymarket_games(markets: list[dict]) -> list[dict[str, Any]]:
    by_event: dict[str, dict[str, Any]] = {}
    for m in markets:
        key = m.get('event_id') or m.get('id') or ''
        entry = by_event.setdefault(
            key,
            {
                'key': f'poly:{key}',
                'exchange': 'polymarket',
                'title': m.get('event_title') or m.get('question') or 'Untitled',
                'markets': [],
            },
        )
        entry['markets'].append(
            {
                'market_id': m.get('id', ''),
                'token_id': m.get('token_id', ''),
                'name': m.get('question', ''),
                'event_id': m.get('event_id', ''),
                'bid': m.get('best_bid', ''),
                'ask': m.get('best_ask', ''),
                'close': m.get('end_date', ''),
            }
        )
    return list(by_event.values())


def _group_kalshi_games(markets: list[dict]) -> list[dict[str, Any]]:
    by_event: dict[str, dict[str, Any]] = {}
    for m in markets:
        key = m.get('event_ticker') or m.get('ticker') or ''
        entry = by_event.setdefault(
            key,
            {
                'key': f'kalshi:{key}',
                'exchange': 'kalshi',
                'title': m.get('title') or key,
                'markets': [],
            },
        )
        entry['markets'].append(
            {
                'market_id': m.get('ticker', ''),
                'token_id': m.get('ticker', ''),
                'name': m.get('title', ''),
                'event_id': m.get('event_ticker', ''),
                # Kalshi quotes cents (0-100); normalize to 0-1 like Polymarket.
                'bid': (m['yes_bid'] / 100) if m.get('yes_bid') else '',
                'ask': (m['yes_ask'] / 100) if m.get('yes_ask') else '',
                'close': m.get('close_time', ''),
            }
        )
    return list(by_event.values())


async def _search_games(query: str, exchange: str) -> list[dict[str, Any]]:
    """Search live markets and group them into 'games' by event.

    ``exchange`` is 'polymarket', 'kalshi' or 'both'. Reuses the read-only
    per-exchange search helpers from ``market_commands``.
    """
    from oracle3.cli import market_commands as mc

    games: list[dict[str, Any]] = []

    if exchange in ('polymarket', 'both'):
        try:
            markets = await mc._polymarket_search_markets(query, 40)
        except Exception:
            logger.debug('polymarket search failed', exc_info=True)
            markets = []
        games.extend(_group_polymarket_games(markets))

    if exchange in ('kalshi', 'both'):
        try:
            markets = await mc._kalshi_search_markets(query, 30, None, None)
        except Exception:
            logger.debug('kalshi search failed (needs keys?)', exc_info=True)
            markets = []
        games.extend(_group_kalshi_games(markets))

    # Cap total groups so the results panel stays scannable.
    return games[:30]


async def _list_live_games(exchange: str) -> list[dict[str, Any]]:
    """List currently open single-matchup games, soonest-closing first.

    No search term — this is the "what can I bet on right now" browse view,
    as opposed to ``_search_games`` which requires a query.
    """
    from oracle3.cli import market_commands as mc

    games: list[dict[str, Any]] = []

    if exchange in ('polymarket', 'both'):
        try:
            markets = await mc._polymarket_live_games(60)
        except Exception:
            logger.debug('polymarket live-games listing failed', exc_info=True)
            markets = []
        games.extend(_group_polymarket_games(markets))

    if exchange in ('kalshi', 'both'):
        try:
            markets = await mc._kalshi_live_games(40, None, None)
        except Exception:
            logger.debug('kalshi live-games listing failed (needs keys?)', exc_info=True)
            markets = []
        games.extend(_group_kalshi_games(markets))

    return games[:30]


async def _list_category_games(category: str, exchange: str) -> list[dict[str, Any]]:
    """Browse open markets in a topic (Sports, Politics, World Cup, …).

    ``category='all'`` (or an unknown key) keeps the original single-matchup
    "what's live right now" behavior (``_list_live_games``); named categories
    use the Gamma tag / Kalshi series-category taxonomy in
    ``market_commands.CATEGORIES`` instead, which also surfaces non-matchup
    markets (e.g. World Cup outright winner) that the sports-only default
    filters out.
    """
    from oracle3.cli import market_commands as mc

    spec = mc.CATEGORIES.get(category)
    if spec is None:
        return await _list_live_games(exchange)

    games: list[dict[str, Any]] = []

    if exchange in ('polymarket', 'both'):
        try:
            markets = await mc._polymarket_category_games(spec['poly_tag'], 60)
        except Exception:
            logger.debug('polymarket category listing failed (%s)', category, exc_info=True)
            markets = []
        games.extend(_group_polymarket_games(markets))

    if exchange in ('kalshi', 'both'):
        try:
            markets = await mc._kalshi_category_games(
                spec['kalshi_category'], spec['kalshi_prefix'], 40, None, None
            )
        except Exception:
            logger.debug('kalshi category listing failed (%s)', category, exc_info=True)
            markets = []
        games.extend(_group_kalshi_games(markets))

    return games[:30]


def _build_games_app(manager: 'GameManager'):  # noqa: C901
    """FastAPI app for the multi-game live + simulation dashboard."""
    try:
        from fastapi import FastAPI, Request, WebSocket
        from fastapi.responses import FileResponse, JSONResponse
    except ImportError as exc:
        raise RuntimeError(
            'FastAPI not installed. Install with: pip install fastapi uvicorn'
        ) from exc

    from decimal import Decimal

    from oracle3.dashboard.game_manager import GameMarket

    app = FastAPI(title='Oracle3 Games Dashboard', version='1.0.0')

    @app.on_event('startup')
    async def _startup():
        # Start resolution polling inside the server's own event loop so all
        # game engines + feed tasks share one loop.
        manager.start_resolution_polling()

    @app.get('/')
    async def index():
        return FileResponse(STATIC_DIR / 'games.html', media_type='text/html')

    @app.get('/api/search')
    async def search(q: str = '', exchange: str = 'both'):
        if not q.strip():
            return JSONResponse({'games': []})
        return JSONResponse({'games': await _search_games(q, exchange)})

    @app.get('/api/live')
    async def live(exchange: str = 'both', category: str = 'all'):
        return JSONResponse({'games': await _list_category_games(category, exchange)})

    @app.get('/api/games')
    async def get_games():
        return JSONResponse(manager.aggregate_state())

    @app.post('/api/games')
    async def add_game(request: Request):
        body = await request.json()
        exchange = body.get('exchange', 'polymarket')
        title = body.get('title', 'Game')
        raw_markets = body.get('markets', [])
        alloc = body.get('allocation')
        allocation = Decimal(str(alloc)) if alloc not in (None, '') else None

        markets: list[GameMarket] = []
        for rm in raw_markets:
            no_token = rm.get('no_token_id', '')
            if exchange == 'polymarket' and not no_token and rm.get('market_id'):
                try:
                    from oracle3.cli.market_commands import _polymarket_market_info

                    info = await _polymarket_market_info(rm['market_id'])
                    if info:
                        no_token = info.get('no_token_id', '')
                except Exception:
                    logger.debug('market info enrich failed', exc_info=True)
            markets.append(
                GameMarket(
                    market_id=rm.get('market_id', ''),
                    token_id=rm.get('token_id', ''),
                    no_token_id=no_token,
                    event_id=rm.get('event_id', ''),
                    name=rm.get('name', ''),
                )
            )
        if not markets:
            return JSONResponse({'ok': False, 'error': 'no markets'}, status_code=400)

        session = await manager.add_game(title, exchange, markets, allocation)
        return JSONResponse(
            {'ok': True, 'game_id': session.game_id, 'status': session.status}
        )

    @app.delete('/api/games/{game_id}')
    async def delete_game(game_id: str):
        ok = await manager.remove_game(game_id)
        return JSONResponse({'ok': ok})

    @app.post('/api/games/{game_id}/bot')
    async def toggle_bot(game_id: str, request: Request):
        body = await request.json()
        action = body.get('action', 'pause')
        ok = (
            manager.pause_bot(game_id)
            if action == 'pause'
            else manager.resume_bot(game_id)
        )
        return JSONResponse({'ok': ok})

    @app.get('/api/games/{game_id}/strategy')
    async def get_strategy(game_id: str):
        cfg = manager.get_strategy_config(game_id)
        if cfg is None:
            return JSONResponse({'ok': False, 'error': 'game not found'}, status_code=404)
        return JSONResponse({'ok': True, **cfg})

    @app.post('/api/games/{game_id}/strategy')
    async def set_strategy(game_id: str, request: Request):
        body = await request.json()
        kwargs: dict[str, Any] = {}
        for key in ('min_edge', 'min_confidence', 'cooldown_seconds'):
            if body.get(key) not in (None, ''):
                try:
                    kwargs[key] = float(body[key])
                except (TypeError, ValueError):
                    return JSONResponse(
                        {'ok': False, 'error': f'invalid {key}'}, status_code=400
                    )
        ok = manager.set_strategy_config(game_id, **kwargs)
        return JSONResponse({'ok': ok}, status_code=200 if ok else 404)

    @app.post('/api/games/{game_id}/order')
    async def manual_order(game_id: str, request: Request):
        body = await request.json()
        result = await manager.manual_order(
            game_id,
            market_id=body.get('market_id', ''),
            side=body.get('side', 'buy'),
            price=float(body.get('price', 0)),
            quantity=float(body.get('quantity', 0)),
            is_no=bool(body.get('is_no', False)),
        )
        status = 200 if result.get('ok') else 400
        return JSONResponse(result, status_code=status)

    @app.post('/api/command/estop')
    async def estop():
        manager.emergency_stop()
        return JSONResponse({'ok': True, 'status': 'halted'})

    @app.get('/api/sim/markets')
    async def sim_markets(history_file: str = 'data/backtest_sample.jsonl'):
        from oracle3.dashboard.sim import list_sim_markets

        try:
            return JSONResponse({'markets': list_sim_markets(history_file)})
        except Exception as exc:
            return JSONResponse({'markets': [], 'error': str(exc)}, status_code=400)

    @app.post('/api/sim/run')
    async def sim_run(request: Request):
        from oracle3.dashboard.sim import run_simulation

        body = await request.json()
        kwargs: dict[str, Any] = {}
        for key in ('min_confidence', 'min_edge', 'cooldown_seconds'):
            if body.get(key) not in (None, ''):
                kwargs[key] = (
                    int(body[key]) if key == 'cooldown_seconds' else float(body[key])
                )
        try:
            result = await run_simulation(
                history_file=body.get('history_file', 'data/backtest_sample.jsonl'),
                market_id=str(body.get('market_id', '')),
                event_id=str(body.get('event_id', '')),
                initial_capital=Decimal(str(body.get('initial_capital', '1000'))),
                commission_rate=Decimal(str(body.get('commission_rate', '0'))),
                strategy_kwargs=kwargs,
            )
            return JSONResponse({'ok': True, **result})
        except Exception as exc:
            logger.debug('sim run failed', exc_info=True)
            return JSONResponse({'ok': False, 'error': str(exc)}, status_code=400)

    @app.websocket('/ws')
    async def websocket_endpoint(websocket: WebSocket):
        from starlette.websockets import WebSocketDisconnect

        await websocket.accept()
        try:
            while True:
                try:
                    await websocket.send_json(manager.aggregate_state())
                except Exception:
                    logger.debug('WS send failed', exc_info=True)
                    break
                await asyncio.sleep(2.0)
        except WebSocketDisconnect:
            pass

    return app


class GamesDashboardServer:
    """Runs the multi-game dashboard (uvicorn in a background thread)."""

    def __init__(self, manager: 'GameManager', host: str = '0.0.0.0', port: int = 3000):
        self.manager = manager
        self.host = host
        self.port = port
        self._server_thread: threading.Thread | None = None
        self._uvicorn_server: Any = None

    def start(self) -> None:
        import uvicorn

        app = _build_games_app(self.manager)
        config = uvicorn.Config(
            app, host=self.host, port=self.port, log_level='warning'
        )
        self._uvicorn_server = uvicorn.Server(config)

        def _run():
            asyncio.run(self._uvicorn_server.serve())

        self._server_thread = threading.Thread(
            target=_run, daemon=True, name='games-dashboard'
        )
        self._server_thread.start()
        logger.info('Games dashboard on http://%s:%d', self.host, self.port)

    def stop(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True


class DashboardServer:
    """Manages the FastAPI dashboard server lifecycle.

    Runs uvicorn in a background thread so the main asyncio loop
    remains free for the TradingEngine.
    """

    def __init__(
        self,
        engine: 'TradingEngine',
        host: str = '0.0.0.0',
        port: int = 3000,
    ):
        self.engine = engine
        self.host = host
        self.port = port
        self._server_thread: threading.Thread | None = None
        self._uvicorn_server: Any = None

    def start(self) -> None:
        """Start the dashboard server in a background thread."""
        import uvicorn

        app = _build_dashboard_app(self.engine)
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level='warning',
        )
        self._uvicorn_server = uvicorn.Server(config)

        def _run():
            asyncio.run(self._uvicorn_server.serve())

        self._server_thread = threading.Thread(
            target=_run, daemon=True, name='dashboard'
        )
        self._server_thread.start()
        logger.info('Dashboard server started on http://%s:%d', self.host, self.port)

    def stop(self) -> None:
        """Signal the uvicorn server to shut down."""
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
