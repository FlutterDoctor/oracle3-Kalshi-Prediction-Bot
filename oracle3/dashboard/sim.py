"""Historical simulation helpers for the dashboard's Simulation tab.

Thin wrappers over the existing backtest stack (``HistoricalDataSource`` +
``PaperTrader`` + ``TradingEngine`` + ``PerformanceAnalyzer``). Unlike
``_run_backtest_once`` in ``research_commands`` these return the full equity
curve + trade log so the UI can plot P&L, and expose a market picker.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from oracle3.core.trading_engine import TradingEngine
from oracle3.data.backtest.historical_data_source import HistoricalDataSource
from oracle3.data.backtest.history_reader import iter_history_rows
from oracle3.data.market_data_manager import MarketDataManager
from oracle3.position.position_manager import Position, PositionManager
from oracle3.risk.risk_manager import NoRiskManager
from oracle3.strategy.contrib.fair_value_strategy import FairValueStrategy
from oracle3.ticker.ticker import CashTicker, PolyMarketTicker, Ticker
from oracle3.trader.paper_trader import PaperTrader


def list_sim_markets(history_file: str) -> list[dict[str, Any]]:
    """List the tradable markets in a history file for the picker dropdown."""
    markets: list[dict[str, Any]] = []
    for row in iter_history_rows(history_file):
        series = (row.get('time_series') or {}).get('Yes') or []
        if len(series) < 2:
            continue
        name = (
            row.get('question')
            or row.get('title')
            or row.get('market_title')
            or row.get('name')
            or str(row.get('market_id', ''))
        )
        markets.append(
            {
                'market_id': str(row.get('market_id', '')),
                'event_id': str(row.get('event_id', '')),
                'name': name,
                'points': len(series),
                'first_price': series[0].get('p'),
                'last_price': series[-1].get('p'),
            }
        )
    return markets


async def run_simulation(
    *,
    history_file: str,
    market_id: str,
    event_id: str,
    initial_capital: Decimal,
    commission_rate: Decimal = Decimal('0.0'),
    fill_rate: Decimal = Decimal('1.0'),
    spread: Decimal = Decimal('0.0'),  # marketable fills, matches backtest CLI
    strategy_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay one historical market through the default Wang strategy.

    ``fill_rate`` sets both min and max fill so runs are deterministic.
    """
    ticker = PolyMarketTicker(
        symbol='SIM_TOKEN',
        name='Simulation Market',
        market_id=market_id,
        event_id=event_id,
        token_id='SIM_TOKEN',
    )
    strategy = FairValueStrategy(platform='polymarket', **(strategy_kwargs or {}))
    data_source = HistoricalDataSource(history_file, ticker, include_all_markets=False)
    market_data = MarketDataManager(
        spread=spread, max_history_per_ticker=None, max_timeline_events=None
    )
    position_manager = PositionManager()
    position_manager.update_position(
        Position(
            ticker=CashTicker.POLYMARKET_USDC,
            quantity=initial_capital,
            average_cost=Decimal('0'),
            realized_pnl=Decimal('0'),
        )
    )
    trader = PaperTrader(
        market_data=market_data,
        risk_manager=NoRiskManager(),
        position_manager=position_manager,
        min_fill_rate=fill_rate,
        max_fill_rate=fill_rate,
        commission_rate=commission_rate,
    )
    tradable: list[Ticker | str] = [ticker]
    no_ticker = ticker.get_no_ticker()
    if no_ticker is not None:
        tradable.append(no_ticker)
    trader.set_allowed_tickers(tradable)

    engine = TradingEngine(
        data_source=data_source,
        strategy=strategy,
        trader=trader,
        initial_capital=initial_capital,
    )
    await engine.start()

    stats = engine._perf.get_stats()
    curve = engine._perf.get_equity_curve()
    trades: list[dict[str, Any]] = []
    for order in trader.orders:
        for t in order.trades:
            trades.append(
                {
                    'side': t.side.value,
                    'name': getattr(t.ticker, 'name', '') or t.ticker.symbol,
                    'price': str(t.price),
                    'qty': str(t.quantity),
                }
            )

    return {
        'metrics': {
            'total_trades': stats.total_trades,
            'winning_trades': stats.winning_trades,
            'losing_trades': stats.losing_trades,
            'win_rate': str(stats.win_rate),
            'total_pnl': str(stats.total_pnl),
            'profit_factor': str(stats.profit_factor),
            'max_drawdown': str(stats.max_drawdown),
            'sharpe_ratio': str(stats.sharpe_ratio),
            'final_equity': str(engine._perf.get_current_equity()),
            'return_pct': str(engine._perf.get_return_pct()),
        },
        'equity_curve': [
            {'equity': str(pt.equity), 'trade_index': pt.trade_index} for pt in curve
        ],
        'trades': trades,
    }
