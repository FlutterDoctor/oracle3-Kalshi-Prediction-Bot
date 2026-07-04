"""Tests for the multi-game dashboard coordinator.

These exercise per-game isolation, the pause-bot-but-keep-manual behaviour,
and emergency stop — all without touching the network (no ``start()``).
"""

from decimal import Decimal

import pytest

from oracle3.dashboard.game_manager import (
    GameManager,
    GameMarket,
    GameSession,
    SharedMarketFeed,
)
from oracle3.order.order_book import Level, OrderBook


def _make_session(alloc: str = '1000', token: str = 'tokA') -> GameSession:
    feed = SharedMarketFeed()
    market = GameMarket(
        market_id='m1',
        token_id=token,
        no_token_id=f'{token}_no',
        event_id='e1',
        name='Team A vs Team B',
    )
    return GameSession(
        game_id='g1',
        title='Team A vs Team B',
        exchange='polymarket',
        markets=[market],
        allocation=Decimal(alloc),
        feed=feed,
    )


def _seed_book(session: GameSession) -> None:
    ticker = session._ticker_for('m1', is_no=False)
    ob = OrderBook()
    ob.update(
        asks=[Level(price=Decimal('0.55'), size=Decimal('1000'))],
        bids=[Level(price=Decimal('0.50'), size=Decimal('1000'))],
    )
    session.market_data.order_books[ticker] = ob


@pytest.mark.asyncio
async def test_games_have_isolated_capital_and_positions() -> None:
    g1 = _make_session(alloc='1000', token='tokA')
    g2 = _make_session(alloc='2000', token='tokB')
    _seed_book(g1)
    _seed_book(g2)

    result = await g1.manual_order('m1', 'buy', 0.55, 10, is_no=False)
    assert result['ok'], result

    # g1 spent cash and opened a position; g2 is completely untouched.
    g1_cash = g1.position_manager.get_position(g1._ticker_for('m1', False).collateral)
    g2_cash = g2.position_manager.get_position(g2._ticker_for('m1', False).collateral)
    assert g1_cash is not None and g1_cash.quantity < Decimal('1000')
    assert g2_cash is not None and g2_cash.quantity == Decimal('2000')
    assert g1.position_manager.get_non_cash_positions()
    assert not g2.position_manager.get_non_cash_positions()


@pytest.mark.asyncio
async def test_pausing_bot_still_allows_manual_orders() -> None:
    session = _make_session()
    _seed_book(session)

    session.pause_bot()
    assert session.status == 'bot_paused'
    assert session.strategy.is_paused()

    result = await session.manual_order('m1', 'buy', 0.55, 5, is_no=False)
    assert result['ok'], result


@pytest.mark.asyncio
async def test_emergency_stop_blocks_manual_orders() -> None:
    manager = GameManager()
    session = _make_session()
    _seed_book(session)
    manager.games[session.game_id] = session

    manager.emergency_stop()
    assert manager._read_only
    assert session.trader.read_only

    result = await manager.manual_order(session.game_id, 'm1', 'buy', 0.55, 5)
    assert not result['ok']


@pytest.mark.asyncio
async def test_kalshi_without_keys_reports_needs_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('KALSHI_API_KEY_ID', raising=False)
    monkeypatch.delenv('KALSHI_PRIVATE_KEY_PATH', raising=False)
    manager = GameManager()
    market = GameMarket(
        market_id='KXNBA-1',
        token_id='KXNBA-1',
        no_token_id='',
        event_id='KXNBA',
        name='Lakers vs Celtics',
    )
    session = await manager.add_game('Lakers vs Celtics', 'kalshi', [market])
    assert session.status == 'needs_keys'


@pytest.mark.asyncio
async def test_aggregate_state_sums_across_games() -> None:
    manager = GameManager()
    g1 = _make_session(alloc='1000', token='tokA')
    g2 = _make_session(alloc='2000', token='tokB')
    manager.games[g1.game_id] = g1
    manager.games['g2'] = g2

    state = manager.aggregate_state()
    assert state['summary']['num_games'] == 2
    assert Decimal(state['summary']['total_allocated']) == Decimal('3000')
    assert len(state['games']) == 2
