"""CLI commands for browsing prediction markets on Polymarket and Kalshi."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

import click
import httpx

# ---------------------------------------------------------------------------
# Polymarket helpers (via Gamma API — no auth required)
# ---------------------------------------------------------------------------

GAMMA_EVENTS_URL = 'https://gamma-api.polymarket.com/events'
GAMMA_MARKETS_URL = 'https://gamma-api.polymarket.com/markets'
GAMMA_SEARCH_URL = 'https://gamma-api.polymarket.com/public-search'
CLOB_PRICES_HISTORY_URL = 'https://clob.polymarket.com/prices-history'


def _parse_clob_ids(mkt: dict) -> list[str]:
    """Return the parsed list of CLOB token IDs from a market dict (handles JSON-string encoding)."""
    raw = mkt.get('clobTokenIds') or mkt.get('clob_token_ids') or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            raw = []
    return list(raw)


def _event_markets(event: dict) -> list[dict[str, Any]]:
    """Flatten one Gamma event's ``markets`` into our common market-dict shape."""
    return [
        {
            'id': mkt.get('id', ''),
            'question': mkt.get('question', ''),
            'event_id': str(event.get('id', '')),
            'event_title': event.get('title', ''),
            'token_id': _parse_clob_ids(mkt)[0] if _parse_clob_ids(mkt) else '',
            'best_bid': mkt.get('bestBid', ''),
            'best_ask': mkt.get('bestAsk', ''),
            'volume': mkt.get('volume', ''),
            'end_date': mkt.get('endDate', ''),
        }
        for mkt in event.get('markets', [])
    ]


async def _polymarket_list_markets(limit: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            GAMMA_EVENTS_URL,
            params={'active': 'true', 'closed': 'false', 'limit': min(limit, 100)},
        )
    if resp.status_code != 200:
        raise click.ClickException(
            f'Polymarket API returned HTTP {resp.status_code}: {resp.text[:200]}'
        )
    events = resp.json()
    markets: list[dict[str, Any]] = []
    for event in events[:limit]:
        markets.extend(_event_markets(event))
        if len(markets) >= limit:
            break
    return markets[:limit]


async def _polymarket_search_markets(query: str, limit: int) -> list[dict]:
    """Full-text search via Gamma's ``public-search`` endpoint.

    Unlike ``_polymarket_list_markets`` (which only sees ~100 events in
    whatever order the API returns them), this searches Gamma's full
    catalog so queries like a team or country name reliably surface
    matching events regardless of volume/recency.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            GAMMA_SEARCH_URL,
            params={
                'q': query,
                'limit_per_type': min(limit, 50),
                'events_status': 'active',
            },
        )
    if resp.status_code != 200:
        raise click.ClickException(
            f'Polymarket API returned HTTP {resp.status_code}: {resp.text[:200]}'
        )
    events = resp.json().get('events', [])
    markets: list[dict[str, Any]] = []
    for event in events:
        if event.get('closed') or not event.get('active'):
            continue
        markets.extend(_event_markets(event))
        if len(markets) >= limit:
            break
    return markets[:limit]


# Topic tabs for browsing without a search query. Each maps to a Polymarket
# Gamma tag slug and, on Kalshi, either a ``Series.category`` value or (for
# World Cup, which isn't its own top-level Kalshi category) a series-ticker
# prefix. Verified against both APIs directly rather than guessed.
CATEGORIES: dict[str, dict[str, str | None]] = {
    'world-cup': {
        'label': 'World Cup',
        'poly_tag': 'fifa-world-cup',
        'kalshi_category': None,
        'kalshi_prefix': 'KXWC',
    },
    'sports': {
        'label': 'Sports',
        'poly_tag': 'sports',
        'kalshi_category': 'Sports',
        'kalshi_prefix': None,
    },
    'politics': {
        'label': 'Politics',
        'poly_tag': 'politics',
        'kalshi_category': 'Politics',
        'kalshi_prefix': None,
    },
    'crypto': {
        'label': 'Crypto',
        'poly_tag': 'crypto',
        'kalshi_category': 'Crypto',
        'kalshi_prefix': None,
    },
    'economy': {
        'label': 'Economy',
        'poly_tag': 'economy',
        'kalshi_category': 'Economics',
        'kalshi_prefix': None,
    },
    'entertainment': {
        'label': 'Entertainment',
        'poly_tag': 'pop-culture',
        'kalshi_category': 'Entertainment',
        'kalshi_prefix': None,
    },
}


async def _polymarket_category_games(tag_slug: str, limit: int) -> list[dict]:
    """List active events under a Gamma tag, highest 24h volume first.

    Caps markets taken per event so one big outright (e.g. "World Cup
    Winner", with one market per team) can't crowd out every other event
    in the tag before ``limit`` is reached.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            GAMMA_EVENTS_URL,
            params={
                'active': 'true',
                'closed': 'false',
                'limit': min(limit, 100),
                'tag_slug': tag_slug,
                'order': 'volume24hr',
                'ascending': 'false',
            },
        )
    if resp.status_code != 200:
        raise click.ClickException(
            f'Polymarket API returned HTTP {resp.status_code}: {resp.text[:200]}'
        )
    markets: list[dict[str, Any]] = []
    for event in resp.json():
        markets.extend(_event_markets(event)[:10])
        if len(markets) >= limit:
            break
    return markets[:limit]


# Sport tags whose event listings include actual head-to-head matchups (as
# opposed to season-long futures/props, which dominate most tag listings —
# see the " vs. " filter in ``_polymarket_live_games``).
_POLYMARKET_GAME_TAGS = (
    'nba',
    'nfl',
    'mlb',
    'nhl',
    'mls',
    'soccer',
    'ncaaf',
    'ncaab',
    'ufc',
)


def _is_future_iso(iso_str: str) -> bool:
    """True if ``iso_str`` (Gamma's ISO-8601 'Z' timestamps) is still ahead of now.

    Some markets stay ``closed=false`` well past their listed end date while
    resolution is pending — those are stale, not "live", so callers filter
    them out with this check.
    """
    if not iso_str:
        return False
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
    except ValueError:
        return False
    return dt > datetime.now(timezone.utc)


async def _polymarket_live_games(limit: int) -> list[dict]:
    """List currently open single-matchup ("X vs. Y") markets, soonest first.

    Tag-filtered event listings are mostly season futures/props; only
    entries whose title reads as a head-to-head matchup are an actual game
    someone could bet on right now, so we filter for that pattern.
    """
    seen_events: set[str] = set()
    games: list[dict] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for tag in _POLYMARKET_GAME_TAGS:
            resp = await client.get(
                GAMMA_EVENTS_URL,
                params={
                    'active': 'true',
                    'closed': 'false',
                    'limit': 200,
                    'tag_slug': tag,
                },
            )
            if resp.status_code != 200:
                continue
            for event in resp.json():
                title = event.get('title', '')
                if ' vs. ' not in title and ' vs ' not in title:
                    continue
                if not _is_future_iso(event.get('endDate', '')):
                    continue
                event_id = str(event.get('id', ''))
                if event_id in seen_events:
                    continue
                seen_events.add(event_id)
                for mkt in event.get('markets', []):
                    clob_ids = _parse_clob_ids(mkt)
                    games.append(
                        {
                            'id': mkt.get('id', ''),
                            'question': mkt.get('question', ''),
                            'event_id': event_id,
                            'event_title': title,
                            'token_id': clob_ids[0] if clob_ids else '',
                            'best_bid': mkt.get('bestBid', ''),
                            'best_ask': mkt.get('bestAsk', ''),
                            'volume': mkt.get('volume', ''),
                            'end_date': mkt.get('endDate', event.get('endDate', '')),
                        }
                    )
    games.sort(key=lambda m: m.get('end_date') or '9999')
    return games[:limit]


async def _polymarket_market_info(market_id: str) -> dict | None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(GAMMA_MARKETS_URL, params={'id': market_id})
    if resp.status_code != 200:
        return None
    data = resp.json()
    if isinstance(data, list) and data:
        mkt = data[0]
    elif isinstance(data, dict):
        mkt = data
    else:
        return None

    clob_ids = _parse_clob_ids(mkt)
    return {
        'id': mkt.get('id', ''),
        'question': mkt.get('question', ''),
        'event_id': str(mkt.get('eventId', '')),
        'token_id': clob_ids[0] if clob_ids else '',
        'no_token_id': clob_ids[1] if len(clob_ids) > 1 else '',
        'best_bid': mkt.get('bestBid', ''),
        'best_ask': mkt.get('bestAsk', ''),
        'volume': mkt.get('volume', ''),
        'end_date': mkt.get('endDate', ''),
        'description': mkt.get('description', ''),
        'active': mkt.get('active', True),
        'closed': mkt.get('closed', False),
    }


# ---------------------------------------------------------------------------
# Kalshi helpers
# ---------------------------------------------------------------------------

KALSHI_API_URL = 'https://api.elections.kalshi.com/trade-api/v2'

# Kalshi's "multivariate collections" product auto-generates large numbers of
# zero-liquidity combo markets whose title mashes together many unrelated
# props (ticker prefix KXMVE...). They currently dominate the plain
# ``status=open`` market listing, so we filter them out everywhere.
_KALSHI_JUNK_TICKER_PREFIX = 'KXMVE'


async def _kalshi_list_markets(
    limit: int, api_key_id: str | None, private_key_path: str | None
) -> list[dict]:
    from kalshi_python import Configuration
    from kalshi_python.api.markets_api import MarketsApi
    from kalshi_python.api_client import ApiClient

    config = Configuration(host=KALSHI_API_URL)
    key_id = api_key_id or os.environ.get('KALSHI_API_KEY_ID')
    pk_path = private_key_path or os.environ.get('KALSHI_PRIVATE_KEY_PATH')
    if key_id and pk_path:
        with open(pk_path) as f:
            config.private_key_pem = f.read()
        config.api_key_id = key_id

    api_client = ApiClient(configuration=config)
    markets_api = MarketsApi(api_client)

    # A single page only returns 200 markets; page via cursor so ``limit``
    # is actually honored for callers requesting more (e.g. keyword search
    # scanning 500), capped at a safety ceiling to avoid runaway requests.
    page_size = 200
    safety_cap = 1000
    markets: list[dict] = []
    cursor: str | None = None
    while len(markets) < min(limit, safety_cap):
        kwargs: dict[str, Any] = {'status': 'open', 'limit': page_size}
        if cursor:
            kwargs['cursor'] = cursor
        response = await asyncio.to_thread(
            lambda kw=kwargs: markets_api.get_markets(**kw)  # type: ignore[misc]
        )
        raw = response.markets if hasattr(response, 'markets') else []
        for m in raw or []:
            d = m.to_dict() if hasattr(m, 'to_dict') else dict(m)
            if d.get('ticker', '').startswith(_KALSHI_JUNK_TICKER_PREFIX):
                continue
            markets.append(
                {
                    'ticker': d.get('ticker', ''),
                    'title': d.get('title', ''),
                    'event_ticker': d.get('event_ticker', ''),
                    'series_ticker': d.get('series_ticker', ''),
                    'yes_bid': d.get('yes_bid', 0),
                    'yes_ask': d.get('yes_ask', 0),
                    'volume': d.get('volume', 0),
                    'close_time': str(d.get('close_time', '')),
                    'status': d.get('status', ''),
                }
            )
        cursor = getattr(response, 'cursor', None)
        if not raw or not cursor:
            break
    return markets[:limit]


async def _kalshi_list_series(
    api_key_id: str | None, private_key_path: str | None
) -> list[dict]:
    """List Kalshi's ~11k series (clean, human-titled market groupings).

    Unlike the raw markets feed (see ``_KALSHI_JUNK_TICKER_PREFIX``), the
    series catalog is not swamped by synthetic combo products, so keyword
    search over series titles is a far better entry point than scanning
    ``get_markets`` directly.
    """
    from kalshi_python import Configuration
    from kalshi_python.api.series_api import SeriesApi
    from kalshi_python.api_client import ApiClient

    config = Configuration(host=KALSHI_API_URL)
    key_id = api_key_id or os.environ.get('KALSHI_API_KEY_ID')
    pk_path = private_key_path or os.environ.get('KALSHI_PRIVATE_KEY_PATH')
    if key_id and pk_path:
        with open(pk_path) as f:
            config.private_key_pem = f.read()
        config.api_key_id = key_id

    api_client = ApiClient(configuration=config)
    series_api = SeriesApi(api_client)
    response = await asyncio.to_thread(lambda: series_api.get_series())
    raw = response.series if hasattr(response, 'series') else []
    out: list[dict] = []
    for s in raw or []:
        d = s.to_dict() if hasattr(s, 'to_dict') else dict(s)
        ticker = d.get('ticker', '')
        if not ticker or ticker.startswith(_KALSHI_JUNK_TICKER_PREFIX):
            continue
        out.append(
            {'ticker': ticker, 'title': d.get('title', ''), 'category': d.get('category', '')}
        )
    return out


# Well-known single-matchup ("X vs Y Winner?") series. Series *titles* are
# generic league names ("Pro Basketball Game"), not team names, so a search
# for e.g. "Lakers" or "Argentina" won't match any series title — these are
# swept directly and their market titles matched against the query so
# team/game-name searches still find live matchups.
_KALSHI_GAME_SERIES = (
    'KXNBAGAME',
    'KXNFLGAME',
    'KXMLBGAME',
    'KXNHLGAME',
    'KXWCGAME',
    'KXMLSGAME',
    'KXUEFAGAME',
    'KXNCAAFGAME',
    'KXNCAABGAME',
)


async def _kalshi_markets_for_series(
    markets_api: Any, series_ticker: str, series_title: str
) -> list[dict]:
    try:
        response = await asyncio.to_thread(
            lambda: markets_api.get_markets(
                series_ticker=series_ticker, status='open', limit=100
            )
        )
    except Exception:
        return []
    raw = response.markets if hasattr(response, 'markets') else []
    out: list[dict] = []
    for m in raw or []:
        d = m.to_dict() if hasattr(m, 'to_dict') else dict(m)
        out.append(
            {
                'ticker': d.get('ticker', ''),
                'title': d.get('title', ''),
                'event_ticker': d.get('event_ticker', ''),
                'series_ticker': series_ticker,
                'series_title': series_title,
                'yes_bid': d.get('yes_bid', 0) or 0,
                'yes_ask': d.get('yes_ask', 0) or 0,
                'volume': d.get('volume', 0) or 0,
                'close_time': str(d.get('close_time', '')),
                'status': d.get('status', ''),
            }
        )
    return out


async def _kalshi_search_markets(
    query: str, limit: int, api_key_id: str | None, private_key_path: str | None
) -> list[dict]:
    """Search by series title, then list each matching series' open markets.

    Searching the raw ``get_markets`` feed directly returns near-entirely
    synthetic combo junk right now (see ``_KALSHI_JUNK_TICKER_PREFIX``), so
    we match against the clean series catalog first. A second bounded pass
    over well-known live "X vs Y" game series catches team-name queries
    that don't appear in any series title (see ``_KALSHI_GAME_SERIES``).
    """
    from kalshi_python import Configuration
    from kalshi_python.api.markets_api import MarketsApi
    from kalshi_python.api_client import ApiClient

    q = query.lower()
    all_series = await _kalshi_list_series(api_key_id, private_key_path)
    matched_series = [s for s in all_series if q in s['title'].lower()][:15]

    config = Configuration(host=KALSHI_API_URL)
    key_id = api_key_id or os.environ.get('KALSHI_API_KEY_ID')
    pk_path = private_key_path or os.environ.get('KALSHI_PRIVATE_KEY_PATH')
    if key_id and pk_path:
        with open(pk_path) as f:
            config.private_key_pem = f.read()
        config.api_key_id = key_id
    api_client = ApiClient(configuration=config)
    markets_api = MarketsApi(api_client)

    markets: list[dict] = []
    seen_tickers: set[str] = set()

    # Live game hits go first — "is there a game for X happening" is the
    # dashboard's primary use case, and topical series (e.g. "Spain
    # election") shouldn't crowd them out of a small ``limit``.
    for series_ticker in _KALSHI_GAME_SERIES:
        for m in await _kalshi_markets_for_series(
            markets_api, series_ticker, series_ticker
        ):
            if m['ticker'] in seen_tickers or q not in m['title'].lower():
                continue
            seen_tickers.add(m['ticker'])
            markets.append(m)

    for series in matched_series:
        if len(markets) >= limit:
            break
        for m in await _kalshi_markets_for_series(
            markets_api, series['ticker'], series['title']
        ):
            if m['ticker'] in seen_tickers:
                continue
            seen_tickers.add(m['ticker'])
            markets.append(m)

    return markets[:limit]


async def _kalshi_live_games(
    limit: int, api_key_id: str | None, private_key_path: str | None
) -> list[dict]:
    """List all open markets in known single-matchup game series, soonest first."""
    from kalshi_python import Configuration
    from kalshi_python.api.markets_api import MarketsApi
    from kalshi_python.api_client import ApiClient

    config = Configuration(host=KALSHI_API_URL)
    key_id = api_key_id or os.environ.get('KALSHI_API_KEY_ID')
    pk_path = private_key_path or os.environ.get('KALSHI_PRIVATE_KEY_PATH')
    if key_id and pk_path:
        with open(pk_path) as f:
            config.private_key_pem = f.read()
        config.api_key_id = key_id
    api_client = ApiClient(configuration=config)
    markets_api = MarketsApi(api_client)

    markets: list[dict] = []
    for series_ticker in _KALSHI_GAME_SERIES:
        markets.extend(
            await _kalshi_markets_for_series(markets_api, series_ticker, series_ticker)
        )
    markets.sort(key=lambda m: m.get('close_time') or '9999')
    return markets[:limit]


async def _kalshi_category_games(
    category: str | None,
    ticker_prefix: str | None,
    limit: int,
    api_key_id: str | None,
    private_key_path: str | None,
) -> list[dict]:
    """List open markets across series matching a topic (see ``CATEGORIES``).

    Matches by ``Series.category`` (e.g. 'Politics'), or by ticker prefix
    for topics like World Cup that span a category rather than being one.
    The series catalog carries no volume signal, so results are bounded to
    a handful of series rather than ranked by popularity.
    """
    from kalshi_python import Configuration
    from kalshi_python.api.markets_api import MarketsApi
    from kalshi_python.api_client import ApiClient

    all_series = await _kalshi_list_series(api_key_id, private_key_path)
    if ticker_prefix:
        matched = [s for s in all_series if s['ticker'].startswith(ticker_prefix)]
    else:
        matched = [s for s in all_series if s.get('category') == category]
    matched = matched[:25]

    config = Configuration(host=KALSHI_API_URL)
    key_id = api_key_id or os.environ.get('KALSHI_API_KEY_ID')
    pk_path = private_key_path or os.environ.get('KALSHI_PRIVATE_KEY_PATH')
    if key_id and pk_path:
        with open(pk_path) as f:
            config.private_key_pem = f.read()
        config.api_key_id = key_id
    api_client = ApiClient(configuration=config)
    markets_api = MarketsApi(api_client)

    markets: list[dict] = []
    for s in matched:
        markets.extend(await _kalshi_markets_for_series(markets_api, s['ticker'], s['title']))
    markets.sort(key=lambda m: m.get('close_time') or '9999')
    return markets[:limit]


async def _kalshi_market_info(
    market_ticker: str, api_key_id: str | None, private_key_path: str | None
) -> dict | None:
    from kalshi_python import Configuration
    from kalshi_python.api.markets_api import MarketsApi
    from kalshi_python.api_client import ApiClient

    config = Configuration(host=KALSHI_API_URL)
    key_id = api_key_id or os.environ.get('KALSHI_API_KEY_ID')
    pk_path = private_key_path or os.environ.get('KALSHI_PRIVATE_KEY_PATH')
    if key_id and pk_path:
        with open(pk_path) as f:
            config.private_key_pem = f.read()
        config.api_key_id = key_id

    api_client = ApiClient(configuration=config)
    markets_api = MarketsApi(api_client)

    response = await asyncio.to_thread(lambda: markets_api.get_market(market_ticker))
    if not response:
        return None
    m = response.market if hasattr(response, 'market') else response
    if m is None:
        return None
    d = m.to_dict() if hasattr(m, 'to_dict') else dict(m)
    return {
        'ticker': d.get('ticker', ''),
        'title': d.get('title', ''),
        'event_ticker': d.get('event_ticker', ''),
        'series_ticker': d.get('series_ticker', ''),
        'yes_bid': d.get('yes_bid', 0),
        'yes_ask': d.get('yes_ask', 0),
        'volume': d.get('volume', 0),
        'close_time': str(d.get('close_time', '')),
        'status': d.get('status', ''),
        'rules_primary': d.get('rules_primary', ''),
    }


# ---------------------------------------------------------------------------
# DFlow/Solana helpers (no auth required)
# ---------------------------------------------------------------------------

DFLOW_METADATA_URL = 'https://dev-prediction-markets-api.dflow.net'


async def _dflow_list_markets(limit: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f'{DFLOW_METADATA_URL}/api/v1/events',
            params={'status': 'active', 'withNestedMarkets': 'true'},
        )
    if resp.status_code != 200:
        raise click.ClickException(
            f'DFlow API returned HTTP {resp.status_code}: {resp.text[:200]}'
        )
    data = resp.json()
    events = (
        data if isinstance(data, list) else data.get('events', data.get('data', []))
    )
    markets: list[dict] = []
    for event in events:
        for mkt in event.get('markets', []):
            if len(markets) >= limit:
                break
            ticker = mkt.get('ticker', mkt.get('marketTicker', ''))
            markets.append(
                {
                    'ticker': ticker,
                    'title': mkt.get('title', mkt.get('question', '')),
                    'event_ticker': event.get(
                        'eventTicker', event.get('event_ticker', '')
                    ),
                    'series_ticker': event.get(
                        'seriesTicker', event.get('series_ticker', '')
                    ),
                    'yes_bid': mkt.get('yesBid', mkt.get('yes_bid', 0)) or 0,
                    'yes_ask': mkt.get('yesAsk', mkt.get('yes_ask', 0)) or 0,
                    'volume': mkt.get('volume', 0),
                    'status': mkt.get('status', 'active'),
                }
            )
        if len(markets) >= limit:
            break
    return markets[:limit]


async def _dflow_search_markets(query: str, limit: int) -> list[dict]:
    all_markets = await _dflow_list_markets(500)
    q = query.lower()
    filtered = [
        m
        for m in all_markets
        if q in m.get('title', '').lower() or q in m.get('ticker', '').lower()
    ]
    return filtered[:limit]


async def _dflow_market_info(market_ticker: str) -> dict | None:
    all_markets = await _dflow_list_markets(1000)
    for m in all_markets:
        if m.get('ticker', '') == market_ticker:
            return m
    return None


def _fmt_dflow_market(m: dict, idx: int) -> str:
    bid = m.get('yes_bid', 0) or 0
    ask = m.get('yes_ask', 0) or 0
    bid_str = f'{bid}' if isinstance(bid, (int, float)) and bid <= 1 else f'{bid}¢'
    ask_str = f'{ask}' if isinstance(ask, (int, float)) and ask <= 1 else f'{ask}¢'
    lines = [f'[{idx}] {m.get("title", "(no title)")}']
    lines.append(f'     Ticker:   {m.get("ticker", "")}')
    lines.append(f'     Event:    {m.get("event_ticker", "")}')
    lines.append(f'     Bid/Ask:  {bid_str} / {ask_str}')
    if m.get('volume'):
        lines.append(f'     Volume:   {m["volume"]}')
    lines.append('     Chain:    Solana (mainnet-beta)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_poly_market(m: dict, idx: int) -> str:
    lines = [f'[{idx}] {m.get("question", "(no question)")}']
    event = m.get('event_title')
    if event and event != m.get('question'):
        lines.append(f'     Event:    {event}')
    lines.append(f'     Market ID: {m.get("id", "")}')
    bid = m.get('best_bid', '')
    ask = m.get('best_ask', '')
    if bid or ask:
        lines.append(f'     Bid/Ask:  {bid} / {ask}')
    if m.get('volume'):
        lines.append(f'     Volume:   {m["volume"]}')
    if m.get('end_date'):
        lines.append(f'     Closes:   {m["end_date"]}')
    return '\n'.join(lines)


def _fmt_kalshi_market(m: dict, idx: int) -> str:
    bid_cents = m.get('yes_bid', 0) or 0
    ask_cents = m.get('yes_ask', 0) or 0
    bid_pct = f'{bid_cents}¢'
    ask_pct = f'{ask_cents}¢'
    lines = [f'[{idx}] {m.get("title", "(no title)")}']
    lines.append(f'     Ticker:   {m.get("ticker", "")}')
    lines.append(f'     Event:    {m.get("event_ticker", "")}')
    lines.append(f'     Bid/Ask:  {bid_pct} / {ask_pct}')
    if m.get('volume'):
        lines.append(f'     Volume:   {m["volume"]}')
    if m.get('close_time'):
        lines.append(f'     Closes:   {m["close_time"]}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Click group + commands
# ---------------------------------------------------------------------------


@click.group()
def market() -> None:
    """Explore prediction markets on Polymarket, Kalshi, and Solana/DFlow."""


@market.command('list')
@click.option(
    '--exchange',
    type=click.Choice(['polymarket', 'kalshi', 'solana']),
    default='polymarket',
    show_default=True,
)
@click.option('--limit', default=20, show_default=True, type=int)
@click.option(
    '--kalshi-api-key-id',
    default=None,
    help='Kalshi API key id (or KALSHI_API_KEY_ID).',
)
@click.option(
    '--kalshi-private-key-path',
    default=None,
    help='Kalshi private key path (or KALSHI_PRIVATE_KEY_PATH).',
)
@click.option('--json', 'as_json', is_flag=True, default=False)
def market_list(
    exchange: str,
    limit: int,
    kalshi_api_key_id: str | None,
    kalshi_private_key_path: str | None,
    as_json: bool,
) -> None:
    """List open markets on a prediction exchange."""
    try:
        if exchange == 'polymarket':
            markets = asyncio.run(_polymarket_list_markets(limit))
        elif exchange == 'kalshi':
            markets = asyncio.run(
                _kalshi_list_markets(limit, kalshi_api_key_id, kalshi_private_key_path)
            )
        else:
            markets = asyncio.run(_dflow_list_markets(limit))
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f'Failed to fetch markets: {exc}') from exc

    if as_json:
        click.echo(
            json.dumps(
                {'exchange': exchange, 'count': len(markets), 'markets': markets}
            )
        )
        return

    if not markets:
        click.echo('No markets found.')
        return

    click.echo(f'Listing {len(markets)} open market(s) on {exchange}:\n')
    for i, m in enumerate(markets, 1):
        if exchange == 'polymarket':
            click.echo(_fmt_poly_market(m, i))
        elif exchange == 'kalshi':
            click.echo(_fmt_kalshi_market(m, i))
        else:
            click.echo(_fmt_dflow_market(m, i))
        click.echo()


@market.command('search')
@click.option(
    '--query', required=True, help='Keyword to search in market title/question.'
)
@click.option(
    '--exchange',
    type=click.Choice(['polymarket', 'kalshi', 'solana']),
    default='polymarket',
    show_default=True,
)
@click.option('--limit', default=20, show_default=True, type=int)
@click.option('--kalshi-api-key-id', default=None)
@click.option('--kalshi-private-key-path', default=None)
@click.option('--json', 'as_json', is_flag=True, default=False)
def market_search(
    query: str,
    exchange: str,
    limit: int,
    kalshi_api_key_id: str | None,
    kalshi_private_key_path: str | None,
    as_json: bool,
) -> None:
    """Search markets by keyword."""
    try:
        if exchange == 'polymarket':
            markets = asyncio.run(_polymarket_search_markets(query, limit))
        elif exchange == 'kalshi':
            markets = asyncio.run(
                _kalshi_search_markets(
                    query, limit, kalshi_api_key_id, kalshi_private_key_path
                )
            )
        else:
            markets = asyncio.run(_dflow_search_markets(query, limit))
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f'Failed to search markets: {exc}') from exc

    if as_json:
        click.echo(
            json.dumps(
                {
                    'exchange': exchange,
                    'query': query,
                    'count': len(markets),
                    'markets': markets,
                }
            )
        )
        return

    if not markets:
        click.echo(f'No markets found matching {query!r}.')
        return

    click.echo(f'Found {len(markets)} market(s) matching {query!r} on {exchange}:\n')
    for i, m in enumerate(markets, 1):
        if exchange == 'polymarket':
            click.echo(_fmt_poly_market(m, i))
        elif exchange == 'kalshi':
            click.echo(_fmt_kalshi_market(m, i))
        else:
            click.echo(_fmt_dflow_market(m, i))
        click.echo()


@market.command('info')
@click.option('--market-id', required=True, help='Market ID or ticker to inspect.')
@click.option(
    '--exchange',
    type=click.Choice(['polymarket', 'kalshi', 'solana']),
    default='polymarket',
    show_default=True,
)
@click.option('--kalshi-api-key-id', default=None)
@click.option('--kalshi-private-key-path', default=None)
@click.option('--json', 'as_json', is_flag=True, default=False)
def market_info(
    market_id: str,
    exchange: str,
    kalshi_api_key_id: str | None,
    kalshi_private_key_path: str | None,
    as_json: bool,
) -> None:
    """Show detailed info and top-of-book for a specific market."""
    try:
        if exchange == 'polymarket':
            info = asyncio.run(_polymarket_market_info(market_id))
        elif exchange == 'kalshi':
            info = asyncio.run(
                _kalshi_market_info(
                    market_id, kalshi_api_key_id, kalshi_private_key_path
                )
            )
        else:
            info = asyncio.run(_dflow_market_info(market_id))
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f'Failed to fetch market info: {exc}') from exc

    if info is None:
        raise click.ClickException(f'Market not found: {market_id}')

    if as_json:
        click.echo(json.dumps({'exchange': exchange, 'market': info}))
        return

    click.echo(f'\nMarket Info ({exchange})')
    click.echo('=' * 60)
    if exchange == 'polymarket':
        click.echo(f'Question:   {info.get("question", "")}')
        click.echo(f'Market ID:  {info.get("id", "")}')
        click.echo(f'Event ID:   {info.get("event_id", "")}')
        click.echo(f'Token ID:   {info.get("token_id", "")}')
        bid = info.get('best_bid', '')
        ask = info.get('best_ask', '')
        click.echo(f'Bid / Ask:  {bid} / {ask}')
        click.echo(f'Volume:     {info.get("volume", "")}')
        click.echo(f'Closes:     {info.get("end_date", "")}')
        click.echo(f'Active:     {info.get("active", True)}')
        desc = info.get('description', '')
        if desc:
            click.echo(f'Description: {desc[:300]}{"…" if len(desc) > 300 else ""}')
    elif exchange == 'kalshi':
        click.echo(f'Title:      {info.get("title", "")}')
        click.echo(f'Ticker:     {info.get("ticker", "")}')
        click.echo(f'Event:      {info.get("event_ticker", "")}')
        bid_c = info.get('yes_bid', 0) or 0
        ask_c = info.get('yes_ask', 0) or 0
        click.echo(f'Bid / Ask:  {bid_c}¢ / {ask_c}¢')
        click.echo(f'Volume:     {info.get("volume", 0)}')
        click.echo(f'Closes:     {info.get("close_time", "")}')
        click.echo(f'Status:     {info.get("status", "")}')
        rules = info.get('rules_primary', '')
        if rules:
            click.echo(f'Rules:      {rules[:300]}{"…" if len(rules) > 300 else ""}')
    else:
        click.echo(f'Title:      {info.get("title", "")}')
        click.echo(f'Ticker:     {info.get("ticker", "")}')
        click.echo(f'Event:      {info.get("event_ticker", "")}')
        click.echo(f'Series:     {info.get("series_ticker", "")}')
        bid = info.get('yes_bid', 0) or 0
        ask = info.get('yes_ask', 0) or 0
        click.echo(f'Bid / Ask:  {bid} / {ask}')
        click.echo(f'Volume:     {info.get("volume", 0)}')
        click.echo(f'Status:     {info.get("status", "")}')
        click.echo('Chain:      Solana (mainnet-beta)')
    click.echo()


# ---------------------------------------------------------------------------
# Polymarket price history
# ---------------------------------------------------------------------------

_INTERVAL_FIDELITY: dict[str, int] = {
    '1d': 1440,
    '6h': 360,
    '1h': 60,
}


async def _polymarket_price_history(
    market_id: str, interval: str, limit: int | None
) -> dict:
    fidelity = _INTERVAL_FIDELITY.get(interval, 1440)

    # Resolve the CLOB token ID from the numeric market ID.
    info = await _polymarket_market_info(market_id)
    if info is None:
        raise click.ClickException(f'Market not found: {market_id}')
    token_id = info.get('token_id', '')
    if not token_id:
        raise click.ClickException(f'No CLOB token ID for market: {market_id}')

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            CLOB_PRICES_HISTORY_URL,
            params={'market': token_id, 'interval': interval, 'fidelity': fidelity},
        )
    if resp.status_code != 200:
        raise click.ClickException(
            f'Polymarket CLOB API returned HTTP {resp.status_code}: {resp.text[:200]}'
        )
    data = resp.json()

    # Response is {"history": [{"t": ..., "p": ...}, ...]}
    raw_history = data.get('history') if isinstance(data, dict) else data
    points: list[dict[str, Any]] = []
    if isinstance(raw_history, list):
        for item in raw_history:
            if isinstance(item, dict) and 't' in item and 'p' in item:
                points.append({'t': item['t'], 'p': item['p']})

    if limit and limit > 0:
        points = points[-limit:]

    first_price = points[0]['p'] if points else None
    last_price = points[-1]['p'] if points else None
    total_move: Any = None
    if first_price is not None and last_price is not None:
        try:
            total_move = round(float(last_price) - float(first_price), 6)
        except (TypeError, ValueError):
            total_move = None

    return {
        'market_id': market_id,
        'token_id': token_id,
        'interval': interval,
        'points': len(points),
        'series': points,
        'first_price': first_price,
        'last_price': last_price,
        'total_move': total_move,
    }


@market.command('history')
@click.option('--market-id', required=True, help='Polymarket market ID.')
@click.option(
    '--interval',
    type=click.Choice(['1d', '6h', '1h']),
    default='1d',
    show_default=True,
    help='Candle interval.',
)
@click.option(
    '--limit',
    default=None,
    type=int,
    help='Take only the last N price points.',
)
@click.option('--json', 'as_json', is_flag=True, default=False)
def market_history(
    market_id: str,
    interval: str,
    limit: int | None,
    as_json: bool,
) -> None:
    """Fetch a market's price history from the Polymarket Gamma API (Polymarket only)."""
    try:
        result = asyncio.run(_polymarket_price_history(market_id, interval, limit))
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f'Failed to fetch price history: {exc}') from exc

    if as_json:
        click.echo(json.dumps(result))
        return

    click.echo(f'\nPrice History — market {market_id} ({interval})')
    click.echo('=' * 60)
    click.echo(f'Points:      {result["points"]}')
    click.echo(f'First price: {result["first_price"]}')
    click.echo(f'Last price:  {result["last_price"]}')
    click.echo(f'Total move:  {result["total_move"]}')
    if result['series']:
        click.echo('\nLast 5 points:')
        for pt in result['series'][-5:]:
            click.echo(f'  t={pt["t"]}  p={pt["p"]}')
    click.echo()
