"""Contract feature extraction and quality scoring.

Extracts market microstructure features from prediction market contracts
that modulate the risk premium and determine tradability:

- **Volume** — proxy for price discovery quality; higher volume means
  prices are closer to fundamental value.
- **Duration** — time to resolution; longer-lived contracts carry larger
  risk premiums because more uncertainty persists.
- **Spread** — bid-ask spread; wider spreads indicate less competition
  and more room for mispricing.
- **Extremity** — distance from 0.5; events near certainty/impossibility
  have different premium structures than toss-ups.
- **Book depth** — cumulative size available near the best bid/ask;
  measures execution capacity.

These features serve three consumers:

1. FairValueEstimator: adjusts the distortion parameter per contract.
2. Strategy selection: filters for high-quality opportunities.
3. Position sizing: scales exposure by contract quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ContractFeatures:
    """Extracted microstructure features for a single contract.

    All fields are optional to handle partial data availability.
    Strategies and estimators should check for None before using.
    """

    # Identifiers
    ticker_symbol: str = ''
    platform: str = ''
    category: str = 'default'

    # Core features
    volume: float | None = None  # total traded volume (USD or shares)
    duration_days: float | None = None  # days from listing to resolution
    elapsed_fraction: float | None = None  # fraction of lifetime elapsed
    spread: float | None = None  # current bid-ask spread
    mid_price: float | None = None  # current mid price
    best_bid: float | None = None
    best_ask: float | None = None

    # Derived features (computed by ContractScorer)
    log_volume: float | None = None
    log_duration: float | None = None
    extremity: float | None = None  # |mid_price - 0.5|
    book_depth_bid: float | None = None  # total size within 5c of best bid
    book_depth_ask: float | None = None

    # Timestamps
    created_at: datetime | None = None
    resolution_at: datetime | None = None
    observed_at: datetime | None = None


@dataclass(frozen=True)
class ContractScore:
    """Composite quality score for a contract."""

    tradability: float  # 0-1 composite score
    liquidity: float  # 0-1 liquidity sub-score
    premium_magnitude: float  # expected premium magnitude (higher = more alpha)
    confidence: float  # confidence in the score itself
    features: ContractFeatures
    breakdown: dict[str, float] = field(default_factory=dict)


class ContractScorer:
    """Score contracts for tradability and premium estimation.

    Computes a composite tradability score from microstructure features.
    The score combines:
    - Liquidity (volume + spread + book depth)
    - Price discovery quality (volume-adjusted spread)
    - Premium opportunity (duration + extremity)

    Parameters
    ----------
    volume_weight:
        Weight for volume in composite score (default 0.30).
    spread_weight:
        Weight for spread in composite score (default 0.25).
    depth_weight:
        Weight for book depth in composite score (default 0.15).
    duration_weight:
        Weight for duration in composite score (default 0.15).
    extremity_weight:
        Weight for extremity in composite score (default 0.15).
    min_volume:
        Minimum volume for a contract to be considered tradeable.
    max_spread:
        Maximum spread for a contract to be considered tradeable.
    """

    def __init__(
        self,
        volume_weight: float = 0.30,
        spread_weight: float = 0.25,
        depth_weight: float = 0.15,
        duration_weight: float = 0.15,
        extremity_weight: float = 0.15,
        min_volume: float = 1000.0,
        max_spread: float = 0.15,
    ) -> None:
        self._weights = {
            'volume': volume_weight,
            'spread': spread_weight,
            'depth': depth_weight,
            'duration': duration_weight,
            'extremity': extremity_weight,
        }
        self._min_volume = min_volume
        self._max_spread = max_spread

    def extract_features(
        self,
        ticker_symbol: str = '',
        platform: str = '',
        category: str = 'default',
        volume: float | None = None,
        duration_days: float | None = None,
        elapsed_fraction: float | None = None,
        best_bid: float | None = None,
        best_ask: float | None = None,
        book_depth_bid: float | None = None,
        book_depth_ask: float | None = None,
        created_at: datetime | None = None,
        resolution_at: datetime | None = None,
    ) -> ContractFeatures:
        """Extract and compute contract features from raw market data."""
        mid_price: float | None = None
        spread: float | None = None
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid

        extremity: float | None = None
        if mid_price is not None:
            extremity = abs(mid_price - 0.5)

        log_vol: float | None = None
        if volume is not None:
            log_vol = math.log(volume + 1.0)

        log_dur: float | None = None
        if duration_days is not None and duration_days > 0:
            log_dur = math.log(duration_days + 1.0)

        return ContractFeatures(
            ticker_symbol=ticker_symbol,
            platform=platform,
            category=category,
            volume=volume,
            duration_days=duration_days,
            elapsed_fraction=elapsed_fraction,
            spread=spread,
            mid_price=mid_price,
            best_bid=best_bid,
            best_ask=best_ask,
            log_volume=log_vol,
            log_duration=log_dur,
            extremity=extremity,
            book_depth_bid=book_depth_bid,
            book_depth_ask=book_depth_ask,
            created_at=created_at,
            resolution_at=resolution_at,
            observed_at=datetime.now(),
        )

    def score(self, features: ContractFeatures) -> ContractScore:
        """Compute composite quality score for a contract."""
        breakdown: dict[str, float] = {}

        # Volume sub-score: sigmoid on log volume
        vol_score = 0.5  # default when unknown
        if features.log_volume is not None:
            # Sigmoid: maps log(volume) to (0, 1), centered at log(10000) ~ 9.2
            vol_score = 1.0 / (1.0 + math.exp(-(features.log_volume - 9.2) / 2.0))
        breakdown['volume'] = vol_score

        # Spread sub-score: narrower is better
        spread_score = 0.5
        if features.spread is not None:
            # 0 spread -> 1.0, max_spread -> ~0.0
            spread_score = max(0.0, 1.0 - features.spread / self._max_spread)
        breakdown['spread'] = spread_score

        # Depth sub-score: deeper is better
        depth_score = 0.5
        if features.book_depth_bid is not None and features.book_depth_ask is not None:
            total_depth = features.book_depth_bid + features.book_depth_ask
            depth_score = 1.0 / (1.0 + math.exp(-(math.log(total_depth + 1) - 6.0)))
        breakdown['depth'] = depth_score

        # Duration sub-score: moderate duration is best (too short = no alpha,
        # too long = too much noise)
        dur_score = 0.5
        if features.log_duration is not None:
            # Bell curve centered at ~30 days (log(30) ~ 3.4)
            dur_score = math.exp(-0.5 * ((features.log_duration - 3.4) / 1.5) ** 2)
        breakdown['duration'] = dur_score

        # Extremity sub-score: moderate extremity (0.1-0.3) is most tradeable
        ext_score = 0.5
        if features.extremity is not None:
            # Bell curve centered at 0.2
            ext_score = math.exp(-0.5 * ((features.extremity - 0.2) / 0.15) ** 2)
        breakdown['extremity'] = ext_score

        # Composite tradability
        tradability = sum(
            self._weights[k] * breakdown[k] for k in self._weights
        )

        # Liquidity sub-score (volume + spread + depth)
        liquidity = (
            0.45 * breakdown['volume']
            + 0.35 * breakdown['spread']
            + 0.20 * breakdown['depth']
        )

        # Premium magnitude estimate: higher for longer duration, moderate prices
        premium_magnitude = 0.5
        if features.log_duration is not None and features.extremity is not None:
            # Longer duration + closer to 0.5 = larger absolute premium
            premium_magnitude = min(
                1.0,
                (features.log_duration / 5.0) * (1.0 - features.extremity),
            )

        # Confidence: higher when more features are available
        n_available = sum(
            1
            for v in [
                features.volume,
                features.spread,
                features.duration_days,
                features.book_depth_bid,
            ]
            if v is not None
        )
        confidence = n_available / 4.0

        return ContractScore(
            tradability=tradability,
            liquidity=liquidity,
            premium_magnitude=premium_magnitude,
            confidence=confidence,
            features=features,
            breakdown=breakdown,
        )

    def is_tradeable(self, features: ContractFeatures) -> bool:
        """Quick check: does this contract meet minimum tradability criteria?"""
        if features.volume is not None and features.volume < self._min_volume:
            return False
        if features.spread is not None and features.spread > self._max_spread:
            return False
        return True
