"""Model-informed position sizing with Wang Transform edge estimation.

Three-tier sizing architecture:

1. **Quant baseline** — Kelly criterion with model-derived edge from
   the Wang pricing model. The edge is the difference between the
   model's physical probability and the market price, adjusted for
   fees. This is always available and never fails.

2. **Confidence scaling** — the quant size is scaled by the model's
   confidence estimate, which incorporates calibration quality,
   data completeness, and volume tier. High-confidence estimates
   get full Kelly; low-confidence gets fractional Kelly.

3. **Volume-tier gating** — implements the key insight from Yang (2026)
   that very-high-volume markets have lambda ≈ 0 (premium already
   competed away). The sizing system automatically reduces or
   eliminates positions in these markets, focusing capital on
   medium-liquidity markets where the model has predictive power.

The system is designed to be fail-safe: if any tier fails, the
output degrades gracefully to the next available tier.

Inspired by the three-tier LLM-enhanced sizing approach in
agent-native prediction market systems, adapted for model-driven
alpha generation with the Wang Transform pricing framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from oracle3.pricing.fair_value import FairValueEstimate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SizingResult:
    """Output of the position sizing computation."""

    trade_size: Decimal  # dollar amount to trade
    kelly_fraction: float  # raw Kelly fraction
    confidence_scale: float  # confidence multiplier applied
    edge: float  # model-derived edge (before fees)
    net_edge: float  # edge after round-trip fees
    side: str  # 'YES' or 'NO'
    reasoning: str  # human-readable explanation
    volume_tier: str  # from the fair value estimate
    is_alpha_target: bool  # whether this is in the sweet spot


class ModelInformedSizer:
    """Position sizing using Wang Transform edge estimation.

    Parameters
    ----------
    max_kelly_fraction:
        Maximum Kelly fraction (default 0.15 = conservative).
        Full Kelly is mathematically optimal but has extreme variance;
        quarter-Kelly (0.25) is common in practice. Default 0.15
        is even more conservative.
    min_edge:
        Minimum net edge (after fees) to justify a trade.
        Default 0.005 = 0.5 cents per contract.
    fee_rate:
        Per-side fee rate (default 0.005 = 0.5%).
    max_trade_size:
        Absolute maximum trade size in dollars (default $100).
    min_trade_size:
        Minimum trade size in dollars (default $5).
    confidence_floor:
        Minimum confidence to allow any trade (default 0.15).
    skip_very_high_volume:
        If True (default), skip trades on very-high-volume markets
        where the premium is already competed away.
    available_capital:
        Total available capital for sizing (default $1000).
    """

    def __init__(
        self,
        max_kelly_fraction: float = 0.15,
        min_edge: float = 0.005,
        fee_rate: float = 0.005,
        max_trade_size: Decimal = Decimal('100'),
        min_trade_size: Decimal = Decimal('5'),
        confidence_floor: float = 0.15,
        skip_very_high_volume: bool = True,
        available_capital: Decimal = Decimal('1000'),
    ) -> None:
        self._max_kelly = max_kelly_fraction
        self._min_edge = min_edge
        self._fee_rate = fee_rate
        self._max_trade = max_trade_size
        self._min_trade = min_trade_size
        self._conf_floor = confidence_floor
        self._skip_vhv = skip_very_high_volume
        self._capital = available_capital

    def compute_size(
        self,
        estimate: FairValueEstimate,
        available_capital: Decimal | None = None,
        win_rate: float | None = None,
    ) -> SizingResult:
        """Compute optimal trade size from a fair value estimate.

        Parameters
        ----------
        estimate: fair value estimate from the pricing model
        available_capital: override available capital
        win_rate: optional empirical win rate for Kelly blending
        """
        capital = available_capital or self._capital

        # Gate 1: volume tier
        if self._skip_vhv and estimate.volume_tier == 'very_high':
            return SizingResult(
                trade_size=Decimal('0'),
                kelly_fraction=0.0,
                confidence_scale=0.0,
                edge=0.0,
                net_edge=0.0,
                side='NONE',
                reasoning='Very-high-volume market: premium already competed away (lambda≈0)',
                volume_tier=estimate.volume_tier,
                is_alpha_target=False,
            )

        # Gate 2: confidence
        if estimate.confidence < self._conf_floor:
            return SizingResult(
                trade_size=Decimal('0'),
                kelly_fraction=0.0,
                confidence_scale=estimate.confidence,
                edge=0.0,
                net_edge=0.0,
                side='NONE',
                reasoning=f'Confidence {estimate.confidence:.2f} below floor {self._conf_floor}',
                volume_tier=estimate.volume_tier,
                is_alpha_target=estimate.is_premium_alpha_target,
            )

        # Compute edge
        edge = estimate.risk_premium  # positive = overpriced
        fee_cost = 2 * self._fee_rate
        net_edge = abs(edge) - fee_cost

        if net_edge < self._min_edge:
            return SizingResult(
                trade_size=Decimal('0'),
                kelly_fraction=0.0,
                confidence_scale=estimate.confidence,
                edge=abs(edge),
                net_edge=net_edge,
                side='YES' if edge < 0 else 'NO',
                reasoning=f'Net edge {net_edge:.4f} below minimum {self._min_edge}',
                volume_tier=estimate.volume_tier,
                is_alpha_target=estimate.is_premium_alpha_target,
            )

        # Determine side
        if edge > 0:
            # Overpriced: buy NO (short the event)
            side = 'NO'
            p_model = estimate.fair_value
            p_market = estimate.market_price
            # Kelly for NO bet: (p_mkt - p*) / p_mkt after fees
            kelly_raw = (p_market - p_model - fee_cost) / p_market if p_market > 0.01 else 0.0
        else:
            # Underpriced: buy YES
            side = 'YES'
            p_model = estimate.fair_value
            p_market = estimate.market_price
            # Kelly for YES bet: (p* - p_mkt) / (1 - p_mkt) after fees
            denom = 1.0 - p_market
            kelly_raw = (p_model - p_market - fee_cost) / denom if denom > 0.01 else 0.0

        # Blend with empirical win rate if available
        if win_rate is not None and win_rate > 0:
            empirical_kelly = (win_rate * (1 / p_market - 1) - (1 - win_rate)) if p_market > 0.01 else 0.0
            kelly_raw = min(kelly_raw, max(empirical_kelly, 0.0))

        # Cap Kelly fraction
        kelly_capped = max(0.0, min(self._max_kelly, kelly_raw))

        # Apply confidence scaling
        confidence_scale = estimate.confidence
        kelly_final = kelly_capped * confidence_scale

        # Convert to dollar amount
        trade_size = Decimal(str(kelly_final)) * capital
        trade_size = min(trade_size, self._max_trade)
        if trade_size < self._min_trade:
            trade_size = Decimal('0')

        return SizingResult(
            trade_size=trade_size,
            kelly_fraction=kelly_final,
            confidence_scale=confidence_scale,
            edge=abs(edge),
            net_edge=net_edge,
            side=side,
            reasoning=(
                f'{side}: edge={abs(edge):.4f}, net={net_edge:.4f}, '
                f'kelly={kelly_capped:.3f}×conf={confidence_scale:.2f}={kelly_final:.3f}, '
                f'tier={estimate.volume_tier}'
            ),
            volume_tier=estimate.volume_tier,
            is_alpha_target=estimate.is_premium_alpha_target,
        )

    def update_capital(self, new_capital: Decimal) -> None:
        """Update available capital (e.g., after P&L changes)."""
        self._capital = new_capital
