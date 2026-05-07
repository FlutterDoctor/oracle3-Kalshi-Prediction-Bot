"""Edge-weighted capital allocation across multiple strategies.

Allocates a capital budget across N concurrent strategies based on
risk-adjusted performance with exponential time decay, reserve
capital requirements, and per-strategy caps.

The allocation score for each strategy is:

    score_i = (pnl_i / max(|drawdown_i|, epsilon)) * exp(-ln(2) * age_days / half_life)

This rewards strategies that have:
- High P&L relative to their worst drawdown (Sharpe-like risk-adjustment)
- Recent performance (exponential time decay with configurable half-life)

Safety features:
- Reserve capital (10% default) is never allocated
- Per-strategy caps (40% default) prevent concentration
- Minimum budgets ensure all active strategies can place at least 1 trade
- Graceful degradation: equal allocation if no performance data exists

Inspired by edge-weighted capital allocation approaches in
multi-strategy prediction market systems, enhanced with
Wang-model awareness for premium-alpha targeting.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyPerformance:
    """Performance snapshot for a strategy."""

    strategy_id: str
    cumulative_pnl: float = 0.0
    max_drawdown: float = 0.0  # negative value
    n_trades: int = 0
    win_rate: float = 0.5
    age_days: float = 0.0
    is_premium_alpha: bool = False  # True if targeting Wang premium
    is_active: bool = True


@dataclass(frozen=True)
class AllocationResult:
    """Capital allocation output."""

    allocations: dict[str, Decimal]  # strategy_id -> budget
    scores: dict[str, float]  # strategy_id -> raw score
    total_allocated: Decimal
    reserve: Decimal
    method: str  # 'performance' or 'equal'


class EdgeWeightedAllocator:
    """Allocate capital across strategies by risk-adjusted edge.

    Parameters
    ----------
    total_capital:
        Total portfolio capital.
    reserve_fraction:
        Fraction held as reserve (default 0.10).
    max_strategy_fraction:
        Maximum fraction for any single strategy (default 0.40).
    min_budget:
        Minimum budget for any active strategy (default $10).
    time_decay_half_life:
        Half-life for performance time decay in days (default 30).
    premium_alpha_bonus:
        Allocation bonus multiplier for premium-alpha strategies
        (default 1.3 = 30% bonus). These are strategies targeting
        the medium-liquidity sweet spot identified by Yang (2026).
    """

    def __init__(
        self,
        total_capital: Decimal = Decimal('1000'),
        reserve_fraction: float = 0.10,
        max_strategy_fraction: float = 0.40,
        min_budget: Decimal = Decimal('10'),
        time_decay_half_life: float = 30.0,
        premium_alpha_bonus: float = 1.3,
    ) -> None:
        self._capital = total_capital
        self._reserve_frac = reserve_fraction
        self._max_frac = max_strategy_fraction
        self._min_budget = min_budget
        self._half_life = time_decay_half_life
        self._premium_bonus = premium_alpha_bonus

    def allocate(
        self,
        strategies: list[StrategyPerformance],
    ) -> AllocationResult:
        """Compute capital allocation for all active strategies.

        Parameters
        ----------
        strategies: performance snapshots for each strategy.

        Returns
        -------
        AllocationResult with per-strategy budgets.
        """
        active = [s for s in strategies if s.is_active]
        if not active:
            return AllocationResult(
                allocations={},
                scores={},
                total_allocated=Decimal('0'),
                reserve=self._capital,
                method='empty',
            )

        reserve = self._capital * Decimal(str(self._reserve_frac))
        allocatable = self._capital - reserve
        max_per_strategy = allocatable * Decimal(str(self._max_frac))

        # Compute scores
        scores: dict[str, float] = {}
        for s in active:
            scores[s.strategy_id] = self._score(s)

        # Check if we have meaningful performance data
        has_data = any(s.n_trades > 0 for s in active)

        if not has_data:
            # Equal allocation
            per_strategy = allocatable / Decimal(str(len(active)))
            per_strategy = min(per_strategy, max_per_strategy)
            allocations = {
                s.strategy_id: max(self._min_budget, per_strategy)
                for s in active
            }
            total = sum(allocations.values(), Decimal('0'))
            return AllocationResult(
                allocations=allocations,
                scores=scores,
                total_allocated=total,
                reserve=self._capital - total,
                method='equal',
            )

        # Performance-weighted allocation
        total_score = sum(max(0, s) for s in scores.values())
        if total_score <= 0:
            # All negative: equal allocation with minimum budgets
            per_strategy = self._min_budget
            allocations = {s.strategy_id: per_strategy for s in active}
            total = sum(allocations.values(), Decimal('0'))
            return AllocationResult(
                allocations=allocations,
                scores=scores,
                total_allocated=total,
                reserve=self._capital - total,
                method='equal_min',
            )

        # Proportional allocation
        allocations = {}
        for s in active:
            raw_score = max(0.0, scores[s.strategy_id])
            fraction = raw_score / total_score
            budget = allocatable * Decimal(str(fraction))
            budget = min(budget, max_per_strategy)
            budget = max(budget, self._min_budget)
            allocations[s.strategy_id] = budget

        total = sum(allocations.values(), Decimal('0'))
        return AllocationResult(
            allocations=allocations,
            scores=scores,
            total_allocated=total,
            reserve=self._capital - total,
            method='performance',
        )

    def update_capital(self, new_capital: Decimal) -> None:
        self._capital = new_capital

    # ── Internal ──────────────────────────────────────────────────────────

    def _score(self, s: StrategyPerformance) -> float:
        """Compute risk-adjusted, time-decayed score for a strategy."""
        if s.n_trades == 0:
            return 0.0

        # Risk-adjusted performance: PnL / max(|drawdown|, epsilon)
        dd = max(abs(s.max_drawdown), 0.01)
        risk_adj = s.cumulative_pnl / dd

        # Time decay: more recent performance counts more
        decay = math.exp(-math.log(2) * s.age_days / self._half_life)

        # Win rate bonus (above 50% is good)
        wr_bonus = 1.0 + max(0, s.win_rate - 0.5)

        score = risk_adj * decay * wr_bonus

        # Premium alpha bonus for strategies targeting the sweet spot
        if s.is_premium_alpha:
            score *= self._premium_bonus

        return score
