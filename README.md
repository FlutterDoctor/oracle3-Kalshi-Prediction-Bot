<p align="center">
  <img src="assets/oracle3-banner.svg" alt="Oracle3" width="600">
</p>

<h1 align="center">Oracle3</h1>
<p align="center">
  <strong>Autonomous On-Chain Trading Agent for Prediction Markets</strong>
</p>

<p align="center">
  <a href="https://github.com/YichengYang-Ethan/oracle3/actions"><img src="https://github.com/YichengYang-Ethan/oracle3/actions/workflows/pytest.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/YichengYang-Ethan/oracle3/actions"><img src="https://github.com/YichengYang-Ethan/oracle3/actions/workflows/ruff.yml/badge.svg" alt="Lint"></a>
  <a href="https://github.com/YichengYang-Ethan/oracle3/actions"><img src="https://github.com/YichengYang-Ethan/oracle3/actions/workflows/mypy.yml/badge.svg" alt="Type Check"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/solana-mainnet--beta-9945FF?logo=solana&logoColor=white" alt="Solana">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  <a href="https://github.com/YichengYang-Ethan/oracle3/releases"><img src="https://img.shields.io/github/v/release/YichengYang-Ethan/oracle3?color=orange" alt="Release"></a>
</p>

<p align="center">
  <em>A fully autonomous agent that reads on-chain data, exploits structural mispricings across prediction markets, signs Solana transactions, and manages risk end-to-end — no human in the loop.</em>
</p>

---

<p align="center">
  <img src="assets/dashboard-live.png" alt="Oracle3 Live Dashboard" width="800">
  <br>
  <sub>Live trading dashboard — real-time equity curve, execution pipeline, 8 on-chain agent capabilities</sub>
</p>

## What's New

| Date | Update |
|------|--------|
| **2026-03-28** | v1.1.0 — Wang Transform pricing engine (Yang 2026), MLE estimator, model Greeks, Kelly sizing, correlation-aware risk, edge-weighted allocation |
| **2026-03-09** | v1.0.0 released — 8 arbitrage strategies, market relation graph, SpreadExecutor, engine control |
| **2026-03-04** | Live trading dashboard with real-time equity curve and 8 on-chain feature cards |
| **2026-02-28** | Initial release — trading engine, multi-exchange support, AI agent strategies |

## Key Features

- **Wang Transform pricing engine** — empirically calibrated from 291K+ contracts across 6 platforms (Yang 2026), decomposes market prices into physical probability + risk premium with exact coefficients, MLE estimator, and model Greeks
- **10 trading strategies** — 6 constraint-based arb + 2 statistical arb + 2 model-driven (fair value divergence, premium lifecycle decay)
- **Model-informed trading infrastructure** — Kelly sizing with Wang-derived edge, edge-weighted multi-strategy allocation, correlation-aware risk management
- **Multi-exchange** — Solana/DFlow, Polymarket, Kalshi with cross-platform arbitrage
- **AI + Quant hybrid** — LLM agent strategies (OpenAI Agents SDK + LiteLLM) alongside adaptive quantitative strategies
- **On-chain execution** — native Solana tx signing, Jito MEV protection, flash loan arbitrage, atomic multi-leg trades
- **Safe multi-leg** — SpreadExecutor with automatic LIFO unwind on partial fills; no naked positions
- **Market relation graph** — persistent knowledge graph with quantitative validation (cointegration, ADF, hedge ratio, distortion calibration)
- **Dual-layer risk** — local limits + Solana `simulateTransaction` pre-flight check
- **On-chain audit trail** — every trade logged via Memo program + agent reputation scoring (0–100)
- **Engine control** — Unix socket runtime control (pause/resume/killswitch) without process restart
- **Live dashboard** — web UI with equity curve, 8 feature cards, execution pipeline + terminal TUI

## Architecture

```
                    ┌───────────────────────────────────────────────┐
                    │              Oracle3 CLI                      │
                    │   paper │ live │ blinks │ monitor │ control   │
                    └───────────────────┬───────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
 ┌────────▼────────┐         ┌──────────▼──────────┐       ┌─────────▼─────────┐
 │  Agent Strategy │         │   Quant Strategy    │       │ Arbitrage Suite   │
 │  (LLM + Tools)  │         │  (Momentum/MR/MM)   │       │ (8 strategies)    │
 └────────┬────────┘         └──────────┬──────────┘       └─────────┬─────────┘
          │                             │                             │
          │               ┌─────────────▼──────────────┐              │
          │               │  Probabilistic Fair Value   │              │
          │               │  Distortion · Calibrator ·  │              │
          │               │  Premium Tracker · Scorer   │              │
          │               └─────────────┬──────────────┘              │
          └─────────────────────────────┼─────────────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────────────┐
                    │             Trading Engine                    │
                    │  Event Loop • Risk Manager • Position Tracker │
                    │  SpreadExecutor • Control Server • Registry   │
                    └───────────────────┬───────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
 ┌────────▼────────┐         ┌──────────▼──────────┐       ┌─────────▼─────────┐
 │  Solana / DFlow │         │    Polymarket       │       │      Kalshi       │
 │  SPL Tokens     │         │    CLOB API         │       │    REST API       │
 └────────┬────────┘         └─────────────────────┘       └───────────────────┘
          │
 ┌────────▼────────┐
 │  Solana Blinks  │  ← Shareable trade URLs
 │  On-Chain Logs  │  ← Memo program audit trail
 │  Jito Bundles   │  ← MEV protection
 └─────────────────┘
```

## Arbitrage Strategies

### Constraint-Based

| Strategy | Invariant | Entry Signal |
|----------|-----------|--------------|
| **Cross-Market Arb** | Same event, same price | Price gap across DFlow/Polymarket/Kalshi exceeds fees |
| **Exclusivity Arb** | P(A) + P(B) ≤ 1 | Mutually exclusive event prices sum > 1 |
| **Implication Arb** | P(A) ≤ P(B) | Implied event priced higher than parent |
| **Conditional Arb** | P(A\|B) ∈ [L, U] | Conditional probability bounds violated |
| **Event Sum Arb** | Σ P(outcome) = 1 | Outcome prices don't sum to 1 |
| **Structural Arb** | P(A) = β·P(B) + α | Price deviates from calibrated model |

### Statistical

| Strategy | Method | Edge Source |
|----------|--------|------------|
| **Cointegration Spread** | Self-calibrating z-score bands | Mean-reverting spread between cointegrated markets |
| **Lead-Lag** | Rolling cross-correlation | Follower market lags leader's price movements |

Every strategy uses a position state machine, fee-aware edge calculation, cooldown windows, and fill-guarded transitions. See [Architecture docs](https://yichengyang-ethan.github.io/oracle3/) for details.

## Wang Transform Pricing Engine

Oracle3 integrates the pricing framework from [Yang (2026)](https://github.com/YichengYang-Ethan/prediction-market-pricing), *"Pricing Prediction Markets: Risk Premiums, Incomplete Markets, and a Decomposition Framework"*, which decomposes market prices into **physical probability** and **risk premium** via a probit-space distortion calibrated on 291,309 contracts across 6 platforms:

```
p_market = Φ(Φ⁻¹(p_physical) + λ)    where λ is estimated via MLE
```

### Empirical Results (from the paper)

| Platform | N | λ | SE | Interpretation |
|----------|---|---|---|----------------|
| Polymarket | 13,738 | 0.166 | 0.011 | A 50% event trades at ~57¢ (7¢ risk premium) |
| Kalshi | 271,699 | 0.187 | 0.003 | Slightly higher premiums than Polymarket |
| Metaculus | 1,845 | 0.287 | 0.033 | Forecasting platform, higher premiums |
| Manifold | 1,681 | **-0.218** | 0.032 | Play-money: *underpriced* (natural negative control) |

**Critical alpha insight**: very-high-volume markets (>$10K) have **λ ≈ 0** — the premium is already competed away. The alpha lives in **medium-liquidity markets** ($500–$10K) where λ = 0.25–0.35.

### Model-Driven Strategies

| Strategy | Alpha Source | Entry Signal |
|----------|------------|--------------|
| **Fair Value Divergence** | Market price deviates from model fair value | net_edge > min_edge with Kelly sizing |
| **Premium Decay** | Risk premiums shrink over contract lifetime (half-life: 33–77%) | Premium > threshold AND early lifecycle |

### Pricing Architecture

| Component | Purpose |
|-----------|---------|
| **Wang MLE** | Batch MLE with analytic gradients, sandwich SEs, clustered SEs — the paper's core estimator |
| **Distortion Functions** | Probit, dual power, proportional hazard — 3 parameterised families |
| **Fair Value Estimator** | Uses exact empirical coefficients: λ_i = 0.259 − 0.072·ln(V) + 0.143·ln(D) − 0.477·\|p−0.5\| |
| **Online Calibrator** | Hybrid batch MLE + streaming EWMA with hierarchical category shrinkage |
| **Model Greeks** | dp/dλ, Kelly fraction, edge decay rate — for sizing and risk management |
| **Premium Tracker** | Time-varying decay: γ₁τ + γ₂τ², optimal entry tau estimation |
| **Kelly Sizer** | Model-informed Kelly with volume-tier gating and confidence scaling |
| **Capital Allocator** | Edge-weighted multi-strategy allocation with time decay and premium-alpha bonus |
| **Correlation Risk** | EWMA correlation matrix, effective exposure limits, concentration gating |

The **favorite-longshot bias** — longshots are overpriced by a larger multiple than favorites — emerges as a *theorem* from the probit distortion (Theorem 1 in Yang 2026), not an empirical anomaly.

> **Theoretical foundation**: Yang, Y. (2026). *Pricing Prediction Markets: Risk Premiums, Incomplete Markets, and a Decomposition Framework.* Working Paper, UIUC. [[Replication package]](https://github.com/YichengYang-Ethan/prediction-market-pricing)

```python
from oracle3.pricing import WangMLE, FairValueEstimator, ProbitDistortion

# 1. Estimate lambda from historical data
mle = WangMLE()
result = mle.fit(prices=market_prices, outcomes=resolved_outcomes)
print(result.summary_table())  # lambda, SEs, AIC/BIC, pseudo-R²

# 2. Real-time fair value with empirical coefficients
estimator = FairValueEstimator()
est = estimator.estimate(market_price=0.57, category='crypto', volume=5000)
print(f'Physical prob: {est.fair_value:.3f}, premium: {est.risk_premium:.4f}')
# → Physical prob: 0.472, premium: 0.098 (crypto has lambda ≈ 0.25)

# 3. Kelly-optimal position sizing
from oracle3.pricing.greeks import kelly_fraction
kelly = kelly_fraction(p_market=0.57, lam=0.25, fee_rate=0.005)
# → kelly ≈ 0.08 (8% of bankroll on this edge)
```

## Quick Start

```bash
git clone https://github.com/YichengYang-Ethan/oracle3.git
cd oracle3
poetry install
```

```bash
# Browse markets
oracle3 market list --exchange solana --limit 10

# Paper trading with live dashboard
oracle3 dashboard --exchange solana \
  --strategy-ref oracle3.strategy.contrib.adaptive_onchain_strategy:AdaptiveOnChainStrategy \
  --initial-capital 10000

# Run the interactive demo
./demo.sh
```

Open **`http://localhost:3000/live`** for the live dashboard. See [Quick Start Guide](https://yichengyang-ethan.github.io/oracle3/CLI_QUICK_START/) for more commands.

## Why Oracle3?

| Feature | Oracle3 | [Polymarket/agents](https://github.com/Polymarket/agents) | [freqtrade](https://github.com/freqtrade/freqtrade) |
|---------|---------|-------------------|-----------|
| Prediction markets | 3 exchanges | Polymarket only | Crypto spot/futures |
| Constraint arbitrage | 8 strategies with formal invariants | None | None |
| Fair value pricing | Wang MLE calibrated on 291K contracts (Yang 2026) | None | None |
| Model-informed sizing | Kelly with Greeks, correlation risk, capital allocation | None | None |
| On-chain atomic execution | Solana native | Off-chain | Off-chain |
| LLM agent + quant hybrid | OpenAI Agents SDK + LiteLLM | RAG pipeline | FreqAI (ML) |
| Multi-leg auto-unwind | SpreadExecutor (LIFO) | None | None |
| MEV protection | Jito Bundles | N/A | N/A |
| On-chain audit trail | Solana Memo program | None | None |

## Motivation

On-chain autonomous agents represent the next evolution of crypto infrastructure. Oracle3 is a reference architecture for unifying LLM reasoning, quantitative signals, constraint-based arbitrage, and on-chain primitives into a single autonomous system. Prediction markets are the sharpest testbed — discrete outcomes, transparent order books, and probability axioms that create mathematically guaranteed arbitrage when prices misbehave.

<p align="center">
  <a href="https://solana.com"><img src="https://img.shields.io/badge/Solana-9945FF?style=for-the-badge&logo=solana&logoColor=white" alt="Solana"></a>
  <a href="https://polymarket.com"><img src="https://img.shields.io/badge/Polymarket-0052FF?style=for-the-badge&logoColor=white" alt="Polymarket"></a>
  <a href="https://kalshi.com"><img src="https://img.shields.io/badge/Kalshi-000000?style=for-the-badge&logoColor=white" alt="Kalshi"></a>
  <a href="https://openai.com"><img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"></a>
  <a href="https://jito.network"><img src="https://img.shields.io/badge/Jito-00C7B7?style=for-the-badge&logoColor=white" alt="Jito"></a>
</p>

## Community

<p align="center">
  <a href="https://github.com/YichengYang-Ethan/oracle3/discussions"><img src="https://img.shields.io/badge/Discussions-181717?style=for-the-badge&logo=github&logoColor=white" alt="Discussions"></a>
  <a href="https://github.com/YichengYang-Ethan/oracle3/issues"><img src="https://img.shields.io/badge/Issues-181717?style=for-the-badge&logo=github&logoColor=white" alt="Issues"></a>
</p>

[Discussions](https://github.com/YichengYang-Ethan/oracle3/discussions) · [Issues](https://github.com/YichengYang-Ethan/oracle3/issues) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Docs](https://yichengyang-ethan.github.io/oracle3/)

<a href="https://github.com/YichengYang-Ethan/oracle3/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=YichengYang-Ethan/oracle3" alt="Contributors" />
</a>

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>This software is for research and educational purposes. Trading on prediction markets involves financial risk.</sub>
</p>
