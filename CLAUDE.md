# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Oracle3 is an autonomous prediction-market trading agent trading across Kalshi, Polymarket, and Solana (via DFlow). It deploys a peer-reviewed pricing model (Wang Transform, calibrated on 291k resolved contracts) to detect favorite-longshot mispricing, plus 8 constraint-based arbitrage strategies and LLM agent strategies, and executes trades through a shared async trading engine with dual-layer risk management.

Package name on disk is `oracle3` (Python 3.10+, Poetry-managed). There is also a small `coinjure` package for cross-platform market matching (used to find implication/exclusivity/complementary relations between Polymarket and Kalshi markets for arbitrage detection).

## Commands

```bash
# Setup
poetry install --with dev,test

# Run all tests (mirrors CI)
poetry run pytest --cov=oracle3 --cov-report=term-missing
pytest tests/ -v                      # simpler local run
pytest tests/test_paper_trader.py -v  # single file
pytest tests/test_paper_trader.py::test_name -v   # single test

# Lint / format / type-check (run before committing; CI enforces all three)
ruff check .
ruff format .
mypy --config-file pyproject.toml .

# CLI (installed as `oracle3` entrypoint -> oracle3.cli.cli:cli)
oracle3 market list --exchange polymarket --limit 10
oracle3 dashboard --exchange solana --initial-capital 10000
oracle3 monitor --watch
oracle3 strategy create --output strategies/<name>.py --class-name <ClassName> --type quant
oracle3 strategy validate --strategy-ref strategies/<name>.py:<ClassName> --dry-run --events 10 --json
oracle3 backtest run --history-file <history.jsonl> --market-id <M> --event-id <E> --strategy-ref <path>:<Class> --json
oracle3 research walk-forward / stress-test / strategy-gate / alpha-pipeline / auto-tune  # research subcommands, see oracle3/cli/research_commands.py
```

Tests exclude `examples/`, `scripts/`, and `tests/` themselves from mypy checking. Several modules with known-messy typing are exempted per-file in `pyproject.toml` (`[[tool.mypy.overrides]]`) — don't fight those; extend the ignore list rather than forcing strict typing on legacy code there.

Ruff config: single quotes, 88-char lines, `E501`/`E203` ignored. Several files have per-file `C901` (complexity) ignores in `pyproject.toml` — new files should stay under the default complexity limit rather than being added to that list.

## Architecture

Full architecture doc: `docs/PROJECT_SPECIFICATION.md` (read this for module-by-module detail; summarized below).

```
CLI (paper|live|backtest|dashboard|blinks|research)
        |
Strategy layer: AgentStrategy (LLM+tools) | QuantStrategy (momentum/MR/MM) | contrib/ (arb, debate, news, multi-agent)
        |
Trading Engine (oracle3/core/trading_engine.py): async event loop, snapshot persistence, Unix-socket control (pause/resume/killswitch)
        |
Trader (exchange-specific): solana_trader / polymarket_trader / kalshi_trader, + jito_submitter, flash_loan, atomic_trader
        |
Kalshi (REST) | Polymarket (CLOB API) | Solana/DFlow (SPL tokens)
```

Execution flow: data sources (market data, news, on-chain signals) -> strategy produces a `StrategyDecision` with confidence/edge -> risk manager validates against portfolio limits (and simulates the tx on-chain for Solana) -> trader signs and submits (optionally via Jito bundle) -> trade is logged on-chain (Solana Memo) and reflected in position/analytics -> dashboard shows live P&L.

Key modules (see `docs/PROJECT_SPECIFICATION.md` for the rest):
- `oracle3/strategy/strategy.py` — abstract `Strategy` base class every strategy subclasses. Must implement `process_event(event, trader)`; optional `on_start`/`on_stop`, `param_schema()` for auto-tune, `record_decision(...)` for the shared decision buffer. `QuantStrategy` sets `supports_auto_tune() -> True`; `AgentStrategy` does not.
- `oracle3/strategy/contrib/` — where strategy contributions go (per `CONTRIBUTING.md`); one file per strategy, each with a matching `C901` ruff exemption if genuinely complex.
- `oracle3/risk/` — `risk_manager.py` (local position/exposure/drawdown limits) and `onchain_risk_manager.py` (Solana `simulateTransaction` pre-flight) — dual-layer by design, don't collapse them.
- `oracle3/pricing/wang_mle.py` — the calibrated Wang Transform pricing model ($\hat\lambda \approx 0.183$) strategies use for fair-value edge.
- `oracle3/trader/atomic_trader.py` / `flash_loan.py` — multi-leg / flash-loan execution; these assume atomicity guarantees from the underlying Solana program, don't add manual unwind logic on top.
- `coinjure/matching/` — market-relation discovery (implication, exclusivity, complementary) across exchanges; feeds the cross-market arbitrage strategies.

### Writing a new strategy

Follow `skills/pm-quant-strategy-authoring` (quant) or `skills/pm-agent-strategy-authoring` (LLM) if present — they encode the validated workflow:
1. Scaffold: `oracle3 strategy create --output strategies/<name>.py --class-name <ClassName> --type quant`
2. Implement `process_event`, keep tunable values as constructor params (JSON-serializable, for auto-tune), call `trader.place_order(...)` and `self.record_decision(...)`.
3. Validate: `oracle3 strategy validate --strategy-ref strategies/<name>.py:<ClassName> --strategy-kwargs-json '<json>' --dry-run --events 10 --json`
4. Backtest against a single market/event before wider runs.
No look-ahead (don't use future information in `process_event`), and no strategy logic embedded directly in CLI scripts — it must live in a `Strategy` subclass file so it's reusable for backtest/paper/live/auto-tune.

## Environment

Copy `.env.example` to `.env`. Needed credentials depend on which venues you're touching: Polymarket CLOB (`POLYMARKET_API_KEY`/`SECRET`/`PASSPHRASE`, plus `POLYMARKET_PRIVATE_KEY` for signing live orders), Kalshi (`KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`), and `DEEPSEEK_API_KEY` for LLM strategies (the CLI/tests also reference `OPENAI_API_KEY`/`TOGETHERAI_API_KEY` in CI).

---

# Behavioral Guidelines

Guidelines to reduce common LLM coding mistakes, applied on top of the project context above.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
