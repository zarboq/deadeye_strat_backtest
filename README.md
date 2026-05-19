# deadeye_strat_backtest

A small Bayesian-edge strategy backtester built on top of
[`deadeye-sdk`](https://docs.rs/deadeye-sdk) — written to understand
how Deadeye distribution markets work in practice by implementing a
non-trivial strategy on top of the SDK.

## Why this exists

This is a self-study artifact. The goal was to:

1. Read the `deadeye-sdk` source carefully enough to understand the
   architecture (Normal-AMM, distribution as the traded object,
   deterministic settle, position as the difference between two
   endpoints).
2. Implement a coherent strategy on top of the SDK's `Strategy` trait
   to exercise the backtest harness end-to-end.
3. Produce a runnable demo whose output visibly illustrates how the
   deterministic settlement filters strategies that hold
   confident-but-wrong beliefs.

It is not a research contribution. The "edge signal" used here —
`KL(belief ‖ market)` — is just Bayesian decision theory: the expected
log-pdf gain at the realized outcome under the strategy's own belief.

## What the strategy does

`BayesianEdgeStrategy` (in [`src/strategy.rs`](src/strategy.rs)):

- Holds a private belief distribution `N(μ_b, σ_b²)`.
- At every Trade event, reads the public market consensus
  `N(μ_m, σ_m²)` from `MarketState`.
- Computes `KL(belief ‖ market)`.
- If that KL exceeds a configured `edge_threshold`, reshapes the
  market to the belief (a `StrategyAction::Trade`).
- At the terminal Settle event, captures two ex-post metrics:
  - `realized_log_score = log p_belief(x*) − log p_market(x*)`
  - `deadeye_payout = pdf(f_eff, x*) − pdf(f_orig, x*)` — the position
    value at settlement, with `f_orig` = market at the moment of the
    first trade and `f_eff` = belief after the last trade.

Three strategies are run side by side in [`src/main.rs`](src/main.rs):

| Strategy        | Belief μ | Belief σ | What it tests                                |
|-----------------|----------|----------|----------------------------------------------|
| Informed        | 49       | 4        | A well-informed trader, close to truth (50). |
| Oracle          | 50       | 4        | A trader who already knows the truth.        |
| Wrong-Confident | 35       | 3        | A confident trader with a wrong belief.      |

The synthetic timeline ([`src/synthetic.rs`](src/synthetic.rs)) is a
30-step drift from `N(45, 8)` toward a *biased* consensus `N(48.5, 5)`,
terminated by a `Settle { x_star: 50.0 }` event. The biased consensus
is intentional: a market that converges exactly to truth would leave no
edge for informed strategies to exploit.

## Running it

Requires Rust 1.92 (pinned in `rust-toolchain.toml` because that's the
MSRV declared by the SDK).

```bash
cargo run --release
cargo test --release
```

## Sample output

```
strategy              trades    collat       <KL>   log-score@x*  Δpdf@x* (×10³)
────────────────────────────────────────────────────────────────────────────────
Informed                   1    -0.031     0.2313        +0.2369         +53.720
Oracle (= truth)           1    -0.035     0.2926        +0.2681         +56.789
Wrong-Confident            1    -0.050     2.2395       -11.9442         -42.946
```

Reading:

- **Informed / Oracle**: modest KL, positive `Δpdf@x*` — the belief
  was both novel *and* correct, so the reshape paid off at settlement.
- **Wrong-Confident**: huge KL (the belief is far from consensus) but
  catastrophic negative `Δpdf@x*` — the strategy paid collateral to
  push the market into a region the realized outcome never visited.

The bridge between the two columns is the point of the demo:
`KL(belief ‖ market) = E_belief[log p_belief − log p_market]` is the
strategy's *expected* log-pdf gain. `Δpdf@x*` is the realized version.
A high KL is necessary for a profitable reshape but not sufficient —
the realized `x*` decides.

### Why each strategy only trades once

Position value at settle is `pdf(f_eff, x*) − pdf(f_orig, x*)` — only
the two endpoints of the position matter. Intermediate reshapes
telescope away. So retrading toward a stationary belief pays collateral
on every transition without changing the payoff.

The first iteration of this backtester retraded every time the market
drifted from the belief — 29 to 30 trades per strategy, collateral
0.76–1.89. Adding a guard that only trades when the belief itself has
moved away from `f_eff` drops every strategy to a single trade with
identical `Δpdf@x*` and 25–40× less collateral. A small refactor that
makes the strategy cohere with the AMM's economic structure.

## Architecture

```
src/
├── main.rs        entry point: builds the engine, runs strategies, prints output
├── strategy.rs    BayesianEdgeStrategy + KL math + log-pdf helpers
└── synthetic.rs   synthetic Normal-AMM timeline (30 trades + 1 settle)
```

The strategy implements `deadeye_sdk::Strategy`, so swapping the
synthetic timeline for a real `BacktestEngine::from_journal(path)`
requires zero changes to strategy code.

## Possible next steps

- Wire `BacktestEngine::from_journal(path)` behind a CLI flag so the
  same code runs against a real trade journal.
- Extend the strategy to handle the Multinoulli family (the SDK
  exposes `CategoricalDistribution`; KL is just a sum over outcomes).
- Plot `KL_t` and the running estimate of expected `Δpdf@x*` over the
  event sequence to make the edge-vs-payoff bridge visual rather than
  tabular.
