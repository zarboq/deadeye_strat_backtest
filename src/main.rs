//! Bayesian edge strategy backtester on top of `deadeye-sdk`.
//!
//! Builds a synthetic Deadeye Normal-AMM timeline where the public
//! consensus drifts toward a *biased* estimate of the truth, then runs
//! a handful of belief-driven strategies through `BacktestEngine`. Each
//! strategy uses `KL(belief ‖ market)` as a pre-trade edge signal and
//! reports two settlement metrics: the Deadeye position value
//! (`pdf(f_eff, x*) − pdf(f_orig, x*)`) and the realized log-score
//! against the final market quote.
//!
//! Run with: `cargo run --release`

mod strategy;
mod synthetic;

use anyhow::Result;
use deadeye_sdk::{BacktestEngine, BacktestResult};

use crate::strategy::{BayesianEdgeStrategy, EdgeParams};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let spec = synthetic::TimelineSpec::default();
    let engine = synthetic::build(spec)?;

    println!("─── Deadeye Bayesian-edge backtester ──────────────────────────");
    println!(
        "Synthetic timeline   : {} trade events + 1 settle",
        spec.steps
    );
    println!(
        "Initial consensus    : N(μ={:.2}, σ={:.2})",
        spec.initial_mu, spec.initial_sigma
    );
    println!(
        "Market drifts toward : N(μ={:.2}, σ={:.2})  (biased: under-estimates truth)",
        spec.market_target_mu, spec.market_target_sigma
    );
    println!("True outcome (x*)    : {:.2}", spec.truth_mu);
    println!();

    let strategies = vec![
        EdgeParams {
            label: "Informed",
            belief_mu: 49.0,
            belief_sigma: 4.0,
            edge_threshold: 0.05,
        },
        EdgeParams {
            label: "Oracle (= truth)",
            belief_mu: spec.truth_mu,
            belief_sigma: 4.0,
            edge_threshold: 0.05,
        },
        EdgeParams {
            label: "Wrong-Confident",
            belief_mu: 35.0,
            belief_sigma: 3.0,
            edge_threshold: 0.05,
        },
    ];

    println!(
        "{:<20} {:>7} {:>9} {:>10} {:>14} {:>15}",
        "strategy", "trades", "collat", "<KL>", "log-score@x*", "Δpdf@x* (×10³)"
    );
    println!("{}", "─".repeat(80));

    let mut runs = Vec::with_capacity(strategies.len());
    for params in strategies {
        let run = run_one(&engine, params);
        print_row(&run);
        runs.push(run);
    }

    println!();
    println!("─── Per-event trace (Informed strategy) ──────────────────────────");
    if let Some(informed) = runs.iter().find(|r| r.params.label == "Informed") {
        print_decisions(informed);
    }

    println!();
    println!("─── Deadeye position decomposition at x* ──────────────────────");
    println!(
        "{:<20} {:>12} {:>12} {:>14}",
        "strategy", "pdf(f_orig)", "pdf(f_eff)", "Δ = position"
    );
    println!("{}", "─".repeat(62));
    for run in &runs {
        match run.payout_breakdown {
            Some((orig, eff)) => println!(
                "{:<20} {:>12.5} {:>12.5} {:>+14.5}",
                run.params.label, orig, eff, eff - orig
            ),
            None => println!(
                "{:<20} {:>12} {:>12} {:>14}",
                run.params.label, "n/a", "n/a", "n/a (0 trades)"
            ),
        }
    }

    println!();
    println!("Reading:");
    println!("  · <KL>           = mean KL(belief ‖ market) across all events");
    println!("                     — the pre-trade Bayesian edge signal");
    println!("  · Δpdf@x*        = realized Deadeye position value:");
    println!("                     pdf(f_eff, x*) − pdf(f_orig, x*), per unit");
    println!("  · log-score@x*   = log p_belief(x*) − log p_market(x*)");
    println!();
    println!("KL(belief ‖ market) = E_belief[log p_belief − log p_market] is the");
    println!("strategy's *expected* log-pdf gain at the realized outcome under its");
    println!("own belief. Δpdf@x* is the realized version.");
    println!();
    println!("  · Informed / Oracle   → modest KL, positive Δpdf@x* (belief paid off)");
    println!("  · Wrong-Confident     → huge KL but negative Δpdf@x* — Deadeye's");
    println!("                          deterministic settle filters confident-but-");
    println!("                          wrong reshapes: belief was *novel*, not *true*");

    Ok(())
}

struct Run {
    params: EdgeParams,
    result: BacktestResult,
    decisions: Vec<strategy::Decision>,
    realized_log_score: Option<f64>,
    deadeye_payout: Option<f64>,
    payout_breakdown: Option<(f64, f64)>,
}

fn run_one(engine: &BacktestEngine, params: EdgeParams) -> Run {
    let mut strategy = BayesianEdgeStrategy::new(params);
    let result = engine.run({
        struct Adapter<'a>(&'a mut BayesianEdgeStrategy);
        impl deadeye_sdk::Strategy for Adapter<'_> {
            fn on_event(
                &mut self,
                state: &deadeye_sdk::MarketState,
                event: &deadeye_sdk::MarketEvent,
            ) -> Vec<deadeye_sdk::StrategyAction> {
                self.0.on_event(state, event)
            }
        }
        Adapter(&mut strategy)
    });

    Run {
        params,
        result,
        decisions: strategy.decisions,
        realized_log_score: strategy.realized_log_score,
        deadeye_payout: strategy.deadeye_payout,
        payout_breakdown: strategy.payout_breakdown,
    }
}

fn print_row(run: &Run) {
    let n = run.decisions.len().max(1) as f64;
    let mean_kl: f64 = run.decisions.iter().map(|d| d.kl).sum::<f64>() / n;
    let ls = run
        .realized_log_score
        .map(|v| format!("{:>+14.4}", v))
        .unwrap_or_else(|| format!("{:>14}", "n/a"));
    let dpdf = run
        .deadeye_payout
        .map(|v| format!("{:>+15.3}", v * 1_000.0))
        .unwrap_or_else(|| format!("{:>15}", "n/a"));

    println!(
        "{:<20} {:>7} {:>+9.3} {:>10.4} {} {}",
        run.params.label,
        run.result.trades_executed,
        run.result.final_pnl,
        mean_kl,
        ls,
        dpdf,
    );
}

fn print_decisions(run: &Run) {
    println!(
        "{:>4} {:>9} {:>9} {:>10} {:>6}",
        "ev", "μ_mkt", "σ_mkt", "KL", "trade"
    );
    println!("{}", "─".repeat(50));
    for d in run.decisions.iter().take(15) {
        println!(
            "{:>4} {:>9.3} {:>9.3} {:>10.4} {:>6}",
            d.event_idx,
            d.market_mu,
            d.market_sigma,
            d.kl,
            if d.traded { "✓" } else { "·" },
        );
    }
    if run.decisions.len() > 15 {
        println!("    … ({} more events truncated)", run.decisions.len() - 15);
    }
}
