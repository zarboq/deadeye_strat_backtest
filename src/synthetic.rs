//! Synthetic market timeline used by the demo.
//!
//! Builds a sequence of Normal-AMM `MarketEvent::Trade` rows whose mean
//! drifts from a wrong initial consensus toward the eventual settlement
//! value, with bounded jitter. A terminal `Settle` event reveals the
//! truth and lets us score every strategy ex-post.

use anyhow::{Context, Result, anyhow};
use deadeye_sdk::core::{NormalDistribution, Sq128};
use deadeye_sdk::starknet::Felt;
use deadeye_sdk::{BacktestEngine, EventDistribution, MarketEvent, MarketState, SimDistribution};

#[derive(Debug, Clone, Copy)]
pub struct TimelineSpec {
    pub initial_mu: f64,
    pub initial_sigma: f64,
    /// What the market consensus drifts toward — typically a *biased*
    /// estimate of the truth, mirroring a real prediction market where
    /// the crowd anchors on visible signals and misses the true value.
    pub market_target_mu: f64,
    pub market_target_sigma: f64,
    /// Realized outcome revealed by the terminal Settle event.
    pub truth_mu: f64,
    pub steps: usize,
    pub jitter: f64,
}

impl Default for TimelineSpec {
    fn default() -> Self {
        Self {
            initial_mu: 45.0,
            initial_sigma: 8.0,
            market_target_mu: 48.5, // market underestimates truth by 1.5
            market_target_sigma: 5.0,
            truth_mu: 50.0,
            steps: 30,
            jitter: 0.4,
        }
    }
}

pub fn build(spec: TimelineSpec) -> Result<BacktestEngine> {
    let initial_state = MarketState {
        distribution: SimDistribution::Normal(make_normal(spec.initial_mu, spec.initial_sigma)?),
        backing: 1_000.0,
        lp_shares: 1_000.0,
        settlement_x_star: None,
    };

    let mut events = Vec::with_capacity(spec.steps + 1);
    for i in 0..spec.steps {
        let t = (i as f64 + 1.0) / (spec.steps as f64);
        // Deterministic, bounded "jitter" via a triangular wave on the
        // odd indices — keeps the demo reproducible without an RNG dep.
        let wave = if i % 2 == 0 { spec.jitter } else { -spec.jitter };
        let mu =
            spec.initial_mu + (spec.market_target_mu - spec.initial_mu) * t + wave * (1.0 - t);
        let sigma = spec.initial_sigma
            + (spec.market_target_sigma - spec.initial_sigma) * t;
        let candidate = make_normal(mu, sigma.max(0.5))?;
        events.push(MarketEvent::Trade {
            trader: Felt::from(i as u64 + 1),
            candidate: EventDistribution::Normal(candidate),
        });
    }
    events.push(MarketEvent::Settle {
        x_star: spec.truth_mu,
    });

    Ok(BacktestEngine::from_indexer_events(events, initial_state))
}

fn make_normal(mu: f64, sigma: f64) -> Result<NormalDistribution> {
    let mu_q = Sq128::from_f64(mu).map_err(|e| anyhow!("Sq128 from μ: {e}"))?;
    let var_q = Sq128::from_f64(sigma * sigma).map_err(|e| anyhow!("Sq128 from σ²: {e}"))?;
    NormalDistribution::from_variance(mu_q, var_q)
        .map_err(|e| anyhow!("NormalDistribution::from_variance: {e}"))
        .context("synthetic normal construction")
}
