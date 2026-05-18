//! Bayesian edge strategy for Deadeye distribution markets.
//!
//! The strategy carries a private belief distribution `N(μ_b, σ_b²)` and
//! a single decision rule: if `KL(belief ‖ market)` exceeds an edge
//! threshold, reshape the market to the belief.
//!
//! `KL(belief ‖ market) = E_belief[log p_belief(x) − log p_market(x)]`
//! is a Bayesian estimate of the log-pdf gain at the realized outcome
//! *under the assumption that the belief is correct*. The Deadeye
//! settlement then provides the ground-truth filter: a confident-but-
//! wrong reshape pays collateral up front and collects a negative
//! position value at `x*`.

use anyhow::{Context, Result, anyhow};
use deadeye_sdk::core::{Distribution, NormalDistribution, Sq128};
use deadeye_sdk::{
    EventDistribution, MarketEvent, MarketState, SimDistribution, Strategy, StrategyAction,
};

#[derive(Debug, Clone, Copy)]
pub struct EdgeParams {
    pub label: &'static str,
    pub belief_mu: f64,
    pub belief_sigma: f64,
    pub edge_threshold: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Decision {
    pub event_idx: usize,
    pub market_mu: f64,
    pub market_sigma: f64,
    pub kl: f64,
    pub traded: bool,
}

pub struct BayesianEdgeStrategy {
    pub params: EdgeParams,
    pub decisions: Vec<Decision>,
    /// `log p_belief(x*) − log p_market(x*)` at settlement.
    pub realized_log_score: Option<f64>,
    /// Deadeye position value at settlement (article §4):
    /// `pdf(f_eff, x*) − pdf(f_orig, x*)`, λ multipliers omitted.
    pub deadeye_payout: Option<f64>,
    /// `(pdf(f_orig, x*), pdf(f_eff, x*))` for narrative clarity.
    pub payout_breakdown: Option<(f64, f64)>,
    /// Last *market-driven* distribution observed (only updated at Trade
    /// events — at AddLiquidity/RemoveLiquidity/Settle the engine leaves
    /// `state.distribution` untouched, so reading it there can return
    /// our own belief from a previous round).
    last_market: Option<(f64, f64)>,
    /// Market state at the moment of the strategy's *first* trade.
    /// Defines the lower endpoint of the Deadeye position.
    f_orig: Option<(f64, f64)>,
    /// Market state after the strategy's *last* trade — in our setup,
    /// always equal to the belief.
    f_eff: Option<(f64, f64)>,
    event_counter: usize,
}

impl BayesianEdgeStrategy {
    pub fn new(params: EdgeParams) -> Self {
        Self {
            params,
            decisions: Vec::new(),
            realized_log_score: None,
            deadeye_payout: None,
            payout_breakdown: None,
            last_market: None,
            f_orig: None,
            f_eff: None,
            event_counter: 0,
        }
    }

    fn build_candidate(&self) -> Result<EventDistribution> {
        let mu = Sq128::from_f64(self.params.belief_mu)
            .map_err(|e| anyhow!("Sq128 from belief_mu: {e}"))?;
        let var = Sq128::from_f64(self.params.belief_sigma.powi(2))
            .map_err(|e| anyhow!("Sq128 from belief_sigma²: {e}"))?;
        let d = NormalDistribution::from_variance(mu, var)
            .map_err(|e| anyhow!("NormalDistribution::from_variance: {e}"))
            .context("building belief distribution")?;
        Ok(EventDistribution::Normal(d))
    }
}

impl Strategy for BayesianEdgeStrategy {
    fn on_event(&mut self, state: &MarketState, event: &MarketEvent) -> Vec<StrategyAction> {
        let idx = self.event_counter;
        self.event_counter += 1;

        if let MarketEvent::Settle { x_star } = event {
            if let Some((m_mu, m_var)) = self.last_market {
                let belief_var = self.params.belief_sigma.powi(2);
                let lp_belief = normal_log_pdf(*x_star, self.params.belief_mu, belief_var);
                let lp_market = normal_log_pdf(*x_star, m_mu, m_var);
                self.realized_log_score = Some(lp_belief - lp_market);
            }
            if let (Some((o_mu, o_var)), Some((e_mu, e_var))) = (self.f_orig, self.f_eff) {
                let pdf_orig = normal_log_pdf(*x_star, o_mu, o_var).exp();
                let pdf_eff = normal_log_pdf(*x_star, e_mu, e_var).exp();
                self.payout_breakdown = Some((pdf_orig, pdf_eff));
                self.deadeye_payout = Some(pdf_eff - pdf_orig);
            }
            return Vec::new();
        }

        let SimDistribution::Normal(ref n) = state.distribution else {
            return Vec::new();
        };
        let market_mu = n.mean().to_f64();
        let market_var = n.variance().to_f64();
        let market_sigma = market_var.sqrt();

        if matches!(event, MarketEvent::Trade { .. }) {
            self.last_market = Some((market_mu, market_var));
        }

        let belief_var = self.params.belief_sigma.powi(2);
        let kl = kl_normal(self.params.belief_mu, belief_var, market_mu, market_var);

        let should_trade =
            matches!(event, MarketEvent::Trade { .. }) && kl > self.params.edge_threshold;

        self.decisions.push(Decision {
            event_idx: idx,
            market_mu,
            market_sigma,
            kl,
            traded: should_trade,
        });

        if should_trade {
            if self.f_orig.is_none() {
                self.f_orig = Some((market_mu, market_var));
            }
            self.f_eff = Some((self.params.belief_mu, belief_var));
            match self.build_candidate() {
                Ok(candidate) => vec![StrategyAction::Trade { candidate }],
                Err(err) => {
                    tracing::warn!(error = %err, "edge: failed to build candidate; holding");
                    Vec::new()
                },
            }
        } else {
            Vec::new()
        }
    }
}

/// KL(N(μ₁, σ₁²) ‖ N(μ₂, σ₂²)) in nats.
pub fn kl_normal(mu1: f64, var1: f64, mu2: f64, var2: f64) -> f64 {
    debug_assert!(var1 > 0.0 && var2 > 0.0, "variances must be positive");
    0.5 * (var2.ln() - var1.ln()) + (var1 + (mu1 - mu2).powi(2)) / (2.0 * var2) - 0.5
}

/// log N(x | μ, σ²).
pub fn normal_log_pdf(x: f64, mu: f64, var: f64) -> f64 {
    -0.5 * (2.0 * std::f64::consts::PI * var).ln() - (x - mu).powi(2) / (2.0 * var)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_is_zero_for_identical_normals() {
        let kl = kl_normal(1.0, 4.0, 1.0, 4.0);
        assert!(kl.abs() < 1e-12, "KL(p,p) must be 0, got {kl}");
    }

    #[test]
    fn kl_is_positive_for_distinct_normals() {
        let kl = kl_normal(1.0, 4.0, 2.0, 4.0);
        assert!(kl > 0.0);
    }

    #[test]
    fn kl_is_asymmetric() {
        let forward = kl_normal(0.0, 1.0, 0.0, 9.0);
        let reverse = kl_normal(0.0, 9.0, 0.0, 1.0);
        assert!((forward - reverse).abs() > 1e-6);
    }

    #[test]
    fn log_pdf_matches_density_at_mean() {
        let var = 4.0_f64;
        let lp = normal_log_pdf(3.0, 3.0, var);
        let expected = -0.5 * (2.0 * std::f64::consts::PI * var).ln();
        assert!((lp - expected).abs() < 1e-12);
    }
}
