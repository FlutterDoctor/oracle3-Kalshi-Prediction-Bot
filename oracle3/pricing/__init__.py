"""Probabilistic fair value engine for prediction markets.

Implements the pricing framework from:

    Yang, Y. (2026). "Pricing Prediction Markets: Risk Premiums,
    Incomplete Markets, and a Decomposition Framework." Working Paper,
    University of Illinois Urbana-Champaign.

Key components:

- **Wang Transform MLE** — batch maximum likelihood estimation with
  analytic gradients, sandwich robust SEs, and hierarchical covariates.

- **Distortion functions** — probit, dual power, proportional hazard
  families for probability-to-price mapping.

- **Fair value estimator** — real-time physical probability extraction
  using empirically calibrated coefficients from 291K+ contracts.

- **Online calibrator** — hybrid batch MLE + streaming EWMA with
  category-level hierarchical shrinkage and empirical priors.

- **Model Greeks** — analytic sensitivities (delta, gamma, Kelly)
  for position sizing and risk management.

- **Premium tracker** — lifecycle analysis of risk premium decay
  for timing-based alpha.

- **Contract scorer** — microstructure quality scoring.
"""

from oracle3.pricing.calibrator import OnlineCalibrator
from oracle3.pricing.contract_scorer import ContractFeatures, ContractScorer
from oracle3.pricing.distortion import (
    DistortionFunction,
    DualPowerDistortion,
    ProbitDistortion,
    ProportionalHazardDistortion,
)
from oracle3.pricing.fair_value import FairValueEstimate, FairValueEstimator
from oracle3.pricing.greeks import (
    compute_greeks_table,
    delta_lambda,
    delta_physical,
    edge_at_percentile,
    implied_edge,
    kelly_fraction,
    premium_amount,
    premium_ratio,
)
from oracle3.pricing.premium_tracker import PremiumSnapshot, PremiumTracker
from oracle3.pricing.wang_mle import MLEResult, WangMLE

__all__ = [
    'DistortionFunction',
    'ProbitDistortion',
    'DualPowerDistortion',
    'ProportionalHazardDistortion',
    'FairValueEstimator',
    'FairValueEstimate',
    'OnlineCalibrator',
    'PremiumTracker',
    'PremiumSnapshot',
    'ContractScorer',
    'ContractFeatures',
    'WangMLE',
    'MLEResult',
    'delta_lambda',
    'delta_physical',
    'premium_amount',
    'premium_ratio',
    'implied_edge',
    'kelly_fraction',
    'edge_at_percentile',
    'compute_greeks_table',
]
