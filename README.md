# Quadeye Market Prediction Challenge — Rank 151 / 7,000+

## What this is

My submission codebase for the Quadeye Market Prediction Challenge, a pan-IIT
quantitative competition involving alpha signal discovery on noisy financial
datasets. Final rank: 151 out of 7,000+ participants.

This repo contains the full model pipeline — not just the winning approach, but
the systematic evaluation across 20+ models, including documented failure modes.
The goal was to understand *why* models fail under distributional shift and
non-stationarity, not just maximize leaderboard score.

## Models included

- LightGBM with custom financial features (top performer)
- Ensemble OU mean-reversion signal
- ARIMA, Kalman filter, and GP baselines
- LSTM for non-stationary sequence modeling
- [add others as you push them]

## Key findings

- Look-ahead contamination was the single most common failure mode across
  naive approaches — rigorous train/test discipline was non-negotiable
- LightGBM with domain-specific features outperformed pure ML approaches
  on this dataset structure
- The OU mean-reversion signal added meaningful alpha when ensembled,
  particularly in low-volatility regimes
- Model performance degraded significantly under distributional shift —
  motivated the regime-conditioning work in my SMA research paper

## Structure

models/
  lgbm_pipeline.py       - LightGBM with custom feature engineering
  ou_signal.py           - Ornstein-Uhlenbeck mean-reversion alpha
  kalman_baseline.py     - Kalman filter baseline
  lstm_model.py          - LSTM for sequence modeling
  ensemble.py            - Final ensemble combining top signals
features/
  feature_engineering.py - Custom financial features
  regime_detection.py    - Regime-conditioned model selection
notebooks/
  analysis.ipynb         - EDA, model comparison, failure mode analysis

## Run it

pip install lightgbm numpy pandas scikit-learn statsmodels matplotlib
python models/lgbm_pipeline.py

## Connection to research

The regime-degradation finding from this challenge directly motivated
my first-author preprint on Statistical Manifold Geometry of Regime Dynamics —
a geometric framework for detecting regime shifts in non-stationary time series.
The challenge was where I first noticed the problem; the paper is where I
solved it formally.
