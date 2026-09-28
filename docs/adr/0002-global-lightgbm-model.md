# ADR-0002 — One global LightGBM model for all series

**Status:** Accepted · **Date:** 2026-09-28

## Context
There are 500 store-item series with similar seasonal shapes. The options are 500 local models, or one global model with store/item as features. The forecast horizon is 1–14 days.

## Decision
Train **one global LightGBM model** (Tweedie objective) with store and item as categorical features. Use the **direct single-model** approach: every lag/rolling feature is shifted by at least 14 days, so the same model predicts every h = 1…14 without recursion.

## Alternatives considered
- *500 local models* — slow, overfit risk, harder to maintain.
- *Recursive forecasting with lag 1* — errors pile up over the horizon, and the back-test code gets more complex.
- *One model per horizon h* — slightly more accurate, but 14× the training cost; kept as a possible extension.

## Consequences
- ➕ Learns shared patterns across series; one training run per refit (about seconds to a minute).
- ➕ Simple back-test with no recursion.
- ➖ Short-horizon forecasts (h = 1–3) cannot use the most recent days (lag < 14) → slightly worse at short h. Accepted, because the policy needs forecasts over the whole lead time anyway.
