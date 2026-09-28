# ADR-0004 — Lost-sales simulation and iso-service comparison

**Status:** Accepted, extended by [ADR-0005](0005-configurable-scenario-and-channels.md) (online stores use partial backorders) · **Date:** 2026-09-28

## Context
The headline claim compares average inventory between policies "at a 95% fill rate". Two design choices change the result a lot:
1. What happens to unmet demand (backorder or lost)?
2. How to compare policies that reach different service levels?

## Decision
1. **Lost sales:** in retail, a customer who finds an empty shelf usually buys elsewhere. Unmet demand is lost and costs the gross margin `p − c`.
2. **Iso-service comparison:** each policy is calibrated (one scalar knob, bisection) to a fill rate in [95.0%, 95.2%] on the 2017 simulation before average inventory is compared. The non-calibrated results (z fixed in advance) are also published.

## Consequences
- ➕ The headline compares like with like; it holds up when a reviewer asks "at what service level?".
- ➕ With lost sales, inventory can never go negative, which makes the simulation simpler.
- ➖ Calibrating on the evaluation year is an oracle setting → this is stated clearly in the README and the methodology, and the non-calibrated numbers are shown next to it.
