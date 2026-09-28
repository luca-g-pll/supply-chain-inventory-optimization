# 09 — KPI Definitions

All KPIs are computed on the **test-period simulation** (2017 in the demo) per `(store, item, channel, policy)`, then aggregated over the **active scenario** (selected stores × selected products, each store under its own channel). Available breakdowns: total, by policy, by channel, by ABC/XYZ class, by store. **Aggregation is always done from the underlying sums** (e.g. fill rate = Σ sales / Σ demand), never as an average of ratios.

## 1. Service KPIs

| KPI | Formula | Unit | Better |
|---|---|---|---|
| **Fill rate** (β) | `Σ sales / Σ demand` | % | ↑ |
| Cycle service level (α) | `cycles with no stock-out / total cycles` (cycle = time between two receipts) | % | ↑ |
| Stock-out days | `# days with lost > 0` | days | ↓ |
| Stock-out events | `# runs of consecutive stock-out days` | count | ↓ |
| Lost units | `Σ lost` | units | ↓ |
| Lost revenue | `Σ lost · p` | € | ↓ |
| Back-ordered units *(online)* | `Σ backordered` | units | ↓ |
| Average backlog *(online)* | `mean_t(backlog)` | units | ↓ |

## 2. Inventory KPIs

| KPI | Formula | Unit | Better |
|---|---|---|---|
| **Average inventory (units)** | `mean_t(on_hand_end)` | units | ↓ |
| **Average inventory (value)** | `mean_t(on_hand_end) · c` | € | ↓ |
| Inventory turns | `Σ sales · c / average inventory value` (annualised) | ×/year | ↑ |
| Days of supply | `average inventory units / mean daily sales` | days | ↓ (at the same service) |
| Number of orders | `# days with order_qty > 0` | count | – |

## 3. Cost KPIs

| KPI | Formula | Unit |
|---|---|---|
| Ordering cost | `n_orders · S` | € |
| Holding cost | `Σ_t on_hand_end(t) · c · i / 365` | € |
| Shortage cost | `Σ lost · (p − c) + Σ backordered · backorder_cost` | € |
| **Total cost** | ordering + holding + shortage | € |

## 4. Forecast KPIs

| KPI | Formula | Better |
|---|---|---|
| **WAPE** | `Σ|A − F| / Σ A` | ↓ |
| MAPE | `mean(|A − F| / A)`, A > 0 | ↓ |
| **Bias** | `Σ(F − A) / Σ A` | → 0 |
| RMSE | `√mean((A − F)²)` | ↓ |
| Improvement vs. baseline | `1 − WAPE_model / WAPE_seasonal_naive` | ↑ |

## 5. Headline KPIs (README & CV)

| Headline | Formula |
|---|---|
| Forecast gain | `1 − WAPE_best / WAPE_seasonal_naive` |
| **Inventory reduction** | `1 − AvgInvValue_P2 / AvgInvValue_P0`, both calibrated to fill rate = 95% |
| Cost reduction | `1 − TotalCost_P2 / TotalCost_P0` and the difference in € |

## 6. Descriptive KPIs (SQL)

| KPI | Query |
|---|---|
| Total units / revenue, # series, date range | `kpi_overview` |
| YoY growth % | `sales_by_year` |
| Day-of-week / month index | `dow_index`, `month_index` |
| Revenue share by item / store | `revenue_by_item`, `revenue_by_store` |
| Weekly CV per item | `weekly_cv_by_item` |
