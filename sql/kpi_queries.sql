-- =============================================================================
-- KPI queries (DuckDB)
-- =============================================================================
-- Tables built by src/data.py::build_duckdb:
--   sales(date, store, item, sales)                 canonical daily sales, zero-filled
--   item_master(item, unit_price, unit_cost, lead_time_days, price_source, cost_source)
--   stores(store, channel)
--   sales_enriched = sales + unit_price, unit_cost, revenue (= sales * unit_price)
--
-- Each query starts with "-- name: <query_name>" and is run by src/sql_runner.py.
-- Named parameters ($ref_start, ...) come from the data profile:
--   $ref_start / $ref_end       reference period = validation period
--   $valid_start / $valid_end   validation period
--   $test_start / $test_end     test (simulation) period
--   $a_threshold / $b_threshold ABC cut-offs, $x_threshold / $y_threshold XYZ cut-offs
-- Store and item IDs are text; ties are broken in natural order (numeric IDs first).
-- =============================================================================


-- name: kpi_overview
-- Purpose : headline numbers of the whole history.
-- Grain   : 1 row.
SELECT
    COUNT(*)                                   AS n_rows,
    COUNT(DISTINCT store)                      AS n_stores,
    COUNT(DISTINCT item)                       AS n_items,
    COUNT(DISTINCT (store, item))              AS n_series,
    MIN(date)                                  AS date_min,
    MAX(date)                                  AS date_max,
    DATE_DIFF('day', MIN(date), MAX(date)) + 1 AS n_days,
    SUM(sales)                                 AS total_units,
    SUM(revenue)                               AS total_revenue,
    SUM(sales) / (DATE_DIFF('day', MIN(date), MAX(date)) + 1) AS avg_units_per_day
FROM sales_enriched;


-- name: daily_totals
-- Purpose : total units and revenue per day (trend chart).
-- Grain   : date.
SELECT date, SUM(sales) AS units, SUM(revenue) AS revenue
FROM sales_enriched
GROUP BY date
ORDER BY date;


-- name: sales_by_year
-- Purpose : volume per year and growth. Growth compares units per day, so a
--           partial first/last year is not penalised.
-- Grain   : year.
WITH y AS (
    SELECT YEAR(date) AS year, COUNT(DISTINCT date) AS n_days,
           SUM(sales) AS units, SUM(revenue) AS revenue
    FROM sales_enriched
    GROUP BY YEAR(date)
)
SELECT
    year, n_days, units, revenue,
    units / n_days AS units_per_day,
    100.0 * (units / n_days) / LAG(units / n_days) OVER (ORDER BY year) - 100.0 AS yoy_growth_pct
FROM y
ORDER BY year;


-- name: sales_by_month
-- Purpose : monthly volume (seasonality and trend).
-- Grain   : year x month.
SELECT
    DATE_TRUNC('month', date) AS month_start,
    YEAR(date)                AS year,
    MONTH(date)               AS month,
    COUNT(DISTINCT date)      AS n_days,
    SUM(sales)                AS units,
    SUM(revenue)              AS revenue
FROM sales_enriched
GROUP BY DATE_TRUNC('month', date), YEAR(date), MONTH(date)
ORDER BY month_start;


-- name: dow_index
-- Purpose : day-of-week index = average daily units on that weekday / overall
--           average daily units (weighted average of the index = 1).
-- Grain   : weekday (0 = Monday).
WITH daily AS (
    SELECT date, SUM(sales) AS units FROM sales GROUP BY date
)
SELECT
    ISODOW(date) - 1                              AS dow,
    DAYNAME(date)                                 AS day_name,
    COUNT(*)                                      AS n_days,
    AVG(units)                                    AS avg_units,
    AVG(units) / (SELECT AVG(units) FROM daily)   AS dow_index
FROM daily
GROUP BY ISODOW(date) - 1, DAYNAME(date)
ORDER BY dow;


-- name: month_index
-- Purpose : month-of-year index = average daily units in that month / overall average.
-- Grain   : month.
WITH daily AS (
    SELECT date, SUM(sales) AS units FROM sales GROUP BY date
)
SELECT
    MONTH(date)                                   AS month,
    MONTHNAME(date)                               AS month_name,
    COUNT(*)                                      AS n_days,
    AVG(units)                                    AS avg_units,
    AVG(units) / (SELECT AVG(units) FROM daily)   AS month_index
FROM daily
GROUP BY MONTH(date), MONTHNAME(date)
ORDER BY month;


-- name: revenue_by_item
-- Purpose : units, revenue and revenue share per item in the reference period.
-- Grain   : item.
WITH r AS (
    SELECT item, SUM(sales) AS units, SUM(revenue) AS revenue
    FROM sales_enriched
    WHERE date BETWEEN $ref_start AND $ref_end
    GROUP BY item
)
SELECT
    item, units, revenue,
    revenue / SUM(revenue) OVER () AS revenue_share,
    ROW_NUMBER() OVER (ORDER BY revenue DESC, TRY_CAST(item AS BIGINT) NULLS LAST, item) AS rank
FROM r
ORDER BY rank;


-- name: revenue_by_store
-- Purpose : units, revenue and revenue share per store in the reference period.
-- Grain   : store.
SELECT
    s.store,
    ANY_VALUE(st.channel)                       AS channel,
    SUM(s.sales)                                AS units,
    SUM(s.revenue)                              AS revenue,
    SUM(s.revenue) / SUM(SUM(s.revenue)) OVER () AS revenue_share
FROM sales_enriched s
JOIN stores st USING (store)
WHERE s.date BETWEEN $ref_start AND $ref_end
GROUP BY s.store
ORDER BY revenue DESC;


-- name: abc_classification
-- Purpose : ABC class per item with window functions. An item is A if its
--           cumulative revenue share (itself included) is <= a_threshold (the top
--           item is always A), B if <= b_threshold, else C. Same rule as
--           src/classification.py (checked by tests/test_sql.py).
-- Grain   : item.
WITH r AS (
    SELECT item, SUM(sales) AS units, SUM(revenue) AS revenue
    FROM sales_enriched
    WHERE date BETWEEN $ref_start AND $ref_end
    GROUP BY item
),
ranked AS (
    SELECT
        item, units, revenue,
        revenue / SUM(revenue) OVER () AS revenue_share,
        ROW_NUMBER() OVER (ORDER BY revenue DESC, TRY_CAST(item AS BIGINT) NULLS LAST, item) AS rank
    FROM r
),
cum AS (
    SELECT *, SUM(revenue_share) OVER (ORDER BY rank ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_share
    FROM ranked
)
SELECT
    item, rank, units, revenue, revenue_share, cum_share,
    CASE
        WHEN rank = 1 OR cum_share <= $a_threshold + 1e-12 THEN 'A'
        WHEN cum_share <= $b_threshold + 1e-12 THEN 'B'
        ELSE 'C'
    END AS abc
FROM cum
ORDER BY rank;


-- name: weekly_cv_by_item
-- Purpose : variability of weekly demand (all stores) in the reference period,
--           on complete 7-day buckets from $ref_start; feeds the XYZ class.
-- Grain   : item.
WITH w AS (
    SELECT
        item,
        DATE_DIFF('day', CAST($ref_start AS DATE), date) // 7 AS bucket,
        SUM(sales)            AS units,
        COUNT(DISTINCT date)  AS n_days
    FROM sales
    WHERE date BETWEEN $ref_start AND $ref_end
    GROUP BY item, bucket
    HAVING COUNT(DISTINCT date) = 7
),
s AS (
    SELECT item, COUNT(*) AS n_weeks, AVG(units) AS weekly_mean, STDDEV_SAMP(units) AS weekly_std
    FROM w
    GROUP BY item
)
SELECT
    item, n_weeks, weekly_mean, weekly_std,
    CASE WHEN weekly_mean > 0 THEN weekly_std / weekly_mean END AS weekly_cv,
    CASE
        WHEN weekly_mean > 0 AND weekly_std / weekly_mean < $x_threshold THEN 'X'
        WHEN weekly_mean > 0 AND weekly_std / weekly_mean < $y_threshold THEN 'Y'
        ELSE 'Z'
    END AS xyz
FROM s
ORDER BY TRY_CAST(item AS BIGINT) NULLS LAST, item;


-- name: store_item_growth
-- Purpose : units per day in the test period vs the validation period, per series
--           (how much demand moved between the period the policies are set on and
--           the period they are tested on).
-- Grain   : store x item.
SELECT
    store, item,
    SUM(CASE WHEN date BETWEEN $valid_start AND $valid_end THEN sales END)
        / (DATE_DIFF('day', CAST($valid_start AS DATE), CAST($valid_end AS DATE)) + 1) AS units_per_day_valid,
    SUM(CASE WHEN date BETWEEN $test_start AND $test_end THEN sales END)
        / (DATE_DIFF('day', CAST($test_start AS DATE), CAST($test_end AS DATE)) + 1)   AS units_per_day_test,
    units_per_day_test / NULLIF(units_per_day_valid, 0) - 1                             AS growth
FROM sales
GROUP BY store, item
ORDER BY TRY_CAST(store AS BIGINT) NULLS LAST, store, TRY_CAST(item AS BIGINT) NULLS LAST, item;


-- name: top_bottom_series
-- Purpose : the 10 highest- and lowest-volume store-item series (whole history).
-- Grain   : series.
WITH s AS (
    SELECT store, item, AVG(sales) AS avg_units_per_day, SUM(revenue) AS revenue
    FROM sales_enriched
    GROUP BY store, item
),
r AS (
    SELECT *,
           ROW_NUMBER() OVER (ORDER BY avg_units_per_day DESC, store, item) AS rank_top,
           ROW_NUMBER() OVER (ORDER BY avg_units_per_day ASC, store, item)  AS rank_bottom
    FROM s
)
SELECT
    CASE WHEN rank_top <= 10 THEN 'top' ELSE 'bottom' END AS side,
    store, item, avg_units_per_day, revenue
FROM r
WHERE rank_top <= 10 OR rank_bottom <= 10
ORDER BY side DESC, avg_units_per_day DESC;


-- name: channel_mix
-- Purpose : units and revenue by sales channel (physical / online stores).
-- Grain   : channel.
SELECT
    st.channel,
    COUNT(DISTINCT s.store) AS n_stores,
    SUM(s.sales)            AS units,
    SUM(s.revenue)          AS revenue
FROM sales_enriched s
JOIN stores st USING (store)
GROUP BY st.channel
ORDER BY revenue DESC;
