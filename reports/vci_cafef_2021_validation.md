# VCI vs CafeF validation for 2021-Q1 and 2021-Q2

## Scope

- Stratified sample: **60 firms**, balanced by exchange and whether VCI contains 2021-Q2.
- Independent source: public CafeF quarterly financial-statement pages.
- Ratios were recomputed from balance sheets and income statements; no VCI value was used in CafeF calculations.
- Market capitalization was not compared because CafeF statement pages do not provide a point-in-time market-cap field.

## 2021-Q1 agreement

| Metric | Matched | Correlation | Median absolute difference | Within tolerance |
|---|---:|---:|---:|---:|
| asset_turnover | 57 | 0.989 | 0.015 | 75.4% |
| current_ratio | 60 | 1.000 | 0.000 | 95.0% |
| debt_to_assets | 60 | 0.999 | 0.0000 | 95.0% |
| gross_margin | 57 | 0.994 | 0.0000 | 91.2% |
| quick_ratio | 60 | 1.000 | 0.000 | 98.3% |
| roa | 57 | 0.993 | 0.0015 | 84.2% |

Tolerance definitions: ROA 0.5 percentage point; debt/assets 1 percentage point; current ratio, quick ratio and asset turnover 0.05; gross margin 1 percentage point.

## 2021-Q2 coverage finding

- Among **30 sampled firms missing 2021-Q2 in VCI**, CafeF has a usable Q2 balance sheet and income statement for **30 firms**.
- Among **30 sampled firms missing 2021-Q2 in VCI**, CafeF has all six recomputed model metrics for **29 firms**.
- The difference between these counts is caused by missing prior-quarter history needed for TTM ratios, not by a missing Q2 filing.
- Firms with a Q2 statement but incomplete TTM history: `VTP`.
- This distinguishes a VCI historical coverage gap from a market-wide absence of filings.

Coverage by stratum:

| Exchange | VCI Q2 present | Sample | Q2 statement | Q2 full metrics | Full-metric coverage |
|---|---:|---:|---:|---:|---:|
| HNX | False | 15 | 15 | 15 | 100.0% |
| HNX | True | 15 | 15 | 13 | 86.7% |
| HOSE | False | 15 | 15 | 14 | 93.3% |
| HOSE | True | 15 | 15 | 14 | 93.3% |

## Decision

The 2021-Q2 collapse in VCI is a source-partition completeness failure. The existing DML exclusion remains appropriate until a controlled backfill is completed.

Recommended backfill: calculate 2021-Q2 ratios from CafeF statements, validate the overlapping VCI firms against metric-specific tolerances, and keep provenance columns so source substitutions remain auditable.

## Limitations

- CafeF is an independent public aggregator, not an audited research database.
- Statement labels differ for some industries; the current study already excludes financial firms.
- Restatements, consolidated versus standalone statements, and minority interest treatment can create small differences.
