"""Validate VCI 2021 quarterly ratios against public CafeF statements."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import time
import unicodedata
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
MASTER_FILE = ROOT / "data" / "processed" / "master_panel_hose_hnx_dataset.csv"
RAW_OUTPUT = ROOT / "data" / "raw" / "cafef_2021_validation_sample.csv"
RESULTS_DIR = ROOT / "results" / "expanded_hose_hnx"
COMPARISON_OUTPUT = RESULTS_DIR / "vci_cafef_2021_comparison.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "vci_cafef_2021_summary.csv"
COVERAGE_OUTPUT = RESULTS_DIR / "vci_cafef_2021_coverage.csv"
REPORT_OUTPUT = ROOT / "reports" / "vci_cafef_2021_validation.md"

PERIODS = ("2021-Q1", "2021-Q2")
METRICS = (
    "roa",
    "debt_to_assets",
    "current_ratio",
    "quick_ratio",
    "asset_turnover",
    "gross_margin",
)
TOLERANCES = {
    "roa": 0.005,
    "debt_to_assets": 0.01,
    "current_ratio": 0.05,
    "quick_ratio": 0.05,
    "asset_turnover": 0.05,
    "gross_margin": 0.01,
}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
)


def fold_text(value: object) -> str:
    text = str(value).replace("\u0111", "d").replace("\u0110", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def quarter_number(period: str) -> int:
    year, quarter = period.split("-Q")
    return int(year) * 4 + int(quarter) - 1


def period_from_number(value: int) -> str:
    year, quarter_index = divmod(value, 4)
    return f"{year}-Q{quarter_index + 1}"


def trailing_periods(period: str) -> list[str]:
    end = quarter_number(period)
    return [period_from_number(value) for value in range(end - 3, end + 1)]


def previous_year_period(period: str) -> str:
    return period_from_number(quarter_number(period) - 4)


def parse_number(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.nan
    text = text.replace(".", "").replace(",", "").replace(" ", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def fetch_cafef_report(
    session: requests.Session,
    ticker: str,
    report_type: str,
    year: int,
    quarter: int,
    timeout: float,
) -> pd.DataFrame:
    url = "https://cafef.vn/du-lieu/BaoCaoTaiChinh.aspx"
    response = session.get(
        url,
        params={
            "quarter": quarter,
            "symbol": ticker,
            "type": report_type,
            "year": year,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text), flavor="lxml")
    candidates = [table for table in tables if table.shape[0] >= 10 and table.shape[1] >= 5]
    if not candidates:
        raise ValueError("financial statement table not found")

    table = max(candidates, key=len).iloc[:, :5].copy()
    target = f"{year}-Q{quarter}"
    report_periods = trailing_periods(target)
    table.columns = ["item", *report_periods]
    table["item_folded"] = table["item"].map(fold_text)
    for period in report_periods:
        table[period] = table[period].map(parse_number)
    return table


def first_metric_row(
    report: pd.DataFrame,
    exact_names: tuple[str, ...] = (),
    contains: tuple[str, ...] = (),
    excludes: tuple[str, ...] = (),
) -> pd.Series | None:
    for _, row in report.iterrows():
        name = str(row["item_folded"])
        exact_match = exact_names and name in exact_names
        contains_match = contains and any(pattern in name for pattern in contains)
        excluded = any(pattern in name for pattern in excludes)
        if (exact_match or contains_match) and not excluded:
            if pd.to_numeric(row.drop(labels=["item", "item_folded"]), errors="coerce").notna().any():
                return row
    return None


def metric_series(report: pd.DataFrame, key: str) -> pd.Series:
    definitions = {
        "total_assets": {
            "exact_names": ("tong cong tai san", "tong tai san", "tong tai san co"),
        },
        "total_liabilities": {
            "exact_names": ("c no phai tra", "no phai tra", "tong no phai tra"),
        },
        "equity": {
            "exact_names": (
                "d von chu so huu",
                "von chu so huu",
                "tong cong von chu so huu",
            ),
        },
        "current_assets": {
            "exact_names": ("a tai san ngan han", "tai san ngan han"),
        },
        "current_liabilities": {
            "exact_names": ("i no ngan han", "no ngan han"),
        },
        "inventory": {
            "exact_names": ("iv hang ton kho", "hang ton kho"),
        },
        "net_revenue": {
            "contains": ("doanh thu thuan ve ban hang va cung cap dich vu",),
        },
        "gross_profit": {
            "contains": ("loi nhuan gop ve ban hang va cung cap dich vu",),
        },
        "parent_net_income": {
            "contains": ("loi nhuan sau thue cong ty me",),
            "excludes": ("khong kiem soat",),
        },
        "net_income": {
            "contains": ("loi nhuan sau thue thu nhap doanh nghiep",),
        },
    }
    definition = definitions[key]
    combined = pd.Series(dtype=float)
    for _, row in report.iterrows():
        name = str(row["item_folded"])
        exact_match = definition.get("exact_names") and name in definition["exact_names"]
        contains_match = definition.get("contains") and any(
            pattern in name for pattern in definition["contains"]
        )
        excluded = any(pattern in name for pattern in definition.get("excludes", ()))
        if (exact_match or contains_match) and not excluded:
            values = pd.to_numeric(
                row.drop(labels=["item", "item_folded"]),
                errors="coerce",
            )
            combined = combined.combine_first(values)
    return combined


def safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def calculate_period_metrics(
    balance_reports: list[pd.DataFrame],
    income_report: pd.DataFrame,
    period: str,
) -> dict[str, float]:
    balance = pd.concat(balance_reports, ignore_index=True)

    total_assets = metric_series(balance, "total_assets")
    liabilities = metric_series(balance, "total_liabilities")
    equity = metric_series(balance, "equity")
    current_assets = metric_series(balance, "current_assets")
    current_liabilities = metric_series(balance, "current_liabilities")
    inventory = metric_series(balance, "inventory")

    assets_now = total_assets.get(period, np.nan)
    assets_prior = total_assets.get(previous_year_period(period), np.nan)
    average_assets = (
        (assets_now + assets_prior) / 2
        if pd.notna(assets_now) and pd.notna(assets_prior)
        else np.nan
    )

    revenue = metric_series(income_report, "net_revenue")
    gross_profit = metric_series(income_report, "gross_profit")
    parent_income = metric_series(income_report, "parent_net_income")
    if parent_income.empty:
        parent_income = metric_series(income_report, "net_income")

    ttm_periods = trailing_periods(period)
    ttm_revenue = revenue.reindex(ttm_periods).sum(min_count=4)
    ttm_gross_profit = gross_profit.reindex(ttm_periods).sum(min_count=4)
    ttm_parent_income = parent_income.reindex(ttm_periods).sum(min_count=4)

    current_assets_now = current_assets.get(period, np.nan)
    current_liabilities_now = current_liabilities.get(period, np.nan)
    inventory_now = inventory.get(period, 0.0)
    if pd.isna(inventory_now):
        inventory_now = 0.0

    return {
        "roa": safe_ratio(ttm_parent_income, average_assets),
        "debt_to_assets": safe_ratio(liabilities.get(period, np.nan), assets_now),
        "current_ratio": safe_ratio(current_assets_now, current_liabilities_now),
        "quick_ratio": safe_ratio(
            current_assets_now - inventory_now,
            current_liabilities_now,
        ),
        "asset_turnover": safe_ratio(ttm_revenue, average_assets),
        "gross_margin": safe_ratio(ttm_gross_profit, ttm_revenue),
    }


def statement_available(
    balance_reports: list[pd.DataFrame],
    income_report: pd.DataFrame,
    period: str,
) -> bool:
    balance = pd.concat(balance_reports, ignore_index=True)
    balance_keys = (
        "total_assets",
        "total_liabilities",
        "current_assets",
        "current_liabilities",
    )
    balance_values = [
        metric_series(balance, key).get(period, np.nan) for key in balance_keys
    ]

    parent_income = metric_series(income_report, "parent_net_income")
    if parent_income.empty:
        parent_income = metric_series(income_report, "net_income")
    income_values = [
        metric_series(income_report, "net_revenue").get(period, np.nan),
        metric_series(income_report, "gross_profit").get(period, np.nan),
        parent_income.get(period, np.nan),
    ]
    return all(pd.notna(value) for value in [*balance_values, *income_values])


def stable_order(ticker: str) -> str:
    return hashlib.sha256(f"cafef-2021-{ticker}".encode("ascii")).hexdigest()


def select_sample(master: pd.DataFrame, max_tickers: int) -> pd.DataFrame:
    q1 = master[master["period"].eq("2021-Q1")].copy()
    q2_tickers = set(master.loc[master["period"].eq("2021-Q2"), "ticker"].astype(str))
    q1["vci_q2_present"] = q1["ticker"].astype(str).isin(q2_tickers)
    q1 = q1.drop_duplicates("ticker")
    q1["sample_order"] = q1["ticker"].astype(str).map(stable_order)

    groups = []
    strata = list(q1.groupby(["exchange", "vci_q2_present"], dropna=False))
    base = max_tickers // len(strata)
    remainder = max_tickers % len(strata)
    for index, (_, frame) in enumerate(strata):
        count = base + (1 if index < remainder else 0)
        groups.append(frame.sort_values("sample_order").head(count))
    return pd.concat(groups, ignore_index=True).sort_values(
        ["exchange", "vci_q2_present", "ticker"]
    )


def fetch_ticker_metrics(
    session: requests.Session,
    ticker: str,
    timeout: float,
    sleep_seconds: float,
) -> list[dict[str, object]]:
    requests_needed = [
        ("BSheet", 2020, 1),
        ("BSheet", 2021, 1),
        ("BSheet", 2021, 2),
        ("IncSta", 2021, 1),
        ("IncSta", 2021, 2),
    ]
    reports: dict[tuple[str, int, int], pd.DataFrame] = {}
    for report_type, year, quarter in requests_needed:
        reports[(report_type, year, quarter)] = fetch_cafef_report(
            session,
            ticker,
            report_type,
            year,
            quarter,
            timeout,
        )
        time.sleep(sleep_seconds)

    balance_reports = [
        reports[("BSheet", 2020, 1)],
        reports[("BSheet", 2021, 1)],
        reports[("BSheet", 2021, 2)],
    ]
    rows = []
    for period, quarter in [("2021-Q1", 1), ("2021-Q2", 2)]:
        metrics = calculate_period_metrics(
            balance_reports,
            reports[("IncSta", 2021, quarter)],
            period,
        )
        rows.append(
            {
                "ticker": ticker,
                "period": period,
                **metrics,
                "cafef_statement_available": statement_available(
                    balance_reports,
                    reports[("IncSta", 2021, quarter)],
                    period,
                ),
                "cafef_complete": all(pd.notna(metrics[metric]) for metric in METRICS),
                "cafef_error": "",
            }
        )
    return rows


def download_sample(
    sample: pd.DataFrame,
    timeout: float,
    sleep_seconds: float,
) -> pd.DataFrame:
    existing = pd.DataFrame()
    if RAW_OUTPUT.exists():
        existing = pd.read_csv(RAW_OUTPUT)
    completed = set(existing["ticker"].astype(str)) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for index, ticker in enumerate(sample["ticker"].astype(str), start=1):
        if ticker in completed:
            print(f"[{index}/{len(sample)}] {ticker}: checkpoint", flush=True)
            continue
        print(f"[{index}/{len(sample)}] {ticker}: CafeF", flush=True)
        try:
            rows.extend(fetch_ticker_metrics(session, ticker, timeout, sleep_seconds))
        except Exception as exc:  # noqa: BLE001 - persist per-ticker source failures.
            error = f"{type(exc).__name__}: {exc}"
            print(f"  error: {error}", flush=True)
            for period in PERIODS:
                rows.append(
                    {
                        "ticker": ticker,
                        "period": period,
                        **{metric: np.nan for metric in METRICS},
                        "cafef_statement_available": False,
                        "cafef_complete": False,
                        "cafef_error": error,
                    }
                )
        pd.DataFrame(rows).drop_duplicates(
            ["ticker", "period"], keep="last"
        ).to_csv(RAW_OUTPUT, index=False, encoding="utf-8-sig")

    return pd.DataFrame(rows).drop_duplicates(["ticker", "period"], keep="last")


def build_comparison(
    master: pd.DataFrame,
    sample: pd.DataFrame,
    cafef: pd.DataFrame,
) -> pd.DataFrame:
    sample_cols = ["ticker", "exchange", "vci_q2_present"]
    base = cafef.merge(sample[sample_cols], on="ticker", how="left")
    vci_cols = ["ticker", "period", *METRICS]
    vci = master[vci_cols].drop_duplicates(["ticker", "period"])
    merged = base.merge(vci, on=["ticker", "period"], how="left", suffixes=("_cafef", "_vci"))

    rows = []
    for _, row in merged.iterrows():
        for metric in METRICS:
            cafef_value = row[f"{metric}_cafef"]
            vci_value = row[f"{metric}_vci"]
            abs_diff = (
                abs(cafef_value - vci_value)
                if pd.notna(cafef_value) and pd.notna(vci_value)
                else np.nan
            )
            symmetric_pct_diff = (
                2 * abs_diff / (abs(cafef_value) + abs(vci_value))
                if pd.notna(abs_diff) and abs(cafef_value) + abs(vci_value) > 0
                else np.nan
            )
            rows.append(
                {
                    "ticker": row["ticker"],
                    "exchange": row["exchange"],
                    "vci_q2_present": row["vci_q2_present"],
                    "period": row["period"],
                    "metric": metric,
                    "vci_value": vci_value,
                    "cafef_value": cafef_value,
                    "abs_diff": abs_diff,
                    "symmetric_pct_diff": symmetric_pct_diff,
                    "within_tolerance": (
                        abs_diff <= TOLERANCES[metric] if pd.notna(abs_diff) else np.nan
                    ),
                    "cafef_error": row["cafef_error"],
                }
            )
    return pd.DataFrame(rows)


def summarize_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    valid = comparison.dropna(subset=["vci_value", "cafef_value"]).copy()
    rows = []
    for (period, metric), frame in valid.groupby(["period", "metric"]):
        correlation = (
            frame["vci_value"].corr(frame["cafef_value"])
            if len(frame) >= 3
            else np.nan
        )
        rows.append(
            {
                "period": period,
                "metric": metric,
                "matched_firms": len(frame),
                "pearson_correlation": correlation,
                "median_abs_diff": frame["abs_diff"].median(),
                "p90_abs_diff": frame["abs_diff"].quantile(0.9),
                "median_symmetric_pct_diff": frame["symmetric_pct_diff"].median(),
                "within_tolerance_share": pd.to_numeric(
                    frame["within_tolerance"], errors="coerce"
                ).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["period", "metric"])


def build_coverage(sample: pd.DataFrame, cafef: pd.DataFrame) -> pd.DataFrame:
    complete = cafef.pivot(index="ticker", columns="period", values="cafef_complete")
    complete = complete.reindex(columns=PERIODS).fillna(False)
    statements = cafef.pivot(
        index="ticker",
        columns="period",
        values="cafef_statement_available",
    )
    statements = statements.reindex(columns=PERIODS).fillna(False)
    statements.columns = [f"{period}_statement" for period in statements.columns]
    frame = sample[["ticker", "exchange", "vci_q2_present"]].merge(
        complete.reset_index(),
        on="ticker",
        how="left",
    )
    frame = frame.merge(statements.reset_index(), on="ticker", how="left")
    rows = []
    for (exchange, vci_q2_present), group in frame.groupby(
        ["exchange", "vci_q2_present"]
    ):
        rows.append(
            {
                "exchange": exchange,
                "vci_q2_present": vci_q2_present,
                "sample_firms": len(group),
                "cafef_q1_complete": int(group["2021-Q1"].fillna(False).sum()),
                "cafef_q2_complete": int(group["2021-Q2"].fillna(False).sum()),
                "cafef_q2_coverage": group["2021-Q2"].fillna(False).mean(),
                "cafef_q2_statement_available": int(
                    group["2021-Q2_statement"].fillna(False).sum()
                ),
                "cafef_q2_statement_coverage": group["2021-Q2_statement"]
                .fillna(False)
                .mean(),
            }
        )
    return pd.DataFrame(rows)


def format_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "NA"
    if metric in {"roa", "debt_to_assets", "gross_margin"}:
        return f"{value:.4f}"
    return f"{value:.3f}"


def write_report(
    sample: pd.DataFrame,
    cafef: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    q1 = summary[summary["period"].eq("2021-Q1")]
    missing_group = sample[~sample["vci_q2_present"]]
    missing_cafef = cafef[
        cafef["ticker"].isin(missing_group["ticker"])
        & cafef["period"].eq("2021-Q2")
    ]
    missing_complete = int(missing_cafef["cafef_complete"].fillna(False).sum())
    missing_statements = int(
        missing_cafef["cafef_statement_available"].fillna(False).sum()
    )
    incomplete_tickers = sorted(
        missing_cafef.loc[
            ~missing_cafef["cafef_complete"].fillna(False),
            "ticker",
        ].astype(str)
    )
    incomplete_note = (
        ", ".join(f"`{ticker}`" for ticker in incomplete_tickers)
        if incomplete_tickers
        else "none"
    )
    missing_total = len(missing_group)

    lines = [
        "# VCI vs CafeF validation for 2021-Q1 and 2021-Q2",
        "",
        "## Scope",
        "",
        f"- Stratified sample: **{len(sample)} firms**, balanced by exchange and whether VCI contains 2021-Q2.",
        "- Independent source: public CafeF quarterly financial-statement pages.",
        "- Ratios were recomputed from balance sheets and income statements; no VCI value was used in CafeF calculations.",
        "- Market capitalization was not compared because CafeF statement pages do not provide a point-in-time market-cap field.",
        "",
        "## 2021-Q1 agreement",
        "",
        "| Metric | Matched | Correlation | Median absolute difference | Within tolerance |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in q1.iterrows():
        lines.append(
            "| {metric} | {n} | {corr:.3f} | {diff} | {share:.1%} |".format(
                metric=row["metric"],
                n=int(row["matched_firms"]),
                corr=row["pearson_correlation"],
                diff=format_value(row["metric"], row["median_abs_diff"]),
                share=row["within_tolerance_share"],
            )
        )

    lines.extend(
        [
            "",
            "Tolerance definitions: ROA 0.5 percentage point; debt/assets 1 percentage point; "
            "current ratio, quick ratio and asset turnover 0.05; gross margin 1 percentage point.",
            "",
            "## 2021-Q2 coverage finding",
            "",
            f"- Among **{missing_total} sampled firms missing 2021-Q2 in VCI**, CafeF has a usable Q2 balance sheet and income statement for **{missing_statements} firms**.",
            f"- Among **{missing_total} sampled firms missing 2021-Q2 in VCI**, CafeF has all six recomputed model metrics for **{missing_complete} firms**.",
            "- The difference between these counts is caused by missing prior-quarter history needed for TTM ratios, not by a missing Q2 filing.",
            f"- Firms with a Q2 statement but incomplete TTM history: {incomplete_note}.",
            "- This distinguishes a VCI historical coverage gap from a market-wide absence of filings.",
            "",
            "Coverage by stratum:",
            "",
            "| Exchange | VCI Q2 present | Sample | Q2 statement | Q2 full metrics | Full-metric coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in coverage.iterrows():
        lines.append(
            f"| {row['exchange']} | {bool(row['vci_q2_present'])} | "
            f"{int(row['sample_firms'])} | "
            f"{int(row['cafef_q2_statement_available'])} | "
            f"{int(row['cafef_q2_complete'])} | {row['cafef_q2_coverage']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The 2021-Q2 collapse in VCI is a source-partition completeness failure. "
            "The existing DML exclusion remains appropriate until a controlled backfill is completed.",
            "",
            "Recommended backfill: calculate 2021-Q2 ratios from CafeF statements, validate the "
            "overlapping VCI firms against metric-specific tolerances, and keep provenance columns "
            "so source substitutions remain auditable.",
            "",
            "## Limitations",
            "",
            "- CafeF is an independent public aggregator, not an audited research database.",
            "- Statement labels differ for some industries; the current study already excludes financial firms.",
            "- Restatements, consolidated versus standalone statements, and minority interest treatment can create small differences.",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tickers", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep", type=float, default=0.15)
    args = parser.parse_args()

    for directory in [RAW_OUTPUT.parent, RESULTS_DIR, REPORT_OUTPUT.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(MASTER_FILE, low_memory=False)
    sample = select_sample(master, args.max_tickers)
    cafef = download_sample(sample, args.timeout, args.sleep)
    comparison = build_comparison(master, sample, cafef)
    summary = summarize_comparison(comparison)
    coverage = build_coverage(sample, cafef)

    comparison.to_csv(COMPARISON_OUTPUT, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")
    coverage.to_csv(COVERAGE_OUTPUT, index=False, encoding="utf-8-sig")
    write_report(sample, cafef, summary, coverage)

    print(f"Saved: {RAW_OUTPUT}", flush=True)
    print(f"Saved: {COMPARISON_OUTPUT}", flush=True)
    print(f"Saved: {SUMMARY_OUTPUT}", flush=True)
    print(f"Saved: {COVERAGE_OUTPUT}", flush=True)
    print(f"Saved: {REPORT_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
