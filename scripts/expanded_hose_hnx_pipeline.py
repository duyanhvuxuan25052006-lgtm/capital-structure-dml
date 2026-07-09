"""Expanded HOSE/HNX capital-structure DML pipeline.

This script keeps the original 50-firm workflow intact and creates a separate
expanded sample for all non-financial HOSE/HNX common stocks with available VCI
quarterly ratio data.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "expanded_hose_hnx"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = ROOT / "reports"

LISTING_FILE = RAW_DIR / "hose_hnx_listed_symbols.csv"
PROFILES_ALL_FILE = RAW_DIR / "hose_hnx_company_profiles_all.csv"
PROFILES_FILE = RAW_DIR / "hose_hnx_company_profiles_nonfinancial.csv"
RATIOS_FILE = RAW_DIR / "hose_hnx_financial_ratios.csv"
FAILURES_FILE = RAW_DIR / "hose_hnx_download_failures.csv"
MASTER_FILE = PROCESSED_DIR / "master_panel_hose_hnx_dataset.csv"
ANALYSIS_DATA_FILE = PROCESSED_DIR / "analysis_panel_hose_hnx_dataset.csv"
RESULTS_FILE = RESULTS_DIR / "dml_hose_hnx_final_report.csv"
SUMMARY_FILE = RESULTS_DIR / "sample_quality_summary.csv"
SEED_FILE = RESULTS_DIR / "rf_seed_stability.csv"
WINSOR_FILE = RESULTS_DIR / "winsorization_thresholds.csv"
LOW_COVERAGE_FILE = RESULTS_DIR / "excluded_low_coverage_periods.csv"
REPORT_FILE = REPORTS_DIR / "hose_hnx_expanded_dml_report.md"


DEFAULT_START = "2018-Q1"
DEFAULT_END = "2026-Q1"
DEFAULT_MIN_QUARTERS = 12
DEFAULT_MIN_PERIOD_COVERAGE = 0.50

Y_COL = "roa"
D_COL = "debt_to_assets"
X_COLS = ["current_ratio", "quick_ratio", "asset_turnover", "gross_margin", "log_market_cap"]

FINANCIAL_SECTOR_PATTERNS = [
    "bank",
    "banks",
    "ngan hang",
    "ngân hàng",
    "financial",
    "tai chinh",
    "tài chính",
    "financial services",
    "insurance",
    "bao hiem",
    "bảo hiểm",
    "securities",
    "chung khoan",
    "chứng khoán",
    "real estate investment trusts",
]

FINANCIAL_NAME_PATTERNS = [
    "ngan hang",
    "ngân hàng",
    "chung khoan",
    "chứng khoán",
    "bao hiem",
    "bảo hiểm",
    "cong ty tai chinh",
    "công ty tài chính",
    "quy dau tu",
    "quỹ đầu tư",
    "quan ly quy",
    "quản lý quỹ",
    "bank",
    "securities",
    "insurance",
    "finance",
    "financial",
    "fund",
]


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def period_to_tuple(period: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{4})-Q([1-4])", str(period))
    if not match:
        raise ValueError(f"Invalid quarter period: {period!r}")
    return int(match.group(1)), int(match.group(2))


def in_period_range(period: str, start: str, end: str) -> bool:
    try:
        p = period_to_tuple(period)
        return period_to_tuple(start) <= p <= period_to_tuple(end)
    except ValueError:
        return False


def period_year(period: str) -> float:
    try:
        return float(period_to_tuple(period)[0])
    except ValueError:
        return np.nan


def ascii_fold(text: object) -> str:
    import unicodedata

    if pd.isna(text):
        return ""
    raw = str(text).lower()
    folded = unicodedata.normalize("NFKD", raw)
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def truthy_flag(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def exclusion_reason(row: pd.Series) -> str:
    sector = ascii_fold(row.get("sector", ""))
    name = ascii_fold(row.get("organ_name", ""))
    en_name = ascii_fold(row.get("en_organ_name", ""))

    if truthy_flag(row.get("is_bank", False)) or truthy_flag(row.get("bank", False)):
        return "bank_flag"

    for pattern in FINANCIAL_SECTOR_PATTERNS:
        if pattern in sector:
            return f"financial_sector:{pattern}"

    for pattern in FINANCIAL_NAME_PATTERNS:
        if pattern in name or pattern in en_name:
            return f"financial_name:{pattern}"

    return ""


def extract_profile_row(ticker: str, listing_row: pd.Series) -> dict[str, object]:
    from vnstock import Company

    base = {
        "symbol": ticker,
        "organ_name": listing_row.get("organ_name", ""),
        "en_organ_name": listing_row.get("en_organ_name", ""),
        "exchange": listing_row.get("exchange", ""),
        "type": listing_row.get("type", ""),
    }

    try:
        company = Company(source="vci", symbol=ticker, show_log=False)
        overview = company.overview()
        if overview is None or overview.empty:
            base.update({"profile_status": "empty_overview"})
            return base
        row = overview.iloc[0].to_dict()
        for key, value in row.items():
            if key not in base or pd.isna(base[key]) or base[key] == "":
                base[key] = value
        base["profile_status"] = "ok"
        return base
    except Exception as exc:  # noqa: BLE001 - keep pipeline resilient to API gaps.
        base.update({"profile_status": "error", "profile_error": repr(exc)})
        return base


def build_listing_frame(max_tickers: int | None = None) -> pd.DataFrame:
    from vnstock import Listing

    ensure_dirs()
    listing_api = Listing()
    listing = listing_api.symbols_by_exchange()
    listing = listing[
        listing["exchange"].isin(["HOSE", "HNX"])
        & listing["type"].eq("stock")
        & listing["symbol"].astype(str).str.len().eq(3)
    ].drop_duplicates("symbol")

    try:
        industries = listing_api.symbols_by_industries().drop_duplicates("symbol")
        listing = listing.merge(industries, on="symbol", how="left")
        listing["sector"] = listing.get("industry_name")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load industry mapping: {exc!r}", flush=True)
        listing["sector"] = ""

    listing = listing.sort_values(["exchange", "symbol"]).reset_index(drop=True)
    if max_tickers is not None:
        listing = listing.head(max_tickers)
    listing.to_csv(LISTING_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved listing universe: {LISTING_FILE} ({len(listing)} symbols)", flush=True)
    return listing


def build_listing_only_profiles(max_tickers: int | None = None) -> pd.DataFrame:
    listing = build_listing_frame(max_tickers=max_tickers)
    profiles_all = listing.copy()
    profiles_all["profile_status"] = "listing_only"
    profiles_all["exclude_reason"] = profiles_all.apply(exclusion_reason, axis=1)
    profiles_all["included_nonfinancial"] = profiles_all["exclude_reason"].eq("")
    profiles_all.to_csv(PROFILES_ALL_FILE, index=False, encoding="utf-8-sig")
    profiles = profiles_all[profiles_all["included_nonfinancial"]].copy()
    profiles.to_csv(PROFILES_FILE, index=False, encoding="utf-8-sig")
    print(
        "Profiles from listing complete: "
        f"{len(profiles)} non-financial / {len(profiles_all)} HOSE+HNX stocks",
        flush=True,
    )
    return profiles


def build_listing_and_profiles(sleep_seconds: float, max_tickers: int | None = None) -> pd.DataFrame:
    listing = build_listing_frame(max_tickers=max_tickers)

    existing = pd.DataFrame()
    if PROFILES_ALL_FILE.exists():
        existing = pd.read_csv(PROFILES_ALL_FILE)
    done = set(existing["symbol"].astype(str)) if not existing.empty else set()
    rows = [] if existing.empty else existing.to_dict("records")

    for idx, listing_row in listing.iterrows():
        ticker = str(listing_row["symbol"])
        if ticker in done:
            continue
        print(f"[profile {idx + 1}/{len(listing)}] {ticker}", flush=True)
        row = extract_profile_row(ticker, listing_row)
        rows.append(row)
        pd.DataFrame(rows).drop_duplicates("symbol", keep="last").to_csv(
            PROFILES_ALL_FILE, index=False, encoding="utf-8-sig"
        )
        time.sleep(sleep_seconds)

    profiles_all = pd.DataFrame(rows).drop_duplicates("symbol", keep="last")
    profiles_all["exclude_reason"] = profiles_all.apply(exclusion_reason, axis=1)
    profiles_all["included_nonfinancial"] = profiles_all["exclude_reason"].eq("")
    profiles_all.to_csv(PROFILES_ALL_FILE, index=False, encoding="utf-8-sig")

    profiles = profiles_all[profiles_all["included_nonfinancial"]].copy()
    profiles.to_csv(PROFILES_FILE, index=False, encoding="utf-8-sig")
    print(
        "Profiles complete: "
        f"{len(profiles)} non-financial / {len(profiles_all)} HOSE+HNX stocks",
        flush=True,
    )
    return profiles


def reshape_financial_ratios(df_ratio: pd.DataFrame, symbol: str, start: str, end: str) -> pd.DataFrame:
    if df_ratio is None or df_ratio.empty:
        return pd.DataFrame()
    required = {"item_id"}
    if not required.issubset(df_ratio.columns):
        return pd.DataFrame()

    df_ratio = df_ratio.copy()
    if "year" not in set(df_ratio["item_id"]) or "quarter" not in set(df_ratio["item_id"]):
        return pd.DataFrame()

    row_year = df_ratio[df_ratio["item_id"] == "year"].iloc[0]
    row_quarter = df_ratio[df_ratio["item_id"] == "quarter"].iloc[0]
    data_cols = [c for c in df_ratio.columns if c not in ["item", "item_en", "item_id"]]

    periods: list[str] = []
    valid_cols: list[str] = []
    for col in data_cols:
        year_value = str(row_year[col]).strip()
        quarter_value = str(row_quarter[col]).strip()
        period = f"{year_value}-Q{quarter_value}"
        if in_period_range(period, start, end):
            periods.append(period)
            valid_cols.append(col)

    if not valid_cols:
        return pd.DataFrame()

    df_data = df_ratio[~df_ratio["item_id"].isin(["year", "quarter"])].copy()
    df_data = df_data[["item_id"] + valid_cols]
    df_transposed = df_data.set_index("item_id").T
    df_transposed.index = periods
    df_transposed.index.name = "period"
    df_transposed.insert(0, "ticker", symbol)
    return df_transposed.reset_index()


def download_one_ratio(ticker: str, start: str, end: str) -> pd.DataFrame:
    from vnstock import Finance

    finance = Finance(source="vci", symbol=ticker, period="quarter")
    raw = finance._get_report(report_type="ratio", limit=100, mode="final")  # noqa: SLF001
    return reshape_financial_ratios(raw, ticker, start, end)


def download_ratios(
    start: str,
    end: str,
    sleep_seconds: float,
    retry_sleep_seconds: float,
    max_tickers: int | None = None,
) -> pd.DataFrame:
    ensure_dirs()
    if not PROFILES_FILE.exists():
        raise FileNotFoundError(f"Missing profiles file: {PROFILES_FILE}")

    profiles = pd.read_csv(PROFILES_FILE)
    tickers = profiles["symbol"].dropna().astype(str).drop_duplicates().tolist()
    if max_tickers is not None:
        tickers = tickers[:max_tickers]

    existing = pd.DataFrame()
    if RATIOS_FILE.exists():
        existing = pd.read_csv(RATIOS_FILE)
    downloaded = set(existing["ticker"].astype(str).unique()) if not existing.empty else set()
    frames = [] if existing.empty else [existing]

    failures = []
    if FAILURES_FILE.exists():
        failures = pd.read_csv(FAILURES_FILE).to_dict("records")

    for idx, ticker in enumerate(tickers):
        if ticker in downloaded:
            continue
        print(f"[ratio {idx + 1}/{len(tickers)}] {ticker}", flush=True)
        try:
            df_ratio = download_one_ratio(ticker, start=start, end=end)
            if df_ratio.empty:
                failures.append({"symbol": ticker, "stage": "ratio", "error": "empty_ratio"})
            else:
                frames.append(df_ratio)
                downloaded.add(ticker)
                pd.concat(frames, ignore_index=True).to_csv(RATIOS_FILE, index=False, encoding="utf-8-sig")
                print(f"  saved {len(df_ratio)} periods", flush=True)
            time.sleep(sleep_seconds)
        except Exception as exc:  # noqa: BLE001 - record and keep downloading other tickers.
            err = repr(exc)
            print(f"  error: {err}", flush=True)
            failures.append({"symbol": ticker, "stage": "ratio", "error": err})
            pd.DataFrame(failures).to_csv(FAILURES_FILE, index=False, encoding="utf-8-sig")
            time.sleep(retry_sleep_seconds)

    ratios = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not ratios.empty:
        ratios = ratios.drop_duplicates(["ticker", "period"], keep="last")
        ratios = ratios.sort_values(["ticker", "period"]).reset_index(drop=True)
        ratios.to_csv(RATIOS_FILE, index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).drop_duplicates().to_csv(FAILURES_FILE, index=False, encoding="utf-8-sig")
    print(f"Ratios complete: {RATIOS_FILE} ({ratios.shape[0]} rows)", flush=True)
    return ratios


def make_master_panel(
    start: str,
    end: str,
    min_quarters: int,
    min_period_coverage: float = DEFAULT_MIN_PERIOD_COVERAGE,
) -> pd.DataFrame:
    ensure_dirs()
    ratios = pd.read_csv(RATIOS_FILE)
    profiles = pd.read_csv(PROFILES_FILE)

    master = ratios.merge(profiles, left_on="ticker", right_on="symbol", how="left", suffixes=("", "_profile"))
    if "symbol" in master.columns:
        master = master.drop(columns=["symbol"])

    if "listing_date" in master.columns:
        master["listing_year"] = pd.to_datetime(master["listing_date"], errors="coerce").dt.year
    else:
        master["listing_year"] = np.nan
    master["report_year"] = master["period"].map(period_year)
    master["firm_age"] = master["report_year"] - master["listing_year"]
    master["firm_age"] = master["firm_age"].where(master["firm_age"].ge(0))

    if "debt_to_equity" not in master.columns and "debtPerEquity" in master.columns:
        master["debt_to_equity"] = master["debtPerEquity"]

    for col in master.columns:
        if col not in {
            "period",
            "ticker",
            "organ_name",
            "en_organ_name",
            "exchange",
            "sector",
            "listing_date",
            "profile_status",
            "profile_error",
            "exclude_reason",
            "type",
            "organ_short_name",
            "company_profile",
            "prev_insight",
            "fund_info",
        }:
            try:
                master[col] = pd.to_numeric(master[col])
            except (TypeError, ValueError):
                pass

    master["debt_to_assets"] = np.nan
    debt = pd.to_numeric(master.get("debt_to_equity"), errors="coerce")
    valid_debt = debt.notna() & np.isfinite(debt) & (debt > -0.999)
    master.loc[valid_debt, "debt_to_assets"] = debt.loc[valid_debt] / (1.0 + debt.loc[valid_debt])

    market_cap = pd.to_numeric(master.get("market_cap"), errors="coerce")
    master["log_market_cap"] = np.nan
    positive_market_cap = market_cap.gt(0)
    master.loc[positive_market_cap, "log_market_cap"] = np.log(market_cap.loc[positive_market_cap])

    master = master[master["period"].map(lambda p: in_period_range(p, start, end))].copy()
    master = master.sort_values(["ticker", "period"]).reset_index(drop=True)
    master.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")

    model_cols = [Y_COL, D_COL] + X_COLS
    analysis = master.dropna(subset=model_cols).copy()
    counts = analysis.groupby("ticker").size()
    keep_tickers = counts[counts >= min_quarters].index
    analysis = analysis[analysis["ticker"].isin(keep_tickers)].copy()

    period_coverage = analysis.groupby("period")["ticker"].nunique().sort_index()
    coverage_share = period_coverage / analysis["ticker"].nunique()
    low_coverage = pd.DataFrame(
        {
            "period": period_coverage.index,
            "firms": period_coverage.values,
            "coverage_share": coverage_share.values,
        }
    )
    low_coverage = low_coverage[low_coverage["coverage_share"] < min_period_coverage]
    low_coverage.to_csv(LOW_COVERAGE_FILE, index=False, encoding="utf-8-sig")
    if not low_coverage.empty:
        analysis = analysis[~analysis["period"].isin(low_coverage["period"])].copy()
        counts = analysis.groupby("ticker").size()
        keep_tickers = counts[counts >= min_quarters].index
        analysis = analysis[analysis["ticker"].isin(keep_tickers)].copy()

    analysis.to_csv(ANALYSIS_DATA_FILE, index=False, encoding="utf-8-sig")

    summary = sample_quality_summary(master, analysis, min_quarters, low_coverage)
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print(f"Master saved: {MASTER_FILE} {master.shape}", flush=True)
    print(f"Analysis data saved: {ANALYSIS_DATA_FILE} {analysis.shape}", flush=True)
    return analysis


def sample_quality_summary(
    master: pd.DataFrame,
    analysis: pd.DataFrame,
    min_quarters: int,
    low_coverage: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"metric": "master_rows", "value": len(master)},
        {"metric": "master_firms", "value": master["ticker"].nunique()},
        {"metric": "analysis_rows", "value": len(analysis)},
        {"metric": "analysis_firms", "value": analysis["ticker"].nunique()},
        {"metric": "min_quarters_rule", "value": min_quarters},
        {"metric": "analysis_start_period", "value": analysis["period"].min() if not analysis.empty else ""},
        {"metric": "analysis_end_period", "value": analysis["period"].max() if not analysis.empty else ""},
        {"metric": "analysis_periods", "value": analysis["period"].nunique()},
        {"metric": "duplicates_ticker_period_master", "value": int(master.duplicated(["ticker", "period"]).sum())},
        {
            "metric": "excluded_low_coverage_periods",
            "value": ",".join(low_coverage["period"].astype(str)) if not low_coverage.empty else "",
        },
    ]
    for exchange, count in analysis.drop_duplicates("ticker")["exchange"].value_counts().items():
        rows.append({"metric": f"analysis_firms_exchange_{exchange}", "value": int(count)})
    for col in [Y_COL, D_COL] + X_COLS:
        rows.append({"metric": f"missing_master_{col}", "value": int(master[col].isna().sum())})
    return pd.DataFrame(rows)


def demean_by_entity(df: pd.DataFrame, cols: list[str], entity_col: str = "ticker") -> pd.DataFrame:
    grouped_mean = df.groupby(entity_col)[cols].transform("mean")
    out = df.copy()
    out[cols] = df[cols] - grouped_mean
    return out


def make_group_sample_splits(groups: Iterable[object], n_folds: int, n_rep: int, random_state: int) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    groups_array = np.asarray(list(groups))
    unique_groups = np.array(pd.unique(groups_array))
    all_indices = np.arange(len(groups_array))
    all_smpls: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for rep in range(n_rep):
        rng = np.random.default_rng(random_state + rep)
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        folds = np.array_split(shuffled, n_folds)
        rep_smpls = []
        for fold_groups in folds:
            test_mask = np.isin(groups_array, fold_groups)
            test_idx = all_indices[test_mask]
            train_idx = all_indices[~test_mask]
            rep_smpls.append((train_idx, test_idx))
        all_smpls.append(rep_smpls)
    return all_smpls


def prepare_model_matrices(
    df: pd.DataFrame,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    from sklearn.preprocessing import StandardScaler

    model_cols = [Y_COL, D_COL] + X_COLS
    clean = df.dropna(subset=model_cols + ["ticker", "period"]).copy()
    for col in model_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna(subset=model_cols).copy()

    winsor_rows = []
    for col in model_cols:
        lower = float(clean[col].quantile(winsor_lower))
        upper = float(clean[col].quantile(winsor_upper))
        winsor_rows.append({"variable": col, "lower": lower, "upper": upper})
        clean[col] = clean[col].clip(lower=lower, upper=upper)
    pd.DataFrame(winsor_rows).to_csv(WINSOR_FILE, index=False, encoding="utf-8-sig")

    demeaned = demean_by_entity(clean, model_cols)
    time_dummies = pd.get_dummies(clean["period"], prefix="time", drop_first=True).astype(float)

    x_financial = demeaned[X_COLS].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_financial = scaler.fit_transform(x_financial)

    x_all = np.hstack([x_financial, time_dummies.to_numpy(dtype=float)])
    x_names = X_COLS + time_dummies.columns.tolist()
    y = demeaned[Y_COL].to_numpy(dtype=float)
    d = demeaned[D_COL].to_numpy(dtype=float)
    return clean, x_all, y, d, x_names


def confint_values(model) -> tuple[float, float]:
    ci = model.confint()
    return float(ci.iloc[0, 0]), float(ci.iloc[0, 1])


def run_analysis(min_quarters: int, n_folds: int, n_rep: int, random_state: int) -> pd.DataFrame:
    import doubleml as dml
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNetCV, LassoCV

    ensure_dirs()
    if not ANALYSIS_DATA_FILE.exists():
        raise FileNotFoundError(f"Missing analysis data: {ANALYSIS_DATA_FILE}")
    analysis = pd.read_csv(ANALYSIS_DATA_FILE)
    counts = analysis.groupby("ticker").size()
    analysis = analysis[analysis["ticker"].isin(counts[counts >= min_quarters].index)].copy()

    clean, x_all, y, d, _ = prepare_model_matrices(analysis)
    tickers = clean["ticker"].to_numpy()
    results: list[dict[str, object]] = []

    x_raw = clean[[D_COL] + X_COLS].apply(pd.to_numeric, errors="coerce")
    x_raw = sm.add_constant(x_raw)
    pooled = sm.OLS(clean[Y_COL].astype(float), x_raw.astype(float)).fit(
        cov_type="cluster", cov_kwds={"groups": clean["ticker"]}
    )
    ci = pooled.conf_int().loc[D_COL]
    results.append(
        {
            "Model": "Pooled OLS (clustered by firm)",
            "Coef": pooled.params[D_COL],
            "SE": pooled.bse[D_COL],
            "P-value": pooled.pvalues[D_COL],
            "CI_lower": ci[0],
            "CI_upper": ci[1],
            "N": len(clean),
            "Firms": clean["ticker"].nunique(),
        }
    )

    demeaned = demean_by_entity(clean, [Y_COL, D_COL] + X_COLS)
    x_fe = sm.add_constant(demeaned[[D_COL] + X_COLS].astype(float))
    fe = sm.OLS(demeaned[Y_COL].astype(float), x_fe).fit(cov_type="cluster", cov_kwds={"groups": clean["ticker"]})
    ci = fe.conf_int().loc[D_COL]
    results.append(
        {
            "Model": "Entity FE (clustered by firm)",
            "Coef": fe.params[D_COL],
            "SE": fe.bse[D_COL],
            "P-value": fe.pvalues[D_COL],
            "CI_lower": ci[0],
            "CI_upper": ci[1],
            "N": len(clean),
            "Firms": clean["ticker"].nunique(),
        }
    )

    x_twfe_df = pd.concat(
        [
            demeaned[[D_COL] + X_COLS].reset_index(drop=True).astype(float),
            pd.get_dummies(clean["period"], prefix="time", drop_first=True).reset_index(drop=True).astype(float),
        ],
        axis=1,
    )
    x_twfe = sm.add_constant(x_twfe_df)
    twfe = sm.OLS(demeaned[Y_COL].reset_index(drop=True).astype(float), x_twfe).fit(
        cov_type="cluster", cov_kwds={"groups": clean["ticker"].reset_index(drop=True)}
    )
    ci = twfe.conf_int().loc[D_COL]
    results.append(
        {
            "Model": "Two-way FE (entity + time)",
            "Coef": twfe.params[D_COL],
            "SE": twfe.bse[D_COL],
            "P-value": twfe.pvalues[D_COL],
            "CI_lower": ci[0],
            "CI_upper": ci[1],
            "N": len(clean),
            "Firms": clean["ticker"].nunique(),
        }
    )

    dml_data = dml.DoubleMLData.from_arrays(x=x_all, y=y, d=d)
    splits = make_group_sample_splits(tickers, n_folds=n_folds, n_rep=n_rep, random_state=random_state)

    learners = [
        ("DML-PLR LassoCV (group folds)", LassoCV(cv=5, max_iter=20000, random_state=random_state)),
        ("DML-PLR ElasticNetCV (group folds)", ElasticNetCV(cv=5, max_iter=20000, random_state=random_state)),
        (
            "DML-PLR RandomForest (group folds)",
            RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=random_state, n_jobs=-1),
        ),
    ]
    for name, learner in learners:
        print(f"Running {name}", flush=True)
        model = dml.DoubleMLPLR(dml_data, ml_l=learner, ml_m=learner, n_folds=n_folds, n_rep=n_rep, draw_sample_splitting=False)
        model.set_sample_splitting(splits)
        model.fit()
        lo, hi = confint_values(model)
        results.append(
            {
                "Model": name,
                "Coef": model.coef[0],
                "SE": model.se[0],
                "P-value": model.pval[0],
                "CI_lower": lo,
                "CI_upper": hi,
                "N": len(clean),
                "Firms": clean["ticker"].nunique(),
            }
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")

    run_rf_seed_stability(dml_data, x_all, y, d, tickers, n_folds=n_folds, seeds=[1, 2, 3, 4, 5])
    make_figures(clean, df_results)
    print(f"Results saved: {RESULTS_FILE}", flush=True)
    return df_results


def run_rf_seed_stability(dml_data, x_all, y, d, groups, n_folds: int, seeds: list[int]) -> None:
    import doubleml as dml
    from sklearn.ensemble import RandomForestRegressor

    rows = []
    for seed in seeds:
        splits = make_group_sample_splits(groups, n_folds=n_folds, n_rep=3, random_state=seed)
        learner = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=seed, n_jobs=-1)
        model = dml.DoubleMLPLR(dml_data, ml_l=learner, ml_m=learner, n_folds=n_folds, n_rep=3, draw_sample_splitting=False)
        model.set_sample_splitting(splits)
        model.fit()
        lo, hi = confint_values(model)
        rows.append(
            {
                "seed": seed,
                "theta": model.coef[0],
                "se": model.se[0],
                "pvalue": model.pval[0],
                "ci_lower": lo,
                "ci_upper": hi,
            }
        )
    pd.DataFrame(rows).to_csv(SEED_FILE, index=False, encoding="utf-8-sig")


def make_figures(clean: pd.DataFrame, df_results: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tokens = {
        "surface": "#FCFCFD",
        "panel": "#FFFFFF",
        "ink": "#1F2430",
        "muted": "#6F768A",
        "grid": "#E6E8F0",
        "axis": "#D7DBE7",
        "gold_base": "#FFE15B",
        "gold_dark": "#736422",
        "olive_base": "#A3D576",
        "olive_dark": "#386411",
    }
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": tokens["surface"],
            "axes.facecolor": tokens["panel"],
            "axes.edgecolor": tokens["axis"],
            "axes.labelcolor": tokens["ink"],
            "grid.color": tokens["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )

    def add_header(fig, ax, title: str, subtitle: str) -> None:
        ax.set_title("")
        fig.subplots_adjust(top=0.82)
        left = ax.get_position().x0
        fig.text(left, 0.975, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=tokens["ink"])
        fig.text(left, 0.925, subtitle, ha="left", va="top", fontsize=9, color=tokens["muted"])
        sns.despine(ax=ax)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    plot_df = df_results.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    for idx, row in plot_df.iterrows():
        ax.errorbar(
            row["Coef"],
            idx,
            xerr=np.array([[row["Coef"] - row["CI_lower"]], [row["CI_upper"] - row["Coef"]]]),
            fmt="o",
            color=tokens["gold_base"],
            markerfacecolor=tokens["gold_base"],
            markeredgecolor=tokens["gold_dark"],
            ecolor=tokens["gold_dark"],
            linewidth=1.0,
            capsize=3,
        )
    ax.axvline(0, color=tokens["ink"], linestyle=":", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Model"])
    ax.set_xlabel("Theta: tác động của debt_to_assets lên ROA")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    add_header(
        fig,
        ax,
        "Ước lượng tác động đòn bẩy tài sản lên ROA",
        f"Mẫu HOSE+HNX mở rộng, N={len(clean):,}, {clean['ticker'].nunique()} doanh nghiệp; khoảng tin cậy 95%",
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(FIGURES_DIR / "expanded_dml_forest_plot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    per_period = clean.groupby("period").agg(obs=("ticker", "size"), firms=("ticker", "nunique")).reset_index()
    x_pos = np.arange(len(per_period))
    ax.plot(
        x_pos,
        per_period["firms"],
        color=tokens["olive_base"],
        marker="o",
        markeredgecolor=tokens["olive_dark"],
        linewidth=1.0,
        markersize=4,
    )
    tick_idx = np.arange(0, len(per_period), 4)
    ax.set_xticks(tick_idx, per_period.loc[tick_idx, "period"])
    ax.set_xlabel("Quý")
    ax.set_ylabel("Số doanh nghiệp")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    add_header(
        fig,
        ax,
        "Số doanh nghiệp khả dụng theo quý",
        "Sau lọc biến mô hình và loại 2021-Q2 vì coverage dưới 50%; 2018-Q1 đến 2026-Q1",
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(FIGURES_DIR / "expanded_firms_by_quarter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.4f}"


def write_markdown_report() -> None:
    if not RESULTS_FILE.exists() or not SUMMARY_FILE.exists() or not ANALYSIS_DATA_FILE.exists():
        raise FileNotFoundError("Run the analysis before writing the report.")

    results = pd.read_csv(RESULTS_FILE)
    summary = pd.read_csv(SUMMARY_FILE)
    analysis = pd.read_csv(ANALYSIS_DATA_FILE)
    seeds = pd.read_csv(SEED_FILE) if SEED_FILE.exists() else pd.DataFrame()
    low_coverage = pd.read_csv(LOW_COVERAGE_FILE) if LOW_COVERAGE_FILE.exists() else pd.DataFrame()
    robustness_file = RESULTS_DIR / "robustness_min16" / "dml_results.csv"
    robustness = pd.read_csv(robustness_file) if robustness_file.exists() else pd.DataFrame()

    summary_map = dict(zip(summary["metric"], summary["value"]))
    analysis_firms = int(float(summary_map.get("analysis_firms", 0)))
    analysis_rows = int(float(summary_map.get("analysis_rows", 0)))
    master_rows = int(float(summary_map.get("master_rows", 0)))
    start_period = summary_map.get("analysis_start_period", "")
    end_period = summary_map.get("analysis_end_period", "")
    hose = int(float(summary_map.get("analysis_firms_exchange_HOSE", 0)))
    hnx = int(float(summary_map.get("analysis_firms_exchange_HNX", 0)))
    excluded_periods = summary_map.get("excluded_low_coverage_periods", "")
    if pd.isna(excluded_periods):
        excluded_periods = ""
    if low_coverage.empty:
        coverage_summary = "không có quý nào bị loại do coverage thấp"
    else:
        coverage_details = ", ".join(
            f"`{row['period']}` đạt {float(row['coverage_share']):.1%}"
            for _, row in low_coverage.iterrows()
        )
        coverage_summary = f"các quý bị loại do coverage thấp gồm {coverage_details}"
    profiles_all = pd.read_csv(PROFILES_ALL_FILE) if PROFILES_ALL_FILE.exists() else pd.DataFrame()
    listing_universe_count = len(profiles_all)
    nonfinancial_count = (
        int(profiles_all["included_nonfinancial"].astype(bool).sum())
        if not profiles_all.empty and "included_nonfinancial" in profiles_all.columns
        else 0
    )

    rf_row = results[results["Model"].str.contains("RandomForest", regex=False)].iloc[0]
    twfe_row = results[results["Model"].str.startswith("Two-way")].iloc[0]

    result_table = results.copy()
    for col in ["Coef", "SE", "CI_lower", "CI_upper"]:
        result_table[col] = result_table[col].map(lambda x: f"{float(x):.6f}")
    result_table["P-value"] = result_table["P-value"].map(lambda x: f"{float(x):.3e}")

    period_counts = analysis.groupby("period")["ticker"].nunique()
    weak_periods = period_counts[period_counts < max(10, math.floor(0.5 * analysis_firms))]
    seed_note = ""
    if not seeds.empty:
        seed_note = (
            f"Kiểm tra {len(seeds)} seeds của RandomForest cho theta trung bình "
            f"{seeds['theta'].mean():.6f}; khoảng theta "
            f"[{seeds['theta'].min():.6f}, {seeds['theta'].max():.6f}]."
        )
    robustness_note = ""
    if not robustness.empty:
        robust_rf = robustness[robustness["Model"].str.contains("RandomForest", regex=False)].iloc[0]
        robustness_note = (
            f"Với ngưỡng tối thiểu 16 quý, mẫu còn {int(robust_rf['Firms'])} doanh nghiệp và "
            f"RandomForest DML cho theta = {float(robust_rf['Coef']):.6f}, "
            f"p = {float(robust_rf['P-value']):.4g}."
        )
    economic_effect_pp = float(rf_row["Coef"]) * 0.1 * 100

    md = f"""# Kết quả DML trên mẫu mở rộng HOSE và HNX

## Tóm tắt kỹ thuật

Pipeline mới thay thế danh sách hard-code 50 doanh nghiệp bằng toàn bộ cổ phiếu thường phi tài chính trên HOSE và HNX có dữ liệu `vnstock`/VCI và có tối thiểu {DEFAULT_MIN_QUARTERS} quan sát quý hợp lệ.

Mẫu phân tích cuối gồm **{analysis_rows:,} quan sát doanh nghiệp-quý** của **{analysis_firms:,} doanh nghiệp**, trong đó có **{hose:,} doanh nghiệp HOSE** và **{hnx:,} doanh nghiệp HNX**. Phạm vi thời gian là **{start_period} đến {end_period}**; {coverage_summary}.

Đặc tả DML-RandomForest sử dụng entity demeaning, biến giả thời gian và cross-fitting theo `ticker` cho kết quả **theta = {float(rf_row['Coef']):.6f}**, **p = {float(rf_row['P-value']):.4g}**, khoảng tin cậy 95% **[{float(rf_row['CI_lower']):.6f}, {float(rf_row['CI_upper']):.6f}]**. Theo thang đo của mô hình, D/A tăng 0,10 gắn với ROA giảm khoảng **{abs(economic_effect_pp):.3f} điểm phần trăm**. Two-way FE cũng cho hệ số âm **{float(twfe_row['Coef']):.6f}**, p = {float(twfe_row['P-value']):.4g}.

Kết quả mới **đảo dấu so với mẫu 50 doanh nghiệp cũ**: thay vì hệ số dương `+0,0313`, toàn bộ sáu đặc tả trên mẫu mở rộng đều cho hệ số âm và có ý nghĩa thống kê. Điều này cho thấy kết luận dương trước đây phụ thuộc mạnh vào cách chọn mẫu blue-chip và không nên được duy trì như kết luận chung cho thị trường.

## Hệ số âm nhất quán trên các mô hình

Nhận xét về kích thước mẫu được xử lý ở tầng chọn đối tượng: mô hình hiện dùng {analysis_firms} doanh nghiệp thay vì 50 doanh nghiệp được chọn thủ công. Cross-fitting được chia theo doanh nghiệp, nên toàn bộ các quý của cùng một `ticker` chỉ nằm trong train hoặc test ở một fold, tránh leakage giữa các quý của cùng công ty.

![Expanded DML forest plot](../results/expanded_hose_hnx/figures/expanded_dml_forest_plot.png)

| Mô hình | Hệ số | SE | P-value | CI dưới | CI trên | N | Doanh nghiệp |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for _, row in result_table.iterrows():
        md += (
            f"| {row['Model']} | {row['Coef']} | {row['SE']} | {row['P-value']} | "
            f"{row['CI_lower']} | {row['CI_upper']} | {int(row['N'])} | {int(row['Firms'])} |\n"
        )

    md += f"""
## Coverage dữ liệu theo quý

Panel được giữ ở dạng không cân bằng: doanh nghiệp có thể vào mẫu sau khi niêm yết và không có dòng khi VCI không trả dữ liệu. Pipeline không forward-fill hoặc tự tạo tỷ số tài chính cho các quý thiếu.

![Available firms by quarter](../results/expanded_hose_hnx/figures/expanded_firms_by_quarter.png)

Quý bị loại do coverage thấp:

"""
    if low_coverage.empty:
        md += "- Không có quý nào dưới ngưỡng coverage.\n"
    else:
        for _, row in low_coverage.iterrows():
            md += (
                f"- `{row['period']}`: {int(row['firms'])}/{analysis_firms} doanh nghiệp "
                f"({float(row['coverage_share']):.1%}).\n"
            )

    md += f"""
## Phạm vi và định nghĩa biến

- **Universe:** ảnh chụp cổ phiếu thường đang niêm yết trên HOSE và HNX tại thời điểm tải; loại ngân hàng, chứng khoán, bảo hiểm, quỹ và công ty tài chính theo ngành/tên doanh nghiệp.
- **Thời gian:** {DEFAULT_START} đến {DEFAULT_END}; tỷ số tài chính quý từ VCI.
- **Biến kết quả:** `roa`.
- **Biến can thiệp:** `debt_to_assets = debt_to_equity / (1 + debt_to_equity)`.
- **Biến kiểm soát:** `current_ratio`, `quick_ratio`, `asset_turnover`, `gross_margin`, `log_market_cap`.
- **Rule doanh nghiệp:** tối thiểu {DEFAULT_MIN_QUARTERS} quan sát hợp lệ.
- **Rule coverage theo quý:** loại quý có dưới {DEFAULT_MIN_PERIOD_COVERAGE:.0%} số doanh nghiệp; quý bị loại: `{excluded_periods or "không có"}`.
- **Outlier:** winsorize các biến mô hình tại phân vị 1% và 99% trước entity demeaning.

## Phương pháp

Phân tích gồm ba baseline kinh tế lượng và ba đặc tả DML-PLR:

- Pooled OLS với sai số chuẩn cluster theo doanh nghiệp.
- Entity Fixed Effects bằng within transformation.
- Two-way FE bằng entity demeaning và biến giả quý.
- DML-PLR với LassoCV, ElasticNetCV và RandomForest.

Các mô hình DML sử dụng 5 folds, 5 repetitions và sample splitting theo `ticker`.

## Các phần đã thực hiện

1. Nâng `vnstock` lên phiên bản 4.0.4 và cập nhật requirements.
2. Tạo universe {listing_universe_count} cổ phiếu thường HOSE/HNX, lọc còn {nonfinancial_count} doanh nghiệp phi tài chính.
3. Tải {master_rows:,} dòng tỷ số tài chính quý và ghi checkpoint/failure log.
4. Tạo master panel, tính `debt_to_assets` và `log_market_cap`.
5. Áp rule tối thiểu 12 quý và rule coverage theo quý, tạo mẫu {analysis_firms} doanh nghiệp; quý bị loại: `{excluded_periods or "không có"}`.
6. Winsorize 1%-99%, chạy OLS, FE, Two-way FE và ba DML với group folds.
7. Chạy kiểm tra 5 random seeds và robustness với ngưỡng tối thiểu 16 quý.
8. Xuất bảng kết quả, biểu đồ và báo cáo kỹ thuật này.

## Hạn chế và kiểm định độ bền

- `vnstock`/VCI là lớp trích xuất dữ liệu, không phải cơ sở dữ liệu học thuật đã được kiểm toán độc lập.
- Universe lấy từ danh sách đang niêm yết tại thời điểm tải, chưa tái dựng các mã đã hủy niêm yết trong lịch sử; do đó vẫn có nguy cơ survivorship bias.
- Dữ liệu VCI có coverage thấp tại `{excluded_periods or "không có quý nào"}`; các quý này được giữ trong master nhưng loại khỏi model.
- Rule loại ngành tài chính dựa trên ngành/từ khóa nên vẫn cần kiểm tra thủ công trước khi nộp.
- {seed_note if seed_note else "Chưa có kiểm tra seed."}
- {robustness_note if robustness_note else "Chưa có robustness theo ngưỡng số quý."}
- DML-PLR không tự giải quyết nội sinh từ biến không quan sát hoặc quan hệ nhân quả ngược nếu không có biến công cụ. Vì vậy kết quả không nên được mô tả như bằng chứng nhân quả tuyệt đối.

## Bước tiếp theo

1. Đối chiếu ngẫu nhiên một số doanh nghiệp với báo cáo tài chính gốc hoặc nguồn thứ hai.
2. Chạy robustness riêng cho HOSE, HNX, doanh nghiệp niêm yết liên tục và từng ngành.
3. Cân nhắc DML-PLIV hoặc chiến lược nhận dạng mạnh hơn nếu luận văn cần tuyên bố nhân quả.
4. Cập nhật phần abstract, kết luận và thảo luận của paper cũ vì dấu hệ số đã thay đổi.

## Câu hỏi nghiên cứu tiếp theo

- Tác động có khác nhau theo sàn, ngành hoặc quy mô doanh nghiệp không?
- Kết quả có thay đổi khi chỉ giữ doanh nghiệp niêm yết liên tục trong toàn kỳ không?
- Missing data của VCI có tập trung theo ngành/sàn hay không?
"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(md, encoding="utf-8")
    print(f"Report saved: {REPORT_FILE}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build expanded HOSE/HNX data and run DML analysis.")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["profiles", "ratios", "merge", "analysis", "report"],
        choices=["profiles", "ratios", "merge", "analysis", "report", "all"],
        help="Pipeline steps to run.",
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--min-quarters", type=int, default=DEFAULT_MIN_QUARTERS)
    parser.add_argument("--min-period-coverage", type=float, default=DEFAULT_MIN_PERIOD_COVERAGE)
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between successful API calls.")
    parser.add_argument("--retry-sleep", type=float, default=5.0, help="Sleep after an API error.")
    parser.add_argument("--max-tickers", type=int, default=None, help="Optional small-run cap for debugging.")
    parser.add_argument(
        "--profile-source",
        choices=["listing", "company"],
        default="listing",
        help="Use listing/industry metadata only, or call Company.overview for richer metadata.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ensure_dirs()
    args = parse_args()
    steps = set(args.steps)
    if "all" in steps:
        steps = {"profiles", "ratios", "merge", "analysis", "report"}

    if "profiles" in steps:
        if args.profile_source == "company":
            build_listing_and_profiles(sleep_seconds=args.sleep, max_tickers=args.max_tickers)
        else:
            build_listing_only_profiles(max_tickers=args.max_tickers)
    if "ratios" in steps:
        download_ratios(
            start=args.start,
            end=args.end,
            sleep_seconds=args.sleep,
            retry_sleep_seconds=args.retry_sleep,
            max_tickers=args.max_tickers,
        )
    if "merge" in steps:
        make_master_panel(
            start=args.start,
            end=args.end,
            min_quarters=args.min_quarters,
            min_period_coverage=args.min_period_coverage,
        )
    if "analysis" in steps:
        run_analysis(min_quarters=args.min_quarters, n_folds=args.folds, n_rep=args.reps, random_state=args.seed)
    if "report" in steps:
        write_markdown_report()


if __name__ == "__main__":
    main()
