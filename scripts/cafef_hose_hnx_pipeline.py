"""CafeF-only financial statement pipeline for non-financial HOSE/HNX firms.

The downloader uses CafeF's four-quarter financial statement pages, stores a
per-ticker checkpoint, derives a quarterly panel, and estimates OLS, fixed
effects, and clustered Double Machine Learning models.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "cafef_hose_hnx"
CACHE_DIR = RAW_DIR / "cafef_cache" / "components"

PROFILES_FILE = RAW_DIR / "hose_hnx_company_profiles_nonfinancial.csv"
COMPONENTS_FILE = RAW_DIR / "cafef_statement_components_2017_2025.csv"
STATUS_FILE = RAW_DIR / "cafef_download_status.csv"
FAILURES_FILE = RAW_DIR / "cafef_download_failures.csv"
MASTER_FILE = PROCESSED_DIR / "master_panel_cafef_hose_hnx_dataset.csv"
ANALYSIS_FILE = PROCESSED_DIR / "analysis_panel_cafef_hose_hnx_dataset.csv"

RESULTS_FILE = RESULTS_DIR / "dml_cafef_hose_hnx_final_report.csv"
ROBUSTNESS_FILE = RESULTS_DIR / "cafef_robustness_checks.csv"
SEED_FILE = RESULTS_DIR / "cafef_rf_seed_stability.csv"
QUALITY_FILE = RESULTS_DIR / "cafef_data_quality_summary.csv"
COVERAGE_FILE = RESULTS_DIR / "cafef_period_coverage.csv"
WINSOR_FILE = RESULTS_DIR / "cafef_winsorization_thresholds.csv"

START_PERIOD = "2018-Q1"
END_PERIOD = "2025-Q4"
FETCH_YEARS = range(2017, 2026)
Y_COL = "roa"
D_COL = "debt_to_assets"
X_COLS = [
    "current_ratio",
    "quick_ratio",
    "asset_turnover",
    "gross_margin",
    "log_total_assets",
]
MODEL_COLS = [Y_COL, D_COL, *X_COLS]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
)

COMPONENT_DEFINITIONS = {
    "total_assets": {"exact": ("tong cong tai san", "tong tai san", "tong tai san co")},
    "total_liabilities": {"exact": ("c no phai tra", "no phai tra", "tong no phai tra")},
    "equity": {
        "exact": ("d von chu so huu", "von chu so huu", "tong cong von chu so huu")
    },
    "current_assets": {"exact": ("a tai san ngan han", "tai san ngan han")},
    "current_liabilities": {"exact": ("i no ngan han", "no ngan han")},
    "inventory": {"exact": ("iv hang ton kho", "hang ton kho")},
    "net_revenue": {
        "contains": ("doanh thu thuan ve ban hang va cung cap dich vu",)
    },
    "gross_profit": {
        "contains": ("loi nhuan gop ve ban hang va cung cap dich vu",)
    },
    "parent_net_income": {
        "contains": ("loi nhuan sau thue cong ty me",),
        "excludes": ("khong kiem soat",),
    },
    "net_income": {
        "contains": ("loi nhuan sau thue thu nhap doanh nghiep",)
    },
}


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def fold_text(value: object) -> str:
    text = str(value).replace("\u0111", "d").replace("\u0110", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


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


def period_number(period: str) -> int:
    year, quarter = period.split("-Q")
    return int(year) * 4 + int(quarter) - 1


def period_from_number(value: int) -> str:
    year, quarter_index = divmod(value, 4)
    return f"{year}-Q{quarter_index + 1}"


def trailing_periods(period: str) -> list[str]:
    end = period_number(period)
    return [period_from_number(value) for value in range(end - 3, end + 1)]


def expected_periods(year: int) -> list[str]:
    return [f"{year}-Q{quarter}" for quarter in range(1, 5)]


def source_url(ticker: str, report_type: str, year: int) -> str:
    return (
        "https://cafef.vn/du-lieu/BaoCaoTaiChinh.aspx"
        f"?quarter=4&symbol={ticker}&type={report_type}&year={year}"
    )


def make_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def find_component(report: pd.DataFrame, component: str) -> pd.Series:
    definition = COMPONENT_DEFINITIONS[component]
    combined = pd.Series(dtype=float)
    for _, row in report.iterrows():
        name = str(row["item_folded"])
        exact = name in definition.get("exact", ())
        contains = any(pattern in name for pattern in definition.get("contains", ()))
        excluded = any(pattern in name for pattern in definition.get("excludes", ()))
        if (exact or contains) and not excluded:
            values = pd.to_numeric(
                row.drop(labels=["item", "item_folded"]), errors="coerce"
            )
            combined = combined.combine_first(values)
    return combined


def fetch_statement(
    session: requests.Session,
    ticker: str,
    report_type: str,
    year: int,
    timeout: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    url = source_url(ticker, report_type, year)
    started = time.perf_counter()
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    tables = pd.read_html(StringIO(response.text), flavor="lxml")
    candidates = [table for table in tables if table.shape[0] >= 10 and table.shape[1] >= 5]
    if not candidates:
        raise ValueError("financial statement table not found")

    table = max(candidates, key=len).iloc[:, :5].copy()
    periods = expected_periods(year)
    table.columns = ["item", *periods]
    table["item_folded"] = table["item"].map(fold_text)
    for period in periods:
        table[period] = table[period].map(parse_number)

    components = (
        ("total_assets", "total_liabilities", "equity", "current_assets",
         "current_liabilities", "inventory")
        if report_type == "BSheet"
        else ("net_revenue", "gross_profit", "parent_net_income", "net_income")
    )
    rows = []
    for period in periods:
        row: dict[str, object] = {
            "ticker": ticker,
            "period": period,
            "report_scope": "cafef_default",
        }
        for component in components:
            row[component] = find_component(table, component).get(period, np.nan)
        rows.append(row)

    log_row = {
        "ticker": ticker,
        "year": year,
        "report_type": report_type,
        "url": url,
        "http_status": response.status_code,
        "bytes": len(response.content),
        "sha256": digest,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "error": "",
    }
    return pd.DataFrame(rows), log_row


def download_ticker(
    ticker: str,
    timeout: float,
    sleep_seconds: float,
) -> tuple[str, pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    session = make_session()
    balance_frames: list[pd.DataFrame] = []
    income_frames: list[pd.DataFrame] = []
    logs: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for year in FETCH_YEARS:
        for report_type in ("BSheet", "IncSta"):
            try:
                frame, log_row = fetch_statement(
                    session, ticker, report_type, year, timeout
                )
                logs.append(log_row)
                if report_type == "BSheet":
                    balance_frames.append(frame)
                else:
                    income_frames.append(frame)
            except Exception as exc:  # Network/parser errors are recorded per page.
                failures.append(
                    {
                        "ticker": ticker,
                        "year": year,
                        "report_type": report_type,
                        "url": source_url(ticker, report_type, year),
                        "error": f"{type(exc).__name__}: {exc}",
                        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
            time.sleep(sleep_seconds)

    balance = pd.concat(balance_frames, ignore_index=True) if balance_frames else pd.DataFrame()
    income = pd.concat(income_frames, ignore_index=True) if income_frames else pd.DataFrame()
    if balance.empty and income.empty:
        combined = pd.DataFrame()
    elif balance.empty:
        combined = income
    elif income.empty:
        combined = balance
    else:
        combined = balance.merge(
            income,
            on=["ticker", "period", "report_scope"],
            how="outer",
        )
    if not combined.empty:
        combined = combined.sort_values("period").drop_duplicates(
            ["ticker", "period"], keep="last"
        )
    return ticker, combined, logs, failures


def load_universe(max_tickers: int | None = None) -> pd.DataFrame:
    profiles = pd.read_csv(PROFILES_FILE)
    profiles["symbol"] = profiles["symbol"].astype(str).str.upper().str.strip()
    profiles = profiles[
        profiles["exchange"].isin(["HOSE", "HNX"])
        & profiles["included_nonfinancial"].astype(str).str.lower().isin(["true", "1"])
    ].drop_duplicates("symbol")
    profiles = profiles.sort_values(["exchange", "symbol"]).reset_index(drop=True)
    if max_tickers is not None:
        profiles = profiles.head(max_tickers)
    return profiles


def write_progress(
    status_rows: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
) -> None:
    pd.DataFrame(status_rows).sort_values("ticker").to_csv(
        STATUS_FILE, index=False, encoding="utf-8-sig"
    )
    if failure_rows:
        pd.DataFrame(failure_rows).sort_values(
            ["ticker", "year", "report_type"]
        ).to_csv(FAILURES_FILE, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(
            columns=["ticker", "year", "report_type", "url", "error", "retrieved_at_utc"]
        ).to_csv(FAILURES_FILE, index=False, encoding="utf-8-sig")


def combine_cache(universe: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker in universe["symbol"]:
        path = CACHE_DIR / f"{ticker}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(["ticker", "period"], keep="last")
        combined = combined.sort_values(["ticker", "period"]).reset_index(drop=True)
    combined.to_csv(COMPONENTS_FILE, index=False, encoding="utf-8-sig")
    return combined


def download_all(
    workers: int,
    timeout: float,
    sleep_seconds: float,
    max_tickers: int | None,
    force: bool,
) -> pd.DataFrame:
    ensure_dirs()
    universe = load_universe(max_tickers)
    completed_status: set[str] = set()
    if STATUS_FILE.exists() and not force:
        previous_status = pd.read_csv(STATUS_FILE)
        complete_mask = (
            previous_status.get("complete_18_pages", pd.Series(dtype=object))
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )
        completed_status = set(
            previous_status.loc[complete_mask, "ticker"].astype(str)
        )
    cached = {
        ticker for ticker in universe["symbol"]
        if (
            (CACHE_DIR / f"{ticker}.csv").exists()
            and ticker in completed_status
            and not force
        )
    }
    pending = [ticker for ticker in universe["symbol"] if ticker not in cached]
    status_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    if STATUS_FILE.exists() and not force:
        status_rows = pd.read_csv(STATUS_FILE).to_dict("records")
    if FAILURES_FILE.exists() and not force:
        existing_failures = pd.read_csv(FAILURES_FILE)
        if not existing_failures.empty:
            failure_rows = existing_failures.to_dict("records")

    print(
        f"CafeF download: universe={len(universe)}, cached={len(cached)}, "
        f"pending={len(pending)}, workers={workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_ticker, ticker, timeout, sleep_seconds): ticker
            for ticker in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                _, frame, logs, failures = future.result()
                if not frame.empty:
                    frame.to_csv(
                        CACHE_DIR / f"{ticker}.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                status_rows = [
                    row for row in status_rows if str(row.get("ticker")) != ticker
                ]
                status_rows.append(
                    {
                        "ticker": ticker,
                        "pages_ok": len(logs),
                        "pages_failed": len(failures),
                        "component_rows": len(frame),
                        "complete_18_pages": len(logs) == 18,
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                failure_rows = [
                    row for row in failure_rows if str(row.get("ticker")) != ticker
                ]
                failure_rows.extend(failures)
            except Exception as exc:
                status_rows.append(
                    {
                        "ticker": ticker,
                        "pages_ok": 0,
                        "pages_failed": 18,
                        "component_rows": 0,
                        "complete_18_pages": False,
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                        "fatal_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            write_progress(status_rows, failure_rows)
            if index % 10 == 0 or index == len(pending):
                print(f"Downloaded {index}/{len(pending)} pending tickers", flush=True)

    combined = combine_cache(universe)
    print(f"Components saved: {COMPONENTS_FILE} ({len(combined):,} rows)", flush=True)
    return combined


def safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def derive_ticker_panel(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("period").set_index("period")
    rows = []
    for period in [
        period_from_number(value)
        for value in range(period_number(START_PERIOD), period_number(END_PERIOD) + 1)
    ]:
        if period not in frame.index:
            continue
        row = frame.loc[period]
        prior_period = period_from_number(period_number(period) - 4)
        assets_now = row.get("total_assets", np.nan)
        assets_prior = (
            frame.loc[prior_period].get("total_assets", np.nan)
            if prior_period in frame.index
            else np.nan
        )
        average_assets = (
            (assets_now + assets_prior) / 2
            if pd.notna(assets_now) and pd.notna(assets_prior)
            else np.nan
        )
        ttm = trailing_periods(period)
        ttm_frame = frame.reindex(ttm)
        income = ttm_frame["parent_net_income"].combine_first(
            ttm_frame["net_income"]
        )
        ttm_income = income.sum(min_count=4)
        ttm_revenue = ttm_frame["net_revenue"].sum(min_count=4)
        ttm_gross_profit = ttm_frame["gross_profit"].sum(min_count=4)
        current_assets = row.get("current_assets", np.nan)
        current_liabilities = row.get("current_liabilities", np.nan)
        inventory = row.get("inventory", np.nan)
        valid_assets = pd.notna(assets_now) and assets_now > 0
        valid_liabilities = (
            pd.notna(row.get("total_liabilities", np.nan))
            and row.get("total_liabilities", np.nan) >= 0
        )
        valid_liquidity_inputs = (
            pd.notna(current_assets)
            and current_assets >= 0
            and pd.notna(current_liabilities)
            and current_liabilities > 0
            and pd.notna(inventory)
            and inventory >= 0
        )
        valid_revenue = pd.notna(ttm_revenue) and ttm_revenue > 0

        rows.append(
            {
                "ticker": row["ticker"],
                "period": period,
                "roa": safe_ratio(ttm_income, average_assets),
                "debt_to_assets": safe_ratio(
                    row.get("total_liabilities", np.nan)
                    if valid_liabilities
                    else np.nan,
                    assets_now if valid_assets else np.nan,
                ),
                "current_ratio": safe_ratio(
                    current_assets if valid_liquidity_inputs else np.nan,
                    current_liabilities,
                ),
                "quick_ratio": safe_ratio(
                    current_assets - inventory
                    if valid_liquidity_inputs
                    else np.nan,
                    current_liabilities,
                ),
                "asset_turnover": safe_ratio(
                    ttm_revenue if valid_revenue else np.nan, average_assets
                ),
                "gross_margin": safe_ratio(
                    ttm_gross_profit, ttm_revenue if valid_revenue else np.nan
                ),
                "log_total_assets": (
                    float(np.log(assets_now))
                    if pd.notna(assets_now) and assets_now > 0
                    else np.nan
                ),
                "total_assets": assets_now,
                "ttm_parent_net_income": ttm_income,
                "ttm_net_revenue": ttm_revenue,
                "ttm_gross_profit": ttm_gross_profit,
                "report_scope": row.get("report_scope", "cafef_default"),
                "income_fallback_used": bool(
                    ttm_frame["parent_net_income"].isna().any()
                    and ttm_frame["net_income"].notna().any()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_panel(min_quarters: int, min_period_coverage: float) -> pd.DataFrame:
    ensure_dirs()
    components = pd.read_csv(COMPONENTS_FILE)
    numeric_cols = list(COMPONENT_DEFINITIONS)
    for col in numeric_cols:
        if col not in components:
            components[col] = np.nan
        components[col] = pd.to_numeric(components[col], errors="coerce")

    frames = [
        derive_ticker_panel(frame)
        for _, frame in components.groupby("ticker", sort=True)
    ]
    master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    profiles = load_universe().rename(columns={"symbol": "ticker"})
    master = master.merge(profiles, on="ticker", how="left")
    master = master.sort_values(["ticker", "period"]).reset_index(drop=True)
    master.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")

    complete = master.dropna(subset=MODEL_COLS).copy()
    counts = complete.groupby("ticker").size()
    analysis = complete[complete["ticker"].isin(counts[counts >= min_quarters].index)].copy()

    universe_firms = len(load_universe())
    coverage = (
        master.assign(complete_case=master[MODEL_COLS].notna().all(axis=1))
        .groupby("period")
        .agg(
            statement_firms=("ticker", "nunique"),
            complete_case_firms=("complete_case", "sum"),
        )
        .reset_index()
    )
    coverage["universe_firms"] = universe_firms
    coverage["statement_coverage"] = coverage["statement_firms"] / universe_firms
    coverage["complete_case_coverage"] = coverage["complete_case_firms"] / universe_firms
    coverage["excluded_low_coverage"] = (
        coverage["complete_case_coverage"] < min_period_coverage
    )
    coverage.to_csv(COVERAGE_FILE, index=False, encoding="utf-8-sig")

    excluded_periods = set(
        coverage.loc[coverage["excluded_low_coverage"], "period"].astype(str)
    )
    if excluded_periods:
        analysis = analysis[~analysis["period"].isin(excluded_periods)].copy()
        counts = analysis.groupby("ticker").size()
        analysis = analysis[
            analysis["ticker"].isin(counts[counts >= min_quarters].index)
        ].copy()
    analysis.to_csv(ANALYSIS_FILE, index=False, encoding="utf-8-sig")

    status = pd.read_csv(STATUS_FILE) if STATUS_FILE.exists() else pd.DataFrame()
    failures = pd.read_csv(FAILURES_FILE) if FAILURES_FILE.exists() else pd.DataFrame()
    quality_rows = [
        {"metric": "universe_firms", "value": universe_firms},
        {"metric": "downloaded_tickers", "value": components["ticker"].nunique()},
        {"metric": "download_pages_failed", "value": len(failures)},
        {
            "metric": "tickers_complete_18_pages",
            "value": int(
                status.get("complete_18_pages", pd.Series(dtype=bool))
                .astype(str).str.lower().isin(["true", "1"]).sum()
            ),
        },
        {"metric": "master_rows", "value": len(master)},
        {"metric": "master_firms", "value": master["ticker"].nunique()},
        {"metric": "complete_case_rows_before_rules", "value": len(complete)},
        {"metric": "analysis_rows", "value": len(analysis)},
        {"metric": "analysis_firms", "value": analysis["ticker"].nunique()},
        {"metric": "analysis_periods", "value": analysis["period"].nunique()},
        {"metric": "analysis_start", "value": analysis["period"].min()},
        {"metric": "analysis_end", "value": analysis["period"].max()},
        {
            "metric": "excluded_low_coverage_periods",
            "value": ",".join(sorted(excluded_periods)),
        },
        {
            "metric": "duplicate_ticker_period_rows",
            "value": int(master.duplicated(["ticker", "period"]).sum()),
        },
        {
            "metric": "income_fallback_rows",
            "value": int(master["income_fallback_used"].sum()),
        },
    ]
    for col in MODEL_COLS:
        quality_rows.append(
            {"metric": f"missing_master_{col}", "value": int(master[col].isna().sum())}
        )
    for exchange, count in (
        analysis.drop_duplicates("ticker")["exchange"].value_counts().items()
    ):
        quality_rows.append(
            {"metric": f"analysis_firms_{exchange}", "value": int(count)}
        )
    pd.DataFrame(quality_rows).to_csv(QUALITY_FILE, index=False, encoding="utf-8-sig")
    print(f"Master panel: {MASTER_FILE} ({master.shape})", flush=True)
    print(f"Analysis panel: {ANALYSIS_FILE} ({analysis.shape})", flush=True)
    return analysis


def demean_by_entity(
    frame: pd.DataFrame, cols: list[str], entity_col: str = "ticker"
) -> pd.DataFrame:
    output = frame.copy()
    output[cols] = frame[cols] - frame.groupby(entity_col)[cols].transform("mean")
    return output


def prepare_model_data(
    frame: pd.DataFrame,
    write_winsor: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    clean = frame.dropna(subset=[*MODEL_COLS, "ticker", "period"]).copy()
    winsor_rows = []
    for col in MODEL_COLS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        lower = float(clean[col].quantile(0.01))
        upper = float(clean[col].quantile(0.99))
        winsor_rows.append({"variable": col, "lower": lower, "upper": upper})
        clean[col] = clean[col].clip(lower=lower, upper=upper)
    if write_winsor:
        pd.DataFrame(winsor_rows).to_csv(
            WINSOR_FILE, index=False, encoding="utf-8-sig"
        )

    demeaned = demean_by_entity(clean, MODEL_COLS)
    time_dummies = pd.get_dummies(
        clean["period"], prefix="time", drop_first=True
    ).astype(float)
    time_dummies = time_dummies - time_dummies.groupby(clean["ticker"]).transform(
        "mean"
    )
    x_financial = StandardScaler().fit_transform(
        demeaned[X_COLS].to_numpy(dtype=float)
    )
    x_all = np.hstack([x_financial, time_dummies.to_numpy(dtype=float)])
    return (
        clean,
        x_all,
        demeaned[Y_COL].to_numpy(dtype=float),
        demeaned[D_COL].to_numpy(dtype=float),
    )


def append_result(
    rows: list[dict[str, object]],
    model_name: str,
    coefficient: float,
    standard_error: float,
    p_value: float,
    ci_lower: float,
    ci_upper: float,
    clean: pd.DataFrame,
) -> None:
    rows.append(
        {
            "Model": model_name,
            "Coef": coefficient,
            "SE": standard_error,
            "P-value": p_value,
            "CI_lower": ci_lower,
            "CI_upper": ci_upper,
            "N": len(clean),
            "Firms": clean["ticker"].nunique(),
            "Periods": clean["period"].nunique(),
            "SE_type": "firm-clustered",
        }
    )


def run_dml_model(
    clean: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    learner: object,
    n_folds: int,
    n_rep: int,
    seed: int,
) -> tuple[float, float, float, float, float]:
    import doubleml as dml

    cluster_codes = pd.factorize(clean["ticker"], sort=True)[0].reshape(-1, 1)
    dml_data = dml.DoubleMLData.from_arrays(
        x=x, y=y, d=d, cluster_vars=cluster_codes
    )
    np.random.seed(seed)
    model = dml.DoubleMLPLR(
        dml_data,
        ml_l=learner,
        ml_m=learner,
        n_folds=n_folds,
        n_rep=n_rep,
    )
    model.fit()
    confidence = model.confint()
    return (
        float(model.coef[0]),
        float(model.se[0]),
        float(model.pval[0]),
        float(confidence.iloc[0, 0]),
        float(confidence.iloc[0, 1]),
    )


def run_primary_models(n_folds: int, n_rep: int, seed: int) -> pd.DataFrame:
    import statsmodels.api as sm
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNetCV, LassoCV

    analysis = pd.read_csv(ANALYSIS_FILE)
    clean, x_all, y, d = prepare_model_data(analysis, write_winsor=True)
    rows: list[dict[str, object]] = []

    x_raw = sm.add_constant(clean[[D_COL, *X_COLS]].astype(float))
    pooled = sm.OLS(clean[Y_COL].astype(float), x_raw).fit(
        cov_type="cluster", cov_kwds={"groups": clean["ticker"]}
    )
    ci = pooled.conf_int().loc[D_COL]
    append_result(
        rows, "Pooled OLS (clustered by firm)", pooled.params[D_COL],
        pooled.bse[D_COL], pooled.pvalues[D_COL], ci.iloc[0], ci.iloc[1], clean
    )

    demeaned = demean_by_entity(clean, MODEL_COLS)
    x_fe = sm.add_constant(demeaned[[D_COL, *X_COLS]].astype(float))
    entity_fe = sm.OLS(demeaned[Y_COL].astype(float), x_fe).fit(
        cov_type="cluster", cov_kwds={"groups": clean["ticker"]}
    )
    ci = entity_fe.conf_int().loc[D_COL]
    append_result(
        rows, "Entity FE (clustered by firm)", entity_fe.params[D_COL],
        entity_fe.bse[D_COL], entity_fe.pvalues[D_COL], ci.iloc[0], ci.iloc[1], clean
    )

    time_dummies = pd.get_dummies(
        clean["period"], prefix="time", drop_first=True
    ).reset_index(drop=True).astype(float)
    ticker_groups = clean["ticker"].reset_index(drop=True)
    time_dummies = time_dummies - time_dummies.groupby(ticker_groups).transform(
        "mean"
    )
    x_twfe = pd.concat(
        [
            demeaned[[D_COL, *X_COLS]].reset_index(drop=True).astype(float),
            time_dummies,
        ],
        axis=1,
    )
    twfe = sm.OLS(
        demeaned[Y_COL].reset_index(drop=True).astype(float),
        sm.add_constant(x_twfe),
    ).fit(cov_type="cluster", cov_kwds={"groups": clean["ticker"].reset_index(drop=True)})
    ci = twfe.conf_int().loc[D_COL]
    append_result(
        rows, "Two-way FE (entity + time)", twfe.params[D_COL],
        twfe.bse[D_COL], twfe.pvalues[D_COL], ci.iloc[0], ci.iloc[1], clean
    )

    learners = [
        (
            "DML-PLR LassoCV (clustered by firm)",
            LassoCV(cv=5, max_iter=20000, random_state=seed),
        ),
        (
            "DML-PLR ElasticNetCV (clustered by firm)",
            ElasticNetCV(cv=5, max_iter=20000, random_state=seed),
        ),
        (
            "DML-PLR RandomForest (clustered by firm)",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ]
    for name, learner in learners:
        print(f"Running {name}", flush=True)
        values = run_dml_model(
            clean, x_all, y, d, learner, n_folds, n_rep, seed
        )
        append_result(rows, name, *values, clean)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")
    print(f"Primary results: {RESULTS_FILE}", flush=True)
    return results


def run_rf_specification(
    frame: pd.DataFrame,
    label: str,
    n_folds: int,
    n_rep: int,
    seed: int,
) -> dict[str, object]:
    from sklearn.ensemble import RandomForestRegressor

    clean, x_all, y, d = prepare_model_data(frame)
    learner = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    coef, se, pvalue, lo, hi = run_dml_model(
        clean, x_all, y, d, learner, n_folds, n_rep, seed
    )
    return {
        "specification": label,
        "theta": coef,
        "se": se,
        "pvalue": pvalue,
        "ci_lower": lo,
        "ci_upper": hi,
        "N": len(clean),
        "Firms": clean["ticker"].nunique(),
        "Periods": clean["period"].nunique(),
    }


def run_robustness(n_folds: int, n_rep: int, seed: int) -> None:
    analysis = pd.read_csv(ANALYSIS_FILE)
    rows = [
        run_rf_specification(
            analysis, "Primary sample", n_folds, n_rep, seed
        ),
        run_rf_specification(
            analysis[analysis["period"] != "2021-Q2"],
            "Exclude 2021-Q2",
            n_folds,
            n_rep,
            seed,
        ),
    ]
    counts = analysis.groupby("ticker").size()
    min_16 = analysis[analysis["ticker"].isin(counts[counts >= 16].index)]
    rows.append(
        run_rf_specification(
            min_16, "At least 16 quarters", n_folds, n_rep, seed
        )
    )
    pd.DataFrame(rows).to_csv(
        ROBUSTNESS_FILE, index=False, encoding="utf-8-sig"
    )

    seed_rows = []
    for rf_seed in (1, 2, 3, 4, 5):
        print(f"Running RF stability seed={rf_seed}", flush=True)
        seed_rows.append(
            run_rf_specification(
                analysis,
                f"RF seed {rf_seed}",
                n_folds,
                min(n_rep, 3),
                rf_seed,
            )
        )
    pd.DataFrame(seed_rows).to_csv(SEED_FILE, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CafeF statements and run the HOSE/HNX DML study."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["download", "build", "analysis", "robustness", "all"],
        default=["all"],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep", type=float, default=0.20)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-quarters", type=int, default=12)
    parser.add_argument("--min-period-coverage", type=float, default=0.50)
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
        steps = {"download", "build", "analysis", "robustness"}
    if "download" in steps:
        download_all(
            workers=args.workers,
            timeout=args.timeout,
            sleep_seconds=args.sleep,
            max_tickers=args.max_tickers,
            force=args.force,
        )
    if "build" in steps:
        build_panel(args.min_quarters, args.min_period_coverage)
    if "analysis" in steps:
        run_primary_models(args.folds, args.reps, args.seed)
    if "robustness" in steps:
        run_robustness(args.folds, args.reps, args.seed)


if __name__ == "__main__":
    main()
