"""Build the self-contained Vietnamese report for the CafeF replication study."""

from __future__ import annotations

import base64
import hashlib
import html
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "cafef_hose_hnx"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_FILE = ROOT / "reports" / "cafef_hose_hnx_dml_bao_cao_giang_vien.html"

RESULTS_FILE = RESULTS_DIR / "dml_cafef_hose_hnx_final_report.csv"
ROBUSTNESS_FILE = RESULTS_DIR / "cafef_robustness_checks.csv"
SEED_FILE = RESULTS_DIR / "cafef_rf_seed_stability.csv"
QUALITY_FILE = RESULTS_DIR / "cafef_data_quality_summary.csv"
COVERAGE_FILE = RESULTS_DIR / "cafef_period_coverage.csv"
VCI_RESULTS_FILE = ROOT / "results" / "expanded_hose_hnx" / "dml_hose_hnx_final_report.csv"
COMPARISON_FILE = RESULTS_DIR / "cafef_vci_model_comparison.csv"
MANIFEST_FILE = RESULTS_DIR / "cafef_artifact_manifest.csv"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "blue": {"xlight": "#EAF1FE", "base": "#A3BEFA", "dark": "#2E4780"},
    "gold": {"xlight": "#FFF4C2", "base": "#FFE15B", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "base": "#F0986E", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "base": "#A3D576", "dark": "#386411"},
}
NEUTRAL = {"light": "#E2E5EA", "base": "#C5CAD3", "dark": "#464C55"}

MODEL_LABELS = {
    "Pooled OLS (clustered by firm)": "Pooled OLS",
    "Entity FE (clustered by firm)": "Entity FE",
    "Two-way FE (entity + time)": "Two-way FE",
    "DML-PLR LassoCV (clustered by firm)": "DML Lasso",
    "DML-PLR ElasticNetCV (clustered by firm)": "DML ElasticNet",
    "DML-PLR RandomForest (clustered by firm)": "DML Random Forest",
}
VCI_MODEL_MAP = {
    "DML-PLR LassoCV (group folds)": "DML Lasso",
    "DML-PLR ElasticNetCV (group folds)": "DML ElasticNet",
    "DML-PLR RandomForest (group folds)": "DML Random Forest",
}


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"
            ],
            "font.monospace": [
                "SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"
            ],
        },
    )


def add_chart_header(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    subtitle: str,
    title_width: int = 78,
    subtitle_width: int = 112,
) -> None:
    title = textwrap.fill(title, title_width, break_long_words=False)
    subtitle = textwrap.fill(subtitle, subtitle_width, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    ax.set_title("")
    fig.subplots_adjust(
        top=max(0.62, 0.86 - 0.045 * (title_lines - 1) - 0.032 * (subtitle_lines - 1))
    )
    left = ax.get_position().x0
    fig.text(
        left, 0.985, title, ha="left", va="top", fontsize=13,
        fontweight="semibold", color=TOKENS["ink"], linespacing=1.08
    )
    fig.text(
        left, 0.93 - 0.045 * (title_lines - 1), subtitle,
        ha="left", va="top", fontsize=9, color=TOKENS["muted"], linespacing=1.18
    )
    sns.despine(ax=ax)


def save_figure(fig: plt.Figure, stem: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{stem}.png"
    svg = FIGURES_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor=TOKENS["surface"])
    fig.savefig(svg, bbox_inches="tight", facecolor=TOKENS["surface"])
    svg_text = svg.read_text(encoding="utf-8")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return png


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def write_artifact_manifest(paths: list[Path]) -> None:
    rows = []
    for path in paths:
        content = path.read_bytes()
        row_count = ""
        if path.suffix.lower() == ".csv":
            row_count = len(pd.read_csv(path))
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": len(content),
                "rows": row_count,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_csv(MANIFEST_FILE, index=False, encoding="utf-8-sig")


def quality_map(quality: pd.DataFrame) -> dict[str, str]:
    return dict(zip(quality["metric"].astype(str), quality["value"].astype(str)))


def build_comparison(cafef: pd.DataFrame, vci: pd.DataFrame) -> pd.DataFrame:
    cafef_part = cafef[cafef["Model"].str.startswith("DML-PLR")].copy()
    cafef_part["model_family"] = cafef_part["Model"].map(MODEL_LABELS)
    cafef_part["source"] = "CafeF BCTC"
    vci_part = vci[vci["Model"].isin(VCI_MODEL_MAP)].copy()
    vci_part["model_family"] = vci_part["Model"].map(VCI_MODEL_MAP)
    vci_part["source"] = "VCI ratios (previous)"
    columns = [
        "model_family", "source", "Coef", "SE", "P-value",
        "CI_lower", "CI_upper", "N", "Firms"
    ]
    comparison = pd.concat(
        [cafef_part[columns], vci_part[columns]], ignore_index=True
    )
    comparison.to_csv(COMPARISON_FILE, index=False, encoding="utf-8-sig")
    return comparison


def make_forest_chart(results: pd.DataFrame) -> Path:
    plot = results.copy()
    plot["label"] = plot["Model"].map(MODEL_LABELS)
    plot = plot.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    for index, row in plot.iterrows():
        focal = row["label"] == "DML Random Forest"
        fill = COLORS["gold"]["base"] if focal else TOKENS["panel"]
        edge = COLORS["gold"]["dark"] if focal else NEUTRAL["dark"]
        ax.errorbar(
            row["Coef"], index,
            xerr=np.array([[row["Coef"] - row["CI_lower"]],
                           [row["CI_upper"] - row["Coef"]]]),
            fmt="o", color=edge, markerfacecolor=fill, markeredgecolor=edge,
            linewidth=1.0, capsize=3, markersize=7 if focal else 6
        )
        ax.text(
            row["CI_upper"] + 0.0035, index, f"{row['Coef']:.4f}",
            va="center", fontsize=8.5, color=TOKENS["ink"], fontfamily="monospace"
        )
    ax.set_yticks(range(len(plot)), plot["label"])
    ax.set_xlabel("Hệ số của Debt / Assets")
    ax.set_ylabel("")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(plot["CI_lower"].min() - 0.012, 0.012)
    add_chart_header(
        fig, ax,
        "Tất cả mô hình đều ước lượng tác động âm",
        "Điểm là theta; thanh ngang là khoảng tin cậy 95%. N = 14.255; 578 doanh nghiệp; sai số chuẩn phân cụm theo doanh nghiệp.",
    )
    return save_figure(fig, "cafef_model_forest")


def make_coverage_chart(coverage: pd.DataFrame) -> Path:
    plot = coverage.copy()
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(11.4, 5.7))
    ax.plot(
        x, plot["complete_case_coverage"], color=COLORS["blue"]["base"],
        marker="o", markersize=3.8, linewidth=1.0
    )
    excluded = plot["excluded_low_coverage"].astype(bool)
    ax.scatter(
        x[excluded], plot.loc[excluded, "complete_case_coverage"],
        color=COLORS["orange"]["base"], edgecolor=COLORS["orange"]["dark"],
        linewidth=0.8, s=38, zorder=4, label="Bị loại (<50%)"
    )
    ax.axhline(0.5, color=NEUTRAL["dark"], linestyle=":", linewidth=1.0)
    ax.text(
        len(plot) - 0.4, 0.515, "Ngưỡng 50%", ha="right", va="bottom",
        fontsize=8.5, color=NEUTRAL["dark"]
    )
    for period in ("2021-Q1", "2021-Q2", "2024-Q2"):
        row = plot[plot["period"] == period]
        if row.empty:
            continue
        i = int(row.index[0])
        value = float(row["complete_case_coverage"].iloc[0])
        offset = 0.045 if period != "2021-Q2" else -0.07
        ax.annotate(
            f"{period}\n{value:.1%}",
            xy=(i, value), xytext=(i, value + offset), ha="center",
            fontsize=8, color=TOKENS["ink"],
            arrowprops={"arrowstyle": "-", "color": NEUTRAL["base"], "lw": 0.8},
        )
    tick_positions = np.arange(0, len(plot), 4)
    ax.set_xticks(tick_positions, plot.loc[tick_positions, "period"])
    ax.set_ylim(0.35, 0.96)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Quý")
    ax.set_ylabel("Tỷ lệ đủ toàn bộ biến mô hình")
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False,
        borderaxespad=0
    )
    add_chart_header(
        fig, ax,
        "CafeF lấp được 2021-Q2 nhưng thiếu dữ liệu đầu vào tại 2024-Q2",
        "Coverage trên 635 mã hiện niêm yết HOSE/HNX phi tài chính. Thiếu 2024-Q2 lan sang bốn quý kế tiếp vì ROA và các biến hoạt động dùng cửa sổ TTM.",
    )
    return save_figure(fig, "cafef_period_coverage")


def make_robustness_chart(robustness: pd.DataFrame, seeds: pd.DataFrame) -> Path:
    plot = robustness.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    for index, row in plot.iterrows():
        ax.errorbar(
            row["theta"], index,
            xerr=np.array([[row["theta"] - row["ci_lower"]],
                           [row["ci_upper"] - row["theta"]]]),
            fmt="o", color=COLORS["olive"]["dark"],
            markerfacecolor=COLORS["olive"]["base"],
            markeredgecolor=COLORS["olive"]["dark"],
            linewidth=1.0, capsize=3
        )
        ax.text(
            row["ci_upper"] + 0.002, index, f"{row['theta']:.4f}",
            va="center", fontsize=8.5, fontfamily="monospace"
        )
    ax.set_yticks(range(len(plot)), plot["specification"])
    ax.set_xlabel("Theta DML Random Forest")
    ax.set_ylabel("")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(plot["ci_lower"].min() - 0.008, -0.075), 0.006)
    add_chart_header(
        fig, ax,
        "Kết luận không phụ thuộc vào 2021-Q2 hay quy tắc tối thiểu 16 quý",
        f"Khoảng tin cậy 95%. Năm seed Random Forest cho theta từ {seeds['theta'].min():.4f} đến {seeds['theta'].max():.4f}.",
    )
    return save_figure(fig, "cafef_rf_robustness")


def make_comparison_chart(comparison: pd.DataFrame) -> Path:
    order = ["DML Lasso", "DML ElasticNet", "DML Random Forest"]
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    for index, family in enumerate(order):
        part = comparison[comparison["model_family"] == family].set_index("source")
        if len(part) != 2:
            continue
        vci = part.loc["VCI ratios (previous)", "Coef"]
        cafef = part.loc["CafeF BCTC", "Coef"]
        ax.hlines(index, vci, cafef, color=NEUTRAL["base"], linewidth=1.0)
        ax.scatter(
            vci, index, facecolor=TOKENS["panel"], edgecolor=NEUTRAL["dark"],
            linewidth=1.0, s=55, label="VCI trước đây" if index == 0 else None
        )
        ax.scatter(
            cafef, index, facecolor=COLORS["blue"]["base"],
            edgecolor=COLORS["blue"]["dark"], linewidth=1.0, s=55,
            label="CafeF BCTC" if index == 0 else None
        )
    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("Theta")
    ax.set_ylabel("")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(
        loc="upper right", frameon=False, ncol=2, borderaxespad=0.8
    )
    add_chart_header(
        fig, ax,
        "Nguồn CafeF tái lập cùng dấu với kết quả VCI trước đây",
        "So sánh định hướng, không phải kiểm định chênh lệch nguồn thuần túy: CafeF dùng log tổng tài sản, 2018-Q1–2025-Q4; VCI dùng log vốn hóa và kết thúc 2026-Q1.",
    )
    return save_figure(fig, "cafef_vci_dml_comparison")


def p_text(value: float) -> str:
    if value < 0.001:
        return "&lt;0,001"
    return f"{value:.3f}".replace(".", ",")


def results_table(results: pd.DataFrame) -> str:
    rows = []
    for _, row in results.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(MODEL_LABELS[row['Model']])}</td>"
            f"<td class='num'>{row['Coef']:.4f}</td>"
            f"<td class='num'>{row['SE']:.4f}</td>"
            f"<td class='num'>[{row['CI_lower']:.4f}; {row['CI_upper']:.4f}]</td>"
            f"<td class='num'>{p_text(row['P-value'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def robustness_table(robustness: pd.DataFrame) -> str:
    rows = []
    for _, row in robustness.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['specification']))}</td>"
            f"<td class='num'>{row['theta']:.4f}</td>"
            f"<td class='num'>[{row['ci_lower']:.4f}; {row['ci_upper']:.4f}]</td>"
            f"<td class='num'>{int(row['N']):,}</td>"
            f"<td class='num'>{int(row['Firms'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def build_report() -> None:
    use_chart_theme()
    results = pd.read_csv(RESULTS_FILE)
    robustness = pd.read_csv(ROBUSTNESS_FILE)
    seeds = pd.read_csv(SEED_FILE)
    quality = pd.read_csv(QUALITY_FILE)
    coverage = pd.read_csv(COVERAGE_FILE)
    vci_results = pd.read_csv(VCI_RESULTS_FILE)
    quality_values = quality_map(quality)
    comparison = build_comparison(results, vci_results)

    forest_uri = data_uri(make_forest_chart(results))
    coverage_uri = data_uri(make_coverage_chart(coverage))
    robustness_uri = data_uri(make_robustness_chart(robustness, seeds))
    comparison_uri = data_uri(make_comparison_chart(comparison))

    rf = results[results["Model"].str.contains("RandomForest")].iloc[0]
    q1_2021 = coverage.loc[coverage["period"] == "2021-Q1"].iloc[0]
    q2_2021 = coverage.loc[coverage["period"] == "2021-Q2"].iloc[0]
    q2_2024 = coverage.loc[coverage["period"] == "2024-Q2"].iloc[0]
    excluded = coverage[coverage["excluded_low_coverage"].astype(bool)]
    excluded_rows = "".join(
        "<tr>"
        f"<td>{row['period']}</td>"
        f"<td class='num'>{int(row['complete_case_firms'])}</td>"
        f"<td class='num'>{row['complete_case_coverage']:.1%}</td>"
        "</tr>"
        for _, row in excluded.iterrows()
    )
    effect_10pp = rf["Coef"] * 0.10 * 100
    generated = datetime.now().strftime("%d/%m/%Y %H:%M")

    document = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nghiên cứu cấu trúc vốn bằng DML - dữ liệu CafeF</title>
<style>
:root {{
  --surface:#FCFCFD; --panel:#FFFFFF; --ink:#1F2430; --muted:#6F768A;
  --grid:#E6E8F0; --axis:#D7DBE7; --blue:#2E4780; --blue-light:#EAF1FE;
  --gold:#736422; --gold-light:#FFF4C2; --orange:#804126; --orange-light:#FFEDDE;
  --olive:#386411; --olive-light:#D8ECBD;
}}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{
  margin:0; background:var(--surface); color:var(--ink);
  font-family:Aptos,Inter,"Segoe UI",Arial,sans-serif; line-height:1.58;
}}
a {{ color:var(--blue); }}
.page {{ max-width:1120px; margin:0 auto; padding:36px 34px 72px; }}
.masthead {{ border-top:5px solid var(--blue); padding-top:26px; }}
.eyebrow {{ color:var(--blue); font-size:13px; font-weight:700; text-transform:uppercase; }}
h1 {{ font-size:38px; line-height:1.15; margin:8px 0 12px; letter-spacing:0; max-width:900px; }}
.lead {{ font-size:18px; max-width:900px; color:#464C55; margin:0; }}
.meta {{ color:var(--muted); font-size:13px; margin-top:16px; }}
.toc {{
  display:flex; flex-wrap:wrap; gap:18px; padding:16px 0;
  margin:28px 0 0; border-top:1px solid var(--axis); border-bottom:1px solid var(--axis);
}}
.toc a {{ text-decoration:none; font-size:14px; font-weight:600; }}
.kpis {{
  display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
  border-bottom:1px solid var(--axis); margin:0 0 38px;
}}
.kpi {{ padding:22px 18px 22px 0; }}
.kpi + .kpi {{ border-left:1px solid var(--axis); padding-left:18px; }}
.kpi strong {{ display:block; font-size:27px; line-height:1.05; font-variant-numeric:tabular-nums; }}
.kpi span {{ color:var(--muted); font-size:13px; }}
section {{ padding:34px 0; border-bottom:1px solid var(--axis); }}
h2 {{ font-size:26px; margin:0 0 12px; letter-spacing:0; }}
h3 {{ font-size:18px; margin:26px 0 8px; letter-spacing:0; }}
p {{ max-width:900px; }}
.answer {{
  border-left:4px solid var(--gold); background:var(--gold-light);
  padding:18px 20px; margin:18px 0 0; max-width:940px;
}}
.warning {{
  border-left:4px solid var(--orange); background:var(--orange-light);
  padding:16px 20px; margin:20px 0; max-width:940px;
}}
.positive {{
  border-left:4px solid var(--olive); background:var(--olive-light);
  padding:16px 20px; margin:20px 0; max-width:940px;
}}
.figure {{ margin:24px 0 12px; }}
.figure img {{ display:block; width:100%; height:auto; }}
.caption {{ color:var(--muted); font-size:13px; margin-top:8px; }}
.table-wrap {{ overflow-x:auto; margin:18px 0; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); font-size:14px; }}
th {{ text-align:left; color:#464C55; background:#F4F5F7; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--grid); vertical-align:top; }}
.num {{ text-align:right; font-family:Consolas,"SF Mono",monospace; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.method-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:0 36px; }}
code {{ font-family:Consolas,"SF Mono",monospace; background:#F4F5F7; padding:2px 5px; }}
ul {{ padding-left:20px; max-width:920px; }}
li {{ margin:7px 0; }}
.source-list {{ font-size:14px; color:#464C55; }}
footer {{ padding-top:26px; color:var(--muted); font-size:12px; }}
@media (max-width:760px) {{
  .page {{ padding:24px 18px 52px; }}
  h1 {{ font-size:30px; }}
  .kpis {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  .kpi:nth-child(3) {{ border-left:0; }}
  .method-grid {{ grid-template-columns:1fr; }}
}}
@media print {{
  body {{ background:white; }}
  .page {{ max-width:none; padding:0; }}
  .toc {{ display:none; }}
  section,.figure,table {{ break-inside:avoid; }}
  a {{ color:inherit; text-decoration:none; }}
}}
</style>
</head>
<body>
<main class="page">
  <header class="masthead" data-contract-section="title">
    <div class="eyebrow">Báo cáo nghiên cứu · BCTC CafeF · HOSE &amp; HNX</div>
    <h1>Nghiên cứu tác động của cấu trúc vốn lên hiệu quả doanh nghiệp bằng DML</h1>
    <p class="lead">Bản chạy lại độc lập bằng dữ liệu báo cáo tài chính CafeF cho giai đoạn 2018-Q1 đến 2025-Q4.</p>
    <p class="meta">Tạo lúc {generated} · Biến kết quả: ROA TTM · Biến can thiệp: Nợ phải trả / Tổng tài sản</p>
  </header>

  <nav class="toc" aria-label="Mục lục">
    <a href="#ket-luan">Kết luận</a><a href="#du-lieu">Dữ liệu</a>
    <a href="#phuong-phap">Phương pháp</a><a href="#ket-qua">Kết quả</a>
    <a href="#do-ben">Độ bền</a><a href="#gioi-han">Giới hạn</a>
  </nav>

  <div class="kpis" aria-label="Chỉ số chính">
    <div class="kpi"><strong>{int(rf['Firms'])}</strong><span>doanh nghiệp trong mô hình</span></div>
    <div class="kpi"><strong>{int(rf['N']):,}</strong><span>quan sát doanh nghiệp-quý</span></div>
    <div class="kpi"><strong>{rf['Coef']:.4f}</strong><span>theta DML Random Forest</span></div>
    <div class="kpi"><strong>{effect_10pp:.2f} đ.%</strong><span>ROA khi Debt/Assets tăng 10 đ.%</span></div>
  </div>

  <section id="ket-luan" data-contract-section="summary">
    <h2>Kết luận trả lời trực tiếp</h2>
    <div class="answer">
      <strong>Dấu kỳ vọng âm được xác nhận.</strong> DML Random Forest ước lượng theta
      <strong>{rf['Coef']:.4f}</strong>, KTC 95% [{rf['CI_lower']:.4f}; {rf['CI_upper']:.4f}],
      p {p_text(rf['P-value'])}. Theo thang đo tuyến tính của mô hình, Debt/Assets tăng 10 điểm
      phần trăm gắn với ROA giảm khoảng <strong>{abs(effect_10pp):.2f} điểm phần trăm</strong>.
    </div>
    <p>Cả Pooled OLS, Entity FE, Two-way FE và ba biến thể DML đều cho hệ số âm có ý nghĩa thống kê. Kết luận cũng giữ nguyên khi bỏ 2021-Q2, yêu cầu tối thiểu 16 quý và thay năm seed Random Forest.</p>
    <div class="warning"><strong>Cách diễn đạt phù hợp:</strong> đây là bằng chứng nhân quả có điều kiện theo giả định DML, chưa phải quan hệ nhân quả tuyệt đối. DML xử lý phi tuyến và regularization bias nhưng không tự tạo ra biến thiên ngoại sinh hay loại bỏ mọi biến bị bỏ sót.</div>
  </section>

  <section id="du-lieu" data-contract-section="data-quality">
    <h2>Dữ liệu và kiểm soát chất lượng</h2>
    <p>Pipeline tải 18 trang cho mỗi mã: Cân đối kế toán và Kết quả kinh doanh, mỗi năm 2017–2025. Năm 2017 chỉ dùng để tính tài sản bình quân và TTM cho 2018. Toàn bộ 635 mã hoàn tất 18/18 trang, không còn lỗi HTTP hoặc parser sau lượt tải bù.</p>
    <div class="method-grid">
      <div><h3>Phạm vi</h3><p>635 doanh nghiệp phi tài chính đang thuộc HOSE/HNX trong file universe hiện tại; 578 doanh nghiệp đáp ứng complete-case và tối thiểu 12 quý.</p></div>
      <div><h3>Nguồn biến</h3><p>Mọi biến tài chính lấy từ BCTC CafeF. File universe/exchange/sector được tái sử dụng từ danh sách niêm yết của pipeline trước; không dùng tỷ số VCI để ước lượng CafeF.</p></div>
    </div>
    <div class="figure"><img src="{coverage_uri}" alt="Biểu đồ coverage theo quý"><p class="caption">Hình 1. Coverage complete-case theo quý; điểm màu cam là quý bị loại theo ngưỡng 50% định trước.</p></div>
    <div class="positive"><strong>Riêng 2021-Q2 không còn là vấn đề:</strong> Q1 có {int(q1_2021['complete_case_firms'])} doanh nghiệp ({q1_2021['complete_case_coverage']:.1%}); Q2 có {int(q2_2021['complete_case_firms'])} doanh nghiệp ({q2_2021['complete_case_coverage']:.1%}).</div>
    <div class="warning"><strong>Lỗ hổng mới của CafeF:</strong> 2024-Q2 chỉ đạt {int(q2_2024['complete_case_firms'])}/{int(q2_2024['universe_firms'])} ({q2_2024['complete_case_coverage']:.1%}). Do biến TTM cần bốn quý liên tiếp, coverage thấp kéo dài đến 2025-Q2.</div>
    <h3>Các quý bị loại</h3>
    <div class="table-wrap"><table><thead><tr><th>Quý</th><th class="num">Doanh nghiệp đủ biến</th><th class="num">Coverage</th></tr></thead><tbody>{excluded_rows}</tbody></table></div>
  </section>

  <section id="phuong-phap" data-contract-section="methods">
    <h2>Phương pháp và định nghĩa biến</h2>
    <div class="method-grid">
      <div><h3>Biến chính</h3><ul>
        <li><code>ROA</code> = LNST cổ đông công ty mẹ TTM / Tổng tài sản bình quân.</li>
        <li><code>Debt/Assets</code> = Tổng nợ phải trả / Tổng tài sản.</li>
        <li>Nếu CafeF không có LNST công ty mẹ, dùng LNST sau thuế làm fallback và lưu cờ truy vết.</li>
      </ul></div>
      <div><h3>Biến kiểm soát</h3><ul>
        <li>Current ratio, quick ratio, asset turnover và gross margin.</li>
        <li><code>log_total_assets</code> thay cho vốn hóa để không trộn nguồn giá thị trường.</li>
        <li>Winsorize 1% hai đuôi chỉ trên mẫu mô hình; master panel giữ nguyên.</li>
      </ul></div>
      <div><h3>Ước lượng</h3><ul>
        <li>Pooled OLS, Entity FE, Two-way FE.</li>
        <li>DML-PLR với LassoCV, ElasticNetCV và Random Forest.</li>
        <li>5 folds × 5 repetitions; cluster theo doanh nghiệp ở cross-fitting và sai số chuẩn.</li>
      </ul></div>
      <div><h3>Quy tắc mẫu</h3><ul>
        <li>Tối thiểu 12 quý complete-case mỗi doanh nghiệp.</li>
        <li>Loại quý có coverage dưới 50% của 635 mã universe.</li>
        <li>Mẫu cuối: {quality_values['analysis_rows']} hàng, {quality_values['analysis_firms']} doanh nghiệp, {quality_values['analysis_periods']} quý.</li>
      </ul></div>
    </div>
  </section>

  <section id="ket-qua" data-contract-section="results">
    <h2>Kết quả mô hình</h2>
    <div class="figure"><img src="{forest_uri}" alt="Forest plot của sáu mô hình"><p class="caption">Hình 2. Hệ số Debt/Assets và khoảng tin cậy 95%.</p></div>
    <div class="table-wrap"><table>
      <thead><tr><th>Mô hình</th><th class="num">Theta</th><th class="num">SE</th><th class="num">KTC 95%</th><th class="num">p-value</th></tr></thead>
      <tbody>{results_table(results)}</tbody>
    </table></div>
    <h3>Đối chiếu với lần chạy VCI trước</h3>
    <div class="figure"><img src="{comparison_uri}" alt="So sánh theta DML CafeF và VCI"><p class="caption">Hình 3. Đối chiếu dấu và độ lớn; không diễn giải chênh lệch là do nguồn dữ liệu duy nhất.</p></div>
    <p>CafeF tái lập cùng dấu âm trên cả ba learner DML. Tuy nhiên hai lần chạy khác cả nguồn dữ liệu, biến quy mô và điểm kết thúc thời gian, nên đây là kiểm tra tính nhất quán định hướng, không phải thí nghiệm A/B giữa CafeF và VCI.</p>
  </section>

  <section id="do-ben" data-contract-section="robustness">
    <h2>Kiểm tra độ bền</h2>
    <div class="figure"><img src="{robustness_uri}" alt="Kiểm tra độ bền của DML Random Forest"><p class="caption">Hình 4. Ba đặc tả Random Forest và khoảng tin cậy 95%.</p></div>
    <div class="table-wrap"><table>
      <thead><tr><th>Đặc tả</th><th class="num">Theta</th><th class="num">KTC 95%</th><th class="num">N</th><th class="num">Doanh nghiệp</th></tr></thead>
      <tbody>{robustness_table(robustness)}</tbody>
    </table></div>
    <p>Năm seed Random Forest cho theta trong [{seeds['theta'].min():.4f}; {seeds['theta'].max():.4f}], tất cả khoảng tin cậy đều nằm dưới 0. Bỏ 2021-Q2 gần như không làm thay đổi ước lượng.</p>
  </section>

  <section id="gioi-han" data-contract-section="limitations">
    <h2>Giới hạn và việc cần làm tiếp</h2>
    <ul>
      <li><strong>Survivorship bias:</strong> universe là các mã hiện còn trong danh sách, chưa bổ sung toàn bộ doanh nghiệp đã hủy niêm yết trong 2018–2025.</li>
      <li><strong>Phạm vi báo cáo:</strong> endpoint CafeF cũ không gắn nhãn nhất quán “hợp nhất” hay “riêng lẻ”; pipeline lưu phạm vi là <code>cafef_default</code>.</li>
      <li><strong>Chất lượng nguồn:</strong> CafeF tự nêu dữ liệu mang tính tham khảo. Cần đối chiếu mẫu ngẫu nhiên với PDF BCTC công bố chính thức trước khi nộp nghiên cứu cuối.</li>
      <li><strong>Khoảng trống 2024-Q2:</strong> không nội suy. Năm quý bị ảnh hưởng TTM được loại minh bạch.</li>
      <li><strong>Nhận diện nhân quả:</strong> vẫn có thể còn reverse causality và confounder không quan sát; nên cân nhắc lagged treatment, placebo lead và đặc tả động.</li>
    </ul>
    <h3>Khuyến nghị trước khi nộp luận văn</h3>
    <ol>
      <li>Xây historical universe gồm mã hủy niêm yết và ngày IPO thực tế.</li>
      <li>Đối chiếu tối thiểu 30–50 doanh nghiệp-quý với PDF từ HOSE/HNX hoặc trang IR doanh nghiệp.</li>
      <li>Chạy thêm Debt/Assets trễ một quý và placebo lead để kiểm tra chiều nhân quả.</li>
    </ol>
  </section>

  <section data-contract-section="sources">
    <h2>Nguồn và khả năng tái lập</h2>
    <ul class="source-list">
      <li><a href="https://cafef.vn/du-lieu/baocaotaichinh.aspx">CafeF – Báo cáo tài chính</a>: nguồn các khoản mục BCTC; CafeF ghi rõ dữ liệu có giá trị tham khảo.</li>
      <li><code>data/raw/cafef_statement_components_2017_2025.csv</code>: snapshot thành phần BCTC đã phân tích.</li>
      <li><code>data/processed/analysis_panel_cafef_hose_hnx_dataset.csv</code>: mẫu mô hình.</li>
      <li><code>results/cafef_hose_hnx/</code>: kết quả, quality checks, coverage, robustness và biểu đồ.</li>
      <li><code>results/cafef_hose_hnx/cafef_artifact_manifest.csv</code>: SHA-256, kích thước và số hàng của các artifact chính.</li>
      <li><code>scripts/cafef_hose_hnx_pipeline.py</code>: tải dữ liệu, dựng panel và ước lượng.</li>
    </ul>
  </section>
  <footer>Báo cáo kỹ thuật phục vụ trao đổi học thuật. Không phải khuyến nghị đầu tư.</footer>
</main>
</body>
</html>
"""
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(document, encoding="utf-8")
    write_artifact_manifest(
        [
            ROOT / "data" / "raw" / "cafef_statement_components_2017_2025.csv",
            ROOT / "data" / "raw" / "cafef_download_status.csv",
            ROOT / "data" / "raw" / "cafef_download_failures.csv",
            ROOT / "data" / "processed" / "master_panel_cafef_hose_hnx_dataset.csv",
            ROOT / "data" / "processed" / "analysis_panel_cafef_hose_hnx_dataset.csv",
            RESULTS_FILE,
            ROBUSTNESS_FILE,
            SEED_FILE,
            QUALITY_FILE,
            COVERAGE_FILE,
            COMPARISON_FILE,
            REPORT_FILE,
        ]
    )
    print(f"Report saved: {REPORT_FILE}")


if __name__ == "__main__":
    build_report()
