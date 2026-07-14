"""
Script tổng hợp: Chạy toàn bộ pipeline nghiên cứu cấu trúc vốn (1.300 doanh nghiệp)
1. Chạy bước làm sạch dữ liệu (clean_and_validate.py)
2. Chạy bước hồi quy Baseline (run_baseline_expanded.py)
3. Chạy bước hồi quy DML + Seed stability (run_dml_expanded.py)
4. Vẽ biểu đồ Scatter + LOWESS cho 1.300 doanh nghiệp
"""
import sys, os, warnings, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# Them thu muc scripts vao sys.path de import duoc cac script con
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm

def run_lowess_plot():
    print("\n" + "=" * 60)
    print("VẼ BIỂU ĐỒ SCATTER + LOWESS (1.300 DOANH NGHIỆP)")
    print("=" * 60)
    
    df = pd.read_csv("D:/draft 2/data/master_panel_all_firms.csv")
    y_col = 'roa'
    d_col = 'td_a'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Bằng chứng trực quan về tính phi tuyến (1.300 DN)\n(Scatter Plot + LOWESS Smoothing)', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plot_vars = [
        (d_col, 'Đòn bẩy tài sản tổng thể (TD/A)', axes[0, 0]),
        ('current_ratio', 'Khả năng thanh toán hiện thời', axes[0, 1]),
        ('gross_margin', 'Biên lợi nhuận gộp', axes[1, 0]),
        ('asset_turnover', 'Vòng quay tài sản', axes[1, 1]),
    ]
    
    for var, label, ax in plot_vars:
        df_sub = df[[var, y_col]].dropna()
        x_data = df_sub[var].values
        y_data = df_sub[y_col].values
        
        # Remove extreme outliers for visualization (keep 1st-99th percentile)
        mask = (x_data >= np.percentile(x_data, 1)) & (x_data <= np.percentile(x_data, 99))
        x_plot, y_plot = x_data[mask], y_data[mask]
        
        ax.scatter(x_plot, y_plot, alpha=0.10, s=5, color='steelblue', label='Quan sát')
        
        # LOWESS smoothing
        lowess_result = lowess(y_plot, x_plot, frac=0.4)
        ax.plot(lowess_result[:, 0], lowess_result[:, 1], color='red', linewidth=2.5, label='LOWESS')
        
        # OLS linear fit for comparison
        coefs = np.polyfit(x_plot, y_plot, 1)
        x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
        ax.plot(x_line, np.polyval(coefs, x_line), color='orange', linewidth=1.5, 
                linestyle='--', label='OLS tuyến tính')
        
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('ROA', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = 'D:/draft 2/results/figures/scatter_lowess_nonlinearity.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {out_path}")

def main():
    t0 = time.time()
    print("=" * 70)
    print("  BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE PHÂN TÍCH (1.300 DOANH NGHIỆP)")
    print("=" * 70)
    
    # 1. Chạy Clean & Validate
    print("\n[STEP 1/4] Chạy scripts/clean_and_validate.py...")
    import clean_and_validate
    clean_and_validate.main()
    
    # 2. Chạy Baseline Models
    print("\n[STEP 2/4] Chạy scripts/run_baseline_expanded.py...")
    import run_baseline_expanded
    run_baseline_expanded.main()
    
    # 3. Chạy LOWESS Plot
    print("\n[STEP 3/4] Chạy vẽ biểu đồ phi tuyến...")
    run_lowess_plot()
    
    # 4. Chạy DML
    print("\n[STEP 4/4] Chạy scripts/run_dml_expanded.py (Mất khoảng vài phút)...")
    import run_dml_expanded
    run_dml_expanded.main()
    
    # 5. Tự động cập nhật báo cáo
    print("\n[STEP 5/5] Tự động cập nhật kết quả mới vào báo cáo...")
    import update_results_in_reports
    update_results_in_reports.main()
    
    total = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  HOÀN THÀNH TOÀN BỘ PIPELINE TRONG {total:.1f}s ({total/60:.1f} phút)")
    print("=" * 70)

if __name__ == "__main__":
    main()
