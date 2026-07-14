import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

df = pd.read_csv(r'D:\draft 2\data\master_panel_all_firms.csv')
os.makedirs(r'D:\draft 2\results\figures', exist_ok=True)

# --- 1. LOWESS PLOT ---
print('Đang vẽ biểu đồ LOWESS...')
plot_df = df.dropna(subset=['td_a', 'roa']).copy()
q_t = plot_df['td_a'].quantile([0.01, 0.99])
q_r = plot_df['roa'].quantile([0.01, 0.99])
clean_df = plot_df[(plot_df['td_a'] >= q_t[0.01]) & (plot_df['td_a'] <= q_t[0.99]) &
                   (plot_df['roa'] >= q_r[0.01]) & (plot_df['roa'] <= q_r[0.99])]

sample_df = clean_df.sample(n=min(5000, len(clean_df)), random_state=42).sort_values('td_a')

plt.figure(figsize=(10, 6))
plt.scatter(sample_df['td_a'], sample_df['roa'], alpha=0.1, color='gray', label='Dữ liệu thực tế (Sample)')

lowess = sm.nonparametric.lowess(sample_df['roa'], sample_df['td_a'], frac=0.3)
plt.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=3, label='LOWESS Curve (Xu hướng)')

plt.title('Mối quan hệ phi tuyến giữa Nợ (TD_A) và Lợi nhuận (ROA)', fontsize=14, fontweight='bold')
plt.xlabel('Tỷ lệ Nợ trên Tổng Tài Sản (TD_A)', fontsize=12)
plt.ylabel('Tỷ suất Sinh lời trên Tài sản (ROA)', fontsize=12)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'D:\draft 2\results\figures\lowess_tda_roa.png', dpi=300)
print('✅ Đã lưu biểu đồ LOWESS tại: D:\\draft 2\\results\\figures\\lowess_tda_roa.png')

# --- 2. DATA VERIFICATION ---
print('\n=== KIỂM CHỨNG DỮ LIỆU THỰC TẾ ===')
tickers_to_check = ['VNM', 'FPT', 'HPG', 'VIC', 'VCB', 'MWG']
for t in tickers_to_check:
    firm_data = df[df['ticker'] == t]
    if len(firm_data) > 0:
        latest = firm_data.sort_values('period').iloc[-1]
        print(f"[{t}] Kỳ mới nhất trong data: {latest['period']}")
        print(f"   - ROA: {latest['roa']*100:.2f}%")
        print(f"   - Nợ / Tổng TS (TD_A): {latest['td_a']*100:.2f}%")
        print(f"   - Tổng tài sản (Log): {latest['firm_size']:.2f}")
