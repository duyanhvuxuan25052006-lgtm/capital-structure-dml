import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\draft 2\data\master_panel_all_firms.csv')

# Loc doanh nghiep vua va nho (large_cap == 0)
sme_df = df[(df['large_cap'] == 0) & (df['roa'].notna()) & (df['td_a'].notna())]

np.random.seed(42)
random_sme_tickers = np.random.choice(sme_df['ticker'].unique(), 5, replace=False)

print('=== KIỂM CHỨNG DỮ LIỆU: DOANH NGHIỆP VỪA VÀ NHỎ (SME) ===')
for t in random_sme_tickers:
    firm_data = df[df['ticker'] == t].sort_values('period')
    latest = firm_data.iloc[-1]
    
    print(f"[{t}] Kỳ: {latest['period']} | Sàn: {latest['exchange']}")
    print(f"   - Lợi nhuận / TS (ROA): {latest['roa']*100:.2f}%")
    print(f"   - Nợ / Tổng TS (TD_A): {latest['td_a']*100:.2f}%")
    print(f"   - Tổng tài sản (Log): {latest['firm_size']:.2f}")
