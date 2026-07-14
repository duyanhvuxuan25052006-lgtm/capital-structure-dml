"""
Test download financial ratios for a RANDOM sample of 10 tickers
(mix of large and small firms) to check data availability and quality.
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np

# Read the full non-financial ticker list
df_tickers = pd.read_csv("D:/draft 2/data/all_nonfinancial_tickers.csv")
print(f"Total non-financial tickers: {len(df_tickers)}")

# Random sample of 10 tickers (mix positions to get large + small firms)
np.random.seed(42)
sample_indices = np.random.choice(len(df_tickers), 10, replace=False)
sample_tickers = df_tickers.iloc[sample_indices]['symbol'].tolist()
print(f"Test sample: {sample_tickers}")
print()

from vnstock.api.financial import Finance

results = []
for i, ticker in enumerate(sample_tickers):
    print(f"[{i+1}/10] Testing: {ticker}...", end=" ")
    try:
        f = Finance(source="vci", symbol=ticker, period="quarter")
        df_raw = f._get_report(report_type="ratio", limit=100, mode="final")
        
        if df_raw.empty:
            print("NO DATA")
            results.append({'ticker': ticker, 'status': 'NO_DATA', 'periods': 0, 'has_roa': False, 'has_debt': False})
        else:
            # Check what items are available
            items = df_raw['item_id'].tolist() if 'item_id' in df_raw.columns else []
            n_cols = len([c for c in df_raw.columns if c not in ['item', 'item_en', 'item_id']])
            has_roa = 'roa' in items or 'roaa' in items
            has_debt = 'debt_to_assets' in items or 'debt_to_equity' in items or 'total_debt_to_equity' in items
            has_cr = 'current_ratio' in items
            
            print(f"OK - {n_cols} periods, ROA={has_roa}, Debt={has_debt}, CR={has_cr}")
            results.append({
                'ticker': ticker, 'status': 'OK', 'periods': n_cols,
                'has_roa': has_roa, 'has_debt': has_debt, 'has_cr': has_cr,
                'items_sample': items[:20]
            })
        
        time.sleep(4.5)
        
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({'ticker': ticker, 'status': 'ERROR', 'periods': 0})
        time.sleep(10)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
ok_count = sum(1 for r in results if r['status'] == 'OK')
print(f"Success: {ok_count}/10")
print(f"No data: {sum(1 for r in results if r['status'] == 'NO_DATA')}/10")
print(f"Error: {sum(1 for r in results if r['status'] == 'ERROR')}/10")

# Show available item_ids from first successful result
for r in results:
    if r['status'] == 'OK' and 'items_sample' in r:
        print(f"\nAvailable financial ratio items (from {r['ticker']}):")
        for item in r['items_sample']:
            print(f"  - {item}")
        break

# Estimate download time
est_time_min = 1438 * 5 / 60
print(f"\nEstimated download time for all 1438 tickers: ~{est_time_min:.0f} minutes ({est_time_min/60:.1f} hours)")
print(f"With rate-limit retries: ~{est_time_min * 1.3:.0f} minutes ({est_time_min*1.3/60:.1f} hours)")
