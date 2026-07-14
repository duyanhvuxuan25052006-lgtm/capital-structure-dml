import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np

df = pd.read_csv('D:/draft 2/data/raw_financial_ratios_all.csv')
print("=== TAT CA COT TRONG DU LIEU THO ===")
for i, c in enumerate(df.columns):
    print(f"  {i:3d}: {c}")

print(f"\nTong cot: {len(df.columns)}")
print(f"Tong dong: {len(df)}")

# Kiem tra debt_to_equity
if 'debt_to_equity' in df.columns:
    de = pd.to_numeric(df['debt_to_equity'], errors='coerce')
    print(f"\n=== DEBT_TO_EQUITY ===")
    print(f"  Non-null: {de.notna().sum()}")
    print(f"  Min: {de.min():.4f}")
    print(f"  Max: {de.max():.4f}")
    print(f"  Mean: {de.mean():.4f}")
    print(f"  D/E < 0 (equity am): {(de < 0).sum()} obs ({(de < 0).sum()/de.notna().sum()*100:.2f}%)")
    print(f"  D/E = -1 (chia 0): {(de == -1).sum()} obs")
    print(f"  D/E < -0.5: {(de < -0.5).sum()} obs")

# Tim cot lien quan den debt/assets
debt_cols = [c for c in df.columns if 'debt' in c.lower() or 'asset' in c.lower() or 'leverage' in c.lower()]
print(f"\n=== COT LIEN QUAN DEN DEBT/ASSETS ===")
for c in debt_cols:
    vals = pd.to_numeric(df[c], errors='coerce')
    print(f"  {c}: non-null={vals.notna().sum()}, min={vals.min():.4f}, max={vals.max():.4f}")

# Kiem tra doubleml version
import doubleml
print(f"\n=== DOUBLEML VERSION: {doubleml.__version__} ===")
