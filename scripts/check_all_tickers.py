import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.makedirs("D:/draft 2/data", exist_ok=True)

from vnstock import Reference
ref = Reference()
df_all = ref.equity.list()
df3 = df_all[df_all['symbol'].str.len() == 3].copy()
print("Tong so ma 3 chu cai:", len(df3))

# Loc bo tai chinh
exclude_kw = ['ngân hàng', 'chứng khoán', 'bảo hiểm', 'quỹ', 'invest', 'securities', 
              'bank', 'insurance', 'fund', 'capital', 'finance', 'tài chính']
df3['name_lower'] = df3['organ_name'].str.lower()
mask = df3['name_lower'].apply(lambda n: any(kw in str(n) for kw in exclude_kw))
df_financial = df3[mask]
df_nonfinancial = df3[~mask]
print("So ma tai chinh (loai bo):", len(df_financial))
print("So ma phi tai chinh (giu lai):", len(df_nonfinancial))
print()

print("=== CAC MA TAI CHINH BI LOAI (mau) ===")
for _, r in df_financial.head(20).iterrows():
    sym = r['symbol']
    name = r['organ_name']
    print(f"  {sym}: {name}")

print()
print("=== CAC MA PHI TAI CHINH (mau) ===")
for _, r in df_nonfinancial.head(10).iterrows():
    sym = r['symbol']
    name = r['organ_name']
    print(f"  {sym}: {name}")

# Save
df_nonfinancial[['symbol', 'organ_name']].to_csv(
    'D:/draft 2/data/all_nonfinancial_tickers.csv', index=False, encoding='utf-8-sig')
print()
print("Da luu danh sach:", len(df_nonfinancial), "ma vao all_nonfinancial_tickers.csv")
