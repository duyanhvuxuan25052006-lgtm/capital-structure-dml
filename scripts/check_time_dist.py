import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd

df = pd.read_csv('D:/draft 2/data/master_panel_all_firms.csv')
df['year'] = df['period'].str[:4].astype(int)

print('=== PHAN BO QUAN SAT THEO NAM ===')
yearly = df.groupby('year').agg(n_obs=('ticker', 'size'), n_firms=('ticker', 'nunique'))
for yr, row in yearly.iterrows():
    n_obs = int(row['n_obs'])
    n_firms = int(row['n_firms'])
    print(f'  {yr}: {n_obs:>5d} obs, {n_firms:>5d} DN')

print('\n=== KY BAO CAO (period) ===')
period_counts = df['period'].value_counts().sort_index()
for p, c in period_counts.items():
    print(f'  {p}: {c:>5d} obs')

print('\n=== NAM SOM NHAT CO DU LIEU CUA MOI DN ===')
first_year = df.groupby('ticker')['year'].min()
fy_counts = first_year.value_counts().sort_index()
for yr, cnt in fy_counts.items():
    print(f'  {yr}: {cnt} DN bat dau co du lieu')

print('\n=== NAM MUON NHAT CO DU LIEU CUA MOI DN ===')
last_year = df.groupby('ticker')['year'].max()
ly_counts = last_year.value_counts().sort_index()
for yr, cnt in ly_counts.items():
    print(f'  {yr}: {cnt} DN con du lieu')
