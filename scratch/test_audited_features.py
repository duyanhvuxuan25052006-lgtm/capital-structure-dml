import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import doubleml as dml

def main():
    print("=== TESTING AUDITED FEATURES AND PIPELINE UPDATES ===")
    
    # Load raw data
    raw_path = "D:/draft 2/data/raw_financial_ratios_all.csv"
    df = pd.read_csv(raw_path)
    print(f"Loaded raw data: {df.shape}")
    
    # 1. Test D/A and IBD/A calculations
    df['de'] = pd.to_numeric(df['debt_to_equity'], errors='coerce')
    df['ibd_e'] = pd.to_numeric(df['debtPerEquity'], errors='coerce')
    
    # Exclude negative equity
    df_valid = df[(df['de'] >= 0) & (df['de'] <= 100) & (df['ibd_e'] >= 0)].copy()
    
    df_valid['td_a'] = df_valid['de'] / (1 + df_valid['de'])
    df_valid['ibd_a'] = df_valid['ibd_e'] / (1 + df_valid['de'])
    
    print("\nLeverage summary (HPG sample as verification):")
    hpg = df_valid[df_valid['ticker'] == 'HPG'].head(3)
    print(hpg[['ticker', 'period', 'de', 'ibd_e', 'td_a', 'ibd_a']].to_string())
    
    # 2. Test Control Variables: size and tangibility
    df_valid['market_cap'] = pd.to_numeric(df_valid['market_cap'], errors='coerce')
    df_valid['firm_size'] = np.log(df_valid['market_cap'].replace(0, np.nan))
    
    df_valid['at'] = pd.to_numeric(df_valid['asset_turnover'], errors='coerce')
    df_valid['fat'] = pd.to_numeric(df_valid['fixed_asset_turnover'], errors='coerce')
    
    # Tangibility = asset_turnover / fixed_asset_turnover
    # If fat is 0 or negative, set tangibility to 0, clip between 0 and 1
    df_valid['tangibility'] = np.where(df_valid['fat'] > 0, df_valid['at'] / df_valid['fat'], 0.0)
    df_valid['tangibility'] = df_valid['tangibility'].clip(0, 1)
    
    print("\nControls summary (HPG sample):")
    print(df_valid[df_valid['ticker'] == 'HPG'][['ticker', 'period', 'market_cap', 'firm_size', 'at', 'fat', 'tangibility']].head(3).to_string())
    
    # 3. Test Quarter-by-quarter Winsorization
    print("\nTesting quarter-by-quarter winsorization...")
    cols_to_winsorize = ['roa', 'td_a', 'ibd_a', 'current_ratio', 'quick_ratio', 'asset_turnover', 'gross_margin', 'firm_size', 'tangibility']
    for col in cols_to_winsorize:
        df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce')
        
    df_valid = df_valid.dropna(subset=cols_to_winsorize + ['ticker', 'period']).copy()
    
    def winsorize_group(group):
        for col in cols_to_winsorize:
            p1 = group[col].quantile(0.01)
            p99 = group[col].quantile(0.99)
            group[col] = group[col].clip(p1, p99)
        return group
        
    df_winsorized = df_valid.groupby('period', group_keys=False).apply(winsorize_group)
    print(f"Winsorized dataset shape: {df_winsorized.shape}")
    
    # 4. Test VN30 / Large-Cap Split
    # Calculate average market cap for each firm to classify Large-Cap
    avg_mc = df_winsorized.groupby('ticker')['market_cap'].mean().reset_index()
    avg_mc = avg_mc.sort_values(by='market_cap', ascending=False)
    
    # Top 100 firms by average market cap
    large_cap_tickers = set(avg_mc.head(100)['ticker'])
    df_winsorized['large_cap'] = df_winsorized['ticker'].isin(large_cap_tickers).astype(int)
    
    print(f"\nSample breakdown:")
    print(f"  Large-Cap (Top 100 firms) obs: {df_winsorized['large_cap'].sum()}")
    print(f"  Mid/Small-Cap obs           : {len(df_winsorized) - df_winsorized['large_cap'].sum()}")
    
    # 5. Test DoubleML with two-way clustering
    print("\nTesting DoubleML two-way clustering...")
    df_sample = df_winsorized.head(1000).copy() # Use subset for speed
    
    # Demean variables
    vars_to_demean = ['roa', 'td_a'] + ['current_ratio', 'quick_ratio', 'asset_turnover', 'gross_margin', 'firm_size', 'tangibility']
    grouped_mean = df_sample.groupby('ticker')[vars_to_demean].transform('mean')
    for col in vars_to_demean:
        df_sample[col] = df_sample[col] - grouped_mean[col]
        
    # Time dummies
    time_dummies = pd.get_dummies(df_sample['period'], prefix='time', drop_first=True).astype(float)
    X = np.hstack([df_sample[['current_ratio', 'quick_ratio', 'asset_turnover', 'gross_margin', 'firm_size', 'tangibility']].values, time_dummies.values])
    Y = df_sample['roa'].values
    D = df_sample['td_a'].values
    
    ticker_codes = pd.Categorical(df_sample['ticker']).codes
    period_codes = pd.Categorical(df_sample['period']).codes
    
    # Two-way cluster array: shape (n_obs, 2)
    two_way_clusters = np.column_stack([ticker_codes, period_codes])
    
    try:
        dml_data = dml.DoubleMLData.from_arrays(x=X, y=Y, d=D, cluster_vars=two_way_clusters)
        print("DoubleMLData with two-way clustering: SUCCESS!")
        rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42) # fast config
        model = dml.DoubleMLPLR(dml_data, ml_l=rf, ml_m=rf, n_folds=2, n_rep=1)
        model.fit()
        print(f"Fit completed successfully! theta = {model.coef[0]:.6f}, SE = {model.se[0]:.6f}")
    except Exception as e:
        print(f"Failed with two-way clustering: {e}")
        print("Will fall back to one-way firm clustering if needed.")

if __name__ == "__main__":
    main()
