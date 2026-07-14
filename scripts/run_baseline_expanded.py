"""
BƯỚC 4 (AUDITED & UPGRADED): BASELINE MODELS (OLS, Entity FE, Two-way FE)
=====================================================================
IMPLEMENTED AUDIT UPGRADES:
  1. Hồi quy cho cả 2 loại biến kết quả (Y):
     - roa (Tỷ suất sinh lời sau thuế)
     - operating_roa (Tỷ suất sinh lời trước lãi vay - EBIT/Assets) (A1)
  2. Hồi quy cho cả 2 loại đòn bẩy can thiệp (D):
     - td_a (Total Liabilities / Total Assets)
     - ibd_a (Interest-bearing Debt / Total Assets)
  3. Sử dụng bộ kiểm soát đầy đủ (W) có quy mô và tangibility.
=====================================================================
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm

INPUT_FILE = r"D:\draft 2\data\master_panel_all_firms.csv"
OUTPUT_FILE = r"D:\draft 2\results\baseline_results.csv"

def run_regressions(df, y_col, d_col, x_cols):
    print(f"\n" + "-" * 60)
    print(f"HỒI QUY BASELINE: Y = {y_col.upper()} | D = {d_col.upper()}")
    print("-" * 60)
    
    all_vars = [y_col, d_col] + x_cols
    df_clean = df.dropna(subset=all_vars).copy()
    
    results = []
    
    # 1. Pooled OLS
    X_ols = sm.add_constant(df_clean[[d_col] + x_cols])
    y_ols = df_clean[y_col]
    ols_model = sm.OLS(y_ols, X_ols).fit(cov_type='cluster', cov_kwds={'groups': df_clean['ticker']})
    ci = ols_model.conf_int().loc[d_col]
    results.append({
        'Y_variable': y_col, 'Treatment': d_col, 'Model': 'Pooled OLS',
        'theta': ols_model.params[d_col], 'se': ols_model.bse[d_col],
        'pval': ols_model.pvalues[d_col], 'ci_low': ci[0], 'ci_high': ci[1],
        'r2': ols_model.rsquared, 'n_obs': int(ols_model.nobs)
    })
    
    # 2. Entity FE
    vars_to_demean = [y_col, d_col] + x_cols
    grouped_mean = df_clean.groupby('ticker')[vars_to_demean].transform('mean')
    df_demeaned = df_clean.copy()
    for col in vars_to_demean:
        df_demeaned[col] = df_clean[col] - grouped_mean[col]
        
    X_fe = df_demeaned[[d_col] + x_cols]
    y_fe = df_demeaned[y_col]
    fe_model = sm.OLS(y_fe, X_fe).fit(cov_type='cluster', cov_kwds={'groups': df_clean['ticker']})
    ci = fe_model.conf_int().loc[d_col]
    results.append({
        'Y_variable': y_col, 'Treatment': d_col, 'Model': 'Entity FE',
        'theta': fe_model.params[d_col], 'se': fe_model.bse[d_col],
        'pval': fe_model.pvalues[d_col], 'ci_low': ci[0], 'ci_high': ci[1],
        'r2': fe_model.rsquared, 'n_obs': int(fe_model.nobs)
    })
    
    # 3. Two-way FE
    time_dummies = pd.get_dummies(df_clean['period'], prefix='time', drop_first=True).astype(float)
    X_twfe_vars = df_demeaned[[d_col] + x_cols].reset_index(drop=True)
    X_twfe_df = pd.concat([X_twfe_vars, time_dummies.reset_index(drop=True)], axis=1)
    y_twfe = df_demeaned[y_col].reset_index(drop=True).astype(float)
    groups_twfe = df_clean['ticker'].reset_index(drop=True)
    twfe_model = sm.OLS(y_twfe, X_twfe_df).fit(cov_type='cluster', cov_kwds={'groups': groups_twfe})
    ci = twfe_model.conf_int().loc[d_col]
    results.append({
        'Y_variable': y_col, 'Treatment': d_col, 'Model': 'Two-way FE',
        'theta': twfe_model.params[d_col], 'se': twfe_model.bse[d_col],
        'pval': twfe_model.pvalues[d_col], 'ci_low': ci[0], 'ci_high': ci[1],
        'r2': twfe_model.rsquared, 'n_obs': int(twfe_model.nobs)
    })
    
    print(f"  Two-way FE θ = {twfe_model.params[d_col]:.6f}, SE = {twfe_model.bse[d_col]:.6f}, p = {twfe_model.pvalues[d_col]:.6f}")
    return pd.DataFrame(results)

def main():
    print("=" * 70)
    print("  BƯỚC 4 (AUDITED & UPGRADED): BASELINE MODELS")
    print("=" * 70)
    
    df = pd.read_csv(INPUT_FILE)
    print(f"Master Dataset: {df.shape[0]} obs, {df['ticker'].nunique()} firms")
    
    x_cols = ['current_ratio', 'quick_ratio', 'asset_turnover', 'gross_margin', 
              'firm_size', 'tangibility', 'data_tenure']
              
    all_results = []
    
    # Chạy tổ hợp giữa 2 Outcome (roa, operating_roa) và 2 Treatment (td_a, ibd_a)
    for y_var in ['roa', 'operating_roa']:
        for d_var in ['td_a', 'ibd_a']:
            df_res = run_regressions(df, y_var, d_var, x_cols)
            all_results.append(df_res)
            
    df_all_results = pd.concat(all_results, ignore_index=True)
    df_all_results.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "=" * 70)
    print("  BẢNG KẾT QUẢ BASELINE TỔNG HỢP (AUDITED)")
    print("=" * 70)
    print(df_all_results.to_string(index=False))

if __name__ == "__main__":
    main()
