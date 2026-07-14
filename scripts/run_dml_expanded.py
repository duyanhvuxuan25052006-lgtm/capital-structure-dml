"""
BƯỚC 5 (AUDITED & UPGRADED): DML-PLR PIPELINE
=============================================
IMPLEMENTED AUDIT UPGRADES:
  1. Hồi quy cho cả 2 loại đòn bẩy: td_a và ibd_a.
  2. Bổ sung các biến kiểm soát W: size, tangibility.
  3. Xử lý One-way (firm) và Two-way (firm + quarter) clustering (A6).
  4. Hồi quy trễ 1 kỳ (td_a_lag, ibd_a_lag) để xử lý nhân quả ngược (A1).
  5. Hồi quy bậc hai (td_a, td_a_sq) để kiểm định chữ U ngược (A2).
  6. Hồi quy Operating ROA (EBIT/Assets) để loại bỏ kênh kế toán (A1).
  7. Hồi quy loại UPCoM (chỉ giữ HOSE+HNX) để xử lý nhiễu UPCoM (A5).
  8. Kiểm định giả dược (Placebo check) bằng cách hoán vị D (A6).
  9. Chạy phân nhóm HTE: Large-cap (Top 100) vs Mid/Small-cap (Point 9).
  10. Tính toán out-of-fold R² và log ElasticNetCV parameters (Point 4, 8).
  11. Giữ nguyên cấu hình chuẩn học thuật: n_folds = 5, n_rep = 5.
=============================================
"""
import sys, os, warnings, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import doubleml as dml

INPUT_MASTER = r"D:\draft 2\data\master_panel_all_firms.csv"
OUTPUT_DIR = r"D:\draft 2\results"

N_FOLDS = 5
N_REP = 5

def get_oof_r2(model, key, actual):
    """Tính R² out-of-fold trung bình qua các reps."""
    preds = model.predictions[key][:, :, 0]  # shape (n_obs, n_rep)
    r2_list = []
    for r in range(preds.shape[1]):
        u = ((actual - preds[:, r]) ** 2).sum()
        v = ((actual - actual.mean()) ** 2).sum()
        r2_list.append(1 - u / v)
    return np.mean(r2_list)

def run_dml_plr_spec(dml_data, y_actual, d_actual, rf_estimators=100):
    """Ước lượng DML PLR cho 3 thuật toán và trả về kết quả."""
    # 1. LassoCV
    lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=10000, random_state=42))
    dml_lasso = dml.DoubleMLPLR(dml_data, ml_l=lasso, ml_m=lasso, n_folds=N_FOLDS, n_rep=N_REP)
    dml_lasso.fit()
    
    # 2. ElasticNetCV
    enet = make_pipeline(StandardScaler(), ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], max_iter=10000, random_state=42))
    dml_enet = dml.DoubleMLPLR(dml_data, ml_l=enet, ml_m=enet, n_folds=N_FOLDS, n_rep=N_REP)
    dml_enet.fit(store_models=True)
    
    enet_l1_ratios = []
    for r in range(N_REP):
        for f in range(N_FOLDS):
            est = dml_enet.models['ml_l']['d'][r][f]
            # Since est is a Pipeline, we access the named step 'elasticnetcv'
            enet_l1_ratios.append(est.named_steps['elasticnetcv'].l1_ratio_)
    mean_l1 = np.mean(enet_l1_ratios)
    
    # 3. RandomForest
    rf_l = RandomForestRegressor(n_estimators=rf_estimators, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_m = RandomForestRegressor(n_estimators=rf_estimators, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
    dml_rf = dml.DoubleMLPLR(dml_data, ml_l=rf_l, ml_m=rf_m, n_folds=N_FOLDS, n_rep=N_REP)
    dml_rf.fit()
    
    r2_l_lasso = get_oof_r2(dml_lasso, 'ml_l', y_actual)
    r2_m_lasso = get_oof_r2(dml_lasso, 'ml_m', d_actual)
    r2_l_rf = get_oof_r2(dml_rf, 'ml_l', y_actual)
    r2_m_rf = get_oof_r2(dml_rf, 'ml_m', d_actual)
    
    # Sensitivity analysis
    dml_rf.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    rv = dml_rf.sensitivity_params['rv'][0] * 100
    rva = dml_rf.sensitivity_params['rva'][0] * 100
    
    return {
        'Lasso_theta': dml_lasso.coef[0], 'Lasso_se': dml_lasso.se[0], 'Lasso_pval': dml_lasso.pval[0],
        'Lasso_R2_y': r2_l_lasso, 'Lasso_R2_d': r2_m_lasso,
        'Enet_theta': dml_enet.coef[0], 'Enet_se': dml_enet.se[0], 'Enet_pval': dml_enet.pval[0],
        'Enet_mean_l1': mean_l1,
        'RF_theta': dml_rf.coef[0], 'RF_se': dml_rf.se[0], 'RF_pval': dml_rf.pval[0],
        'RF_R2_y': r2_l_rf, 'RF_R2_d': r2_m_rf, 'RF_RV': rv, 'RF_RVa': rva
    }

def main():
    print("=" * 80)
    print("  BƯỚC 5 (AUDITED & UPGRADED): DML-PLR PIPELINE")
    print("=" * 80)
    
    df = pd.read_csv(INPUT_MASTER)
    y_col = 'roa'
    x_cols = ['current_ratio', 'quick_ratio', 'asset_turnover', 'gross_margin', 
              'firm_size', 'tangibility', 'data_tenure']
              
    results = []
    
    # ════════════════════════════════════════════════════════════════════
    # I. CHẠY CÁC MÔ HÌNH CHÍNH (MAIN MODELS)
    # ════════════════════════════════════════════════════════════════════
    print("\n>>> I. CHẠY CÁC MÔ HÌNH CHÍNH (Full sample, Lagged, Placebo, UPCoM, Operating ROA)...")
    
    for t_name in ['td_a', 'ibd_a']:
        # 1. Full sample (One-way cụm theo DN)
        df_sub = df.dropna(subset=[y_col, t_name] + x_cols).copy()
        
        # Demean
        grouped_mean = df_sub.groupby('ticker')[[y_col, t_name] + x_cols].transform('mean')
        df_dem = df_sub.copy()
        for col in [y_col, t_name] + x_cols:
            df_dem[col] = df_sub[col] - grouped_mean[col]
            
        time_dum = pd.get_dummies(df_sub['period'], prefix='time', drop_first=True).astype(float).values
        X_all = np.hstack([df_dem[x_cols].values, time_dum])
        Y_val = df_dem[y_col].values
        D_val = df_dem[t_name].values
        
        ticker_codes = pd.Categorical(df_sub['ticker']).codes
        period_codes = pd.Categorical(df_sub['period']).codes
        clusters_oneway = ticker_codes.reshape(-1, 1)
        clusters_twoway = np.column_stack([ticker_codes, period_codes])
        
        # Hồi quy One-way
        dml_data_oneway = dml.DoubleMLData.from_arrays(x=X_all, y=Y_val, d=D_val, cluster_vars=clusters_oneway)
        res_oneway = run_dml_plr_spec(dml_data_oneway, Y_val, D_val)
        res_oneway.update({'Treatment': t_name, 'Model_Specification': 'Full Sample (One-way Cluster)', 'Y_variable': y_col, 'N_obs': len(Y_val)})
        results.append(res_oneway)
        
        # Hồi quy Two-way (Point A6)
        dml_data_twoway = dml.DoubleMLData.from_arrays(x=X_all, y=Y_val, d=D_val, cluster_vars=clusters_twoway)
        res_twoway = run_dml_plr_spec(dml_data_twoway, Y_val, D_val)
        res_twoway.update({'Treatment': t_name, 'Model_Specification': 'Full Sample (Two-way Cluster)', 'Y_variable': y_col, 'N_obs': len(Y_val)})
        results.append(res_twoway)
        
        # 2. Hồi quy trễ 1 kỳ (Lagged Treatment - Point A1)
        t_lag = t_name + '_lag'
        df_lag = df.dropna(subset=[y_col, t_lag] + x_cols).copy()
        grouped_mean_lag = df_lag.groupby('ticker')[[y_col, t_lag] + x_cols].transform('mean')
        df_dem_lag = df_lag.copy()
        for col in [y_col, t_lag] + x_cols:
            df_dem_lag[col] = df_lag[col] - grouped_mean_lag[col]
            
        time_dum_lag = pd.get_dummies(df_lag['period'], prefix='time', drop_first=True).astype(float).values
        X_lag = np.hstack([df_dem_lag[x_cols].values, time_dum_lag])
        Y_lag = df_dem_lag[y_col].values
        D_lag = df_dem_lag[t_lag].values
        clusters_lag = pd.Categorical(df_lag['ticker']).codes.reshape(-1, 1)
        
        dml_data_lag = dml.DoubleMLData.from_arrays(x=X_lag, y=Y_lag, d=D_lag, cluster_vars=clusters_lag)
        res_lag = run_dml_plr_spec(dml_data_lag, Y_lag, D_lag)
        res_lag.update({'Treatment': t_lag, 'Model_Specification': 'Lagged Treatment', 'Y_variable': y_col, 'N_obs': len(Y_lag)})
        results.append(res_lag)
        
        # 3. Hồi quy giả dược (Placebo check - Point A6)
        # Hoán vị ngẫu nhiên cột treatment D
        np.random.seed(42)
        D_placebo = np.random.permutation(D_val)
        dml_data_placebo = dml.DoubleMLData.from_arrays(x=X_all, y=Y_val, d=D_placebo, cluster_vars=clusters_oneway)
        res_placebo = run_dml_plr_spec(dml_data_placebo, Y_val, D_placebo)
        res_placebo.update({'Treatment': t_name + '_placebo', 'Model_Specification': 'Placebo (Permuted D)', 'Y_variable': y_col, 'N_obs': len(Y_val)})
        results.append(res_placebo)
        
        # 4. Hồi quy loại bỏ UPCoM (Chỉ HOSE + HNX - Point A5)
        df_no_upcom = df[df['exchange'].isin(['HOSE', 'HNX'])].dropna(subset=[y_col, t_name] + x_cols).copy()
        grouped_mean_nu = df_no_upcom.groupby('ticker')[[y_col, t_name] + x_cols].transform('mean')
        df_dem_nu = df_no_upcom.copy()
        for col in [y_col, t_name] + x_cols:
            df_dem_nu[col] = df_no_upcom[col] - grouped_mean_nu[col]
            
        time_dum_nu = pd.get_dummies(df_no_upcom['period'], prefix='time', drop_first=True).astype(float).values
        X_nu = np.hstack([df_dem_nu[x_cols].values, time_dum_nu])
        Y_nu = df_dem_nu[y_col].values
        D_nu = df_dem_nu[t_name].values
        clusters_nu = pd.Categorical(df_no_upcom['ticker']).codes.reshape(-1, 1)
        
        dml_data_nu = dml.DoubleMLData.from_arrays(x=X_nu, y=Y_nu, d=D_nu, cluster_vars=clusters_nu)
        res_nu = run_dml_plr_spec(dml_data_nu, Y_nu, D_nu)
        res_nu.update({'Treatment': t_name, 'Model_Specification': 'Excluding UPCoM (HOSE+HNX only)', 'Y_variable': y_col, 'N_obs': len(Y_nu)})
        results.append(res_nu)
        
        # 5. Hồi quy Operating ROA (EBIT/Assets - Point A1)
        y_op = 'operating_roa'
        df_op = df.dropna(subset=[y_op, t_name] + x_cols).copy()
        grouped_mean_op = df_op.groupby('ticker')[[y_op, t_name] + x_cols].transform('mean')
        df_dem_op = df_op.copy()
        for col in [y_op, t_name] + x_cols:
            df_dem_op[col] = df_op[col] - grouped_mean_op[col]
            
        time_dum_op = pd.get_dummies(df_op['period'], prefix='time', drop_first=True).astype(float).values
        X_op = np.hstack([df_dem_op[x_cols].values, time_dum_op])
        Y_op = df_dem_op[y_op].values
        D_op = df_dem_op[t_name].values
        clusters_op = pd.Categorical(df_op['ticker']).codes.reshape(-1, 1)
        
        dml_data_op = dml.DoubleMLData.from_arrays(x=X_op, y=Y_op, d=D_op, cluster_vars=clusters_op)
        res_op = run_dml_plr_spec(dml_data_op, Y_op, D_op)
        res_op.update({'Treatment': t_name, 'Model_Specification': 'Operating ROA (EBIT/Assets)', 'Y_variable': y_op, 'N_obs': len(Y_op)})
        results.append(res_op)
        
        # 6. HTE Subgroup split (Large cap vs Mid/Small cap - Point 9)
        for sub_name, mask in [('Large-Cap (Top 100)', df['large_cap'] == 1), ('Mid/Small-Cap', df['large_cap'] == 0)]:
            df_subg = df[mask].dropna(subset=[y_col, t_name] + x_cols).copy()
            grouped_mean_sg = df_subg.groupby('ticker')[[y_col, t_name] + x_cols].transform('mean')
            df_dem_sg = df_subg.copy()
            for col in [y_col, t_name] + x_cols:
                df_dem_sg[col] = df_subg[col] - grouped_mean_sg[col]
                
            time_dum_sg = pd.get_dummies(df_subg['period'], prefix='time', drop_first=True).astype(float).values
            X_sg = np.hstack([df_dem_sg[x_cols].values, time_dum_sg])
            Y_sg = df_dem_sg[y_col].values
            D_sg = df_dem_sg[t_name].values
            clusters_sg = pd.Categorical(df_subg['ticker']).codes.reshape(-1, 1)
            
            dml_data_sg = dml.DoubleMLData.from_arrays(x=X_sg, y=Y_sg, d=D_sg, cluster_vars=clusters_sg)
            res_sg = run_dml_plr_spec(dml_data_sg, Y_sg, D_sg)
            res_sg.update({'Treatment': t_name, 'Model_Specification': f'Subgroup: {sub_name}', 'Y_variable': y_col, 'N_obs': len(Y_sg)})
            results.append(res_sg)

    # ════════════════════════════════════════════════════════════════════
    # II. HỒI QUY ĐÒN BẨY BẬC HAI (QUADRATIC DML - Point A2)
    # ════════════════════════════════════════════════════════════════════
    print("\n>>> II. CHẠY HỒI QUY ĐÒN BẨY BẬC HAI (Quadratic DML - Trade-off curve)...")
    
    for t_name in ['td_a', 'ibd_a']:
        t_sq = t_name + '_sq'
        df_quad = df.dropna(subset=[y_col, t_name, t_sq] + x_cols).copy()
        
        # Demean
        grouped_mean_q = df_quad.groupby('ticker')[[y_col, t_name, t_sq] + x_cols].transform('mean')
        df_dem_q = df_quad.copy()
        for col in [y_col, t_name, t_sq] + x_cols:
            df_dem_q[col] = df_quad[col] - grouped_mean_q[col]
            
        time_dum_q = pd.get_dummies(df_quad['period'], prefix='time', drop_first=True).astype(float).values
        X_q = np.hstack([df_dem_q[x_cols].values, time_dum_q])
        Y_q = df_dem_q[y_col].values
        
        # Vector treatment: [D, D^2]
        D_q = df_dem_q[[t_name, t_sq]].values
        clusters_q = pd.Categorical(df_quad['ticker']).codes.reshape(-1, 1)
        
        dml_data_q = dml.DoubleMLData.from_arrays(x=X_q, y=Y_q, d=D_q, cluster_vars=clusters_q)
        
        # Khớp RandomForest cho Quadratic
        rf_l = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf_m = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
        dml_rf_q = dml.DoubleMLPLR(dml_data_q, ml_l=rf_l, ml_m=rf_m, n_folds=N_FOLDS, n_rep=N_REP)
        dml_rf_q.fit()
        
        # Hồi quy bậc hai trả về 2 theta (linear và quadratic)
        print(f"  Quadratic {t_name.upper()} -> Linear theta: {dml_rf_q.coef[0]:.6f}, Quadratic theta: {dml_rf_q.coef[1]:.6f}")
        
        results.append({
            'Treatment': t_name, 'Model_Specification': 'Quadratic DML (Linear Component)', 'Y_variable': y_col,
            'RF_theta': dml_rf_q.coef[0], 'RF_se': dml_rf_q.se[0], 'RF_pval': dml_rf_q.pval[0],
            'Lasso_theta': np.nan, 'Lasso_se': np.nan, 'Lasso_pval': np.nan,
            'Enet_theta': np.nan, 'Enet_se': np.nan, 'Enet_pval': np.nan, 'N_obs': len(Y_q)
        })
        results.append({
            'Treatment': t_sq, 'Model_Specification': 'Quadratic DML (Quadratic Component)', 'Y_variable': y_col,
            'RF_theta': dml_rf_q.coef[1], 'RF_se': dml_rf_q.se[1], 'RF_pval': dml_rf_q.pval[1],
            'Lasso_theta': np.nan, 'Lasso_se': np.nan, 'Lasso_pval': np.nan,
            'Enet_theta': np.nan, 'Enet_se': np.nan, 'Enet_pval': np.nan, 'N_obs': len(Y_q)
        })

    # Lưu kết quả DML tổng hợp
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'dml_results_expanded.csv'), index=False)
    print(f"\nBảng kết quả DML đã được lưu tại: {os.path.join(OUTPUT_DIR, 'dml_results_expanded.csv')}")
    
    # ════════════════════════════════════════════════════════════════════
    # III. KIỂM TRA ĐỘ ỔN ĐỊNH SEED SENSITIVITY (Point 5)
    # ════════════════════════════════════════════════════════════════════
    print("\n>>> III. KIỂM TRA ĐỘ ỔN ĐỊNH SEED SENSITIVITY (RandomForest)...")
    seeds = [42, 123, 456, 789, 2024]
    seed_results = []
    
    # Sử dụng full sample One-way
    for t_name in ['td_a', 'ibd_a']:
        df_sub = df.dropna(subset=[y_col, t_name] + x_cols).copy()
        grouped_mean = df_sub.groupby('ticker')[[y_col, t_name] + x_cols].transform('mean')
        df_dem = df_sub.copy()
        for col in [y_col, t_name] + x_cols:
            df_dem[col] = df_sub[col] - grouped_mean[col]
            
        time_dum = pd.get_dummies(df_sub['period'], prefix='time', drop_first=True).astype(float).values
        X_all = np.hstack([df_dem[x_cols].values, time_dum])
        Y_val = df_dem[y_col].values
        D_val = df_dem[t_name].values
        clusters_oneway = pd.Categorical(df_sub['ticker']).codes.reshape(-1, 1)
        
        dml_data = dml.DoubleMLData.from_arrays(x=X_all, y=Y_val, d=D_val, cluster_vars=clusters_oneway)
        
        for seed in seeds:
            rf_l = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=seed, n_jobs=-1)
            rf_m = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=seed, n_jobs=-1)
            dml_rf = dml.DoubleMLPLR(dml_data, ml_l=rf_l, ml_m=rf_m, n_folds=N_FOLDS, n_rep=N_REP)
            dml_rf.fit()
            ci = dml_rf.confint()
            seed_results.append({
                'Treatment': t_name, 'seed': seed,
                'theta': dml_rf.coef[0], 'se': dml_rf.se[0], 'pval': dml_rf.pval[0],
                'ci_low': ci.iloc[0, 0], 'ci_high': ci.iloc[0, 1]
            })
            print(f"  {t_name.upper()} | Seed {seed:>4d} | θ = {dml_rf.coef[0]:+.6f} | SE = {dml_rf.se[0]:.6f}")
            
    df_seeds = pd.DataFrame(seed_results)
    df_seeds.to_csv(os.path.join(OUTPUT_DIR, 'dml_seed_stability.csv'), index=False)
    
    # ════════════════════════════════════════════════════════════════════
    # PLOTTING convergence diagnostics
    # ════════════════════════════════════════════════════════════════════
    print("\nPloting figures...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#34495e']
    
    for row_idx, t_name in enumerate(['td_a', 'ibd_a']):
        # Left plot: Forest Plot comparing Sample Subgroups
        ax1 = axes[row_idx, 0]
        sub_df = df_results[(df_results['Treatment'] == t_name) & 
                            (df_results['Model_Specification'].str.contains('Full|Subgroup|Excluding'))]
        
        y_labels = []
        thetas = []
        ci_lows = []
        ci_highs = []
        
        for idx, row in sub_df.reset_index(drop=True).iterrows():
            y_labels.append(row['Model_Specification'])
            thetas.append(row['RF_theta'])
            ci_lows.append(row['RF_theta'] - 1.96 * row['RF_se'])
            ci_highs.append(row['RF_theta'] + 1.96 * row['RF_se'])
            
        for i, (n, t, lo, hi) in enumerate(zip(y_labels, thetas, ci_lows, ci_highs)):
            ax1.errorbar(t, i, xerr=[[t-lo], [hi-t]], fmt='o',
                        color=colors[i % len(colors)], markersize=8, capsize=5, linewidth=2)
            ax1.text(hi + 0.002, i, f'{t:.4f}', va='center', fontsize=9)
            
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_yticks(range(len(y_labels)))
        ax1.set_yticklabels(y_labels, fontsize=9)
        ax1.set_xlabel(f'θ ({t_name.upper()} → ROA)', fontsize=11)
        ax1.set_title(f'Forest Plot: {t_name.upper()} Specifications (95% CI)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Right plot: Seed sensitivity for RF
        ax2 = axes[row_idx, 1]
        seed_df = df_seeds[df_seeds['Treatment'] == t_name]
        
        for i, row in seed_df.reset_index(drop=True).iterrows():
            ax2.errorbar(row['theta'], i, xerr=[[row['theta']-row['ci_low']], [row['ci_high']-row['theta']]],
                        fmt='s', color='teal', markersize=8, capsize=5, linewidth=2)
            ax2.text(row['ci_high'] + 0.001, i, f"seed={int(row['seed'])}", va='center', fontsize=9)
            
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_yticks(range(len(seeds)))
        ax2.set_yticklabels([f'Seed {s}' for s in seeds], fontsize=10)
        ax2.set_xlabel(f'θ ({t_name.upper()})', fontsize=11)
        ax2.set_title(f'Seed Sensitivity: {t_name.upper()} (95% CI)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, 'figures', 'convergence_diagnostics_expanded.png')
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Biểu đồ chẩn đoán hội tụ đã được lưu: {plot_file}")

if __name__ == "__main__":
    main()
