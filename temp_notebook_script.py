

from google.colab import files
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Vui lòng upload file: master_panel_all_firms.csv")
uploaded = files.upload()

filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
print(f"\n✅ Đã tải thành công: {filename}")
print(f"   Số DN: {df['ticker'].nunique()}")
print(f"   Tổng obs: {len(df)}")

import statsmodels.api as sm
import time
import os

x_cols = ['current_ratio', 'quick_ratio', 'asset_turnover', 'gross_margin',
          'firm_size', 'tangibility', 'data_tenure']

def run_regressions(df, y_col, d_col, x_cols):
    all_vars = [y_col, d_col] + x_cols
    df_clean = df.dropna(subset=all_vars).copy()
    results = []

    # 1. Pooled OLS
    X_ols = sm.add_constant(df_clean[[d_col] + x_cols])
    y_ols = df_clean[y_col]
    ols_model = sm.OLS(y_ols, X_ols).fit(cov_type='cluster', cov_kwds={'groups': df_clean['ticker']})
    ci = ols_model.conf_int().loc[d_col]
    results.append({'Y_variable': y_col, 'Treatment': d_col, 'Model': 'Pooled OLS',
        'theta': ols_model.params[d_col], 'se': ols_model.bse[d_col],
        'pval': ols_model.pvalues[d_col], 'ci_low': ci[0], 'ci_high': ci[1],
        'r2': ols_model.rsquared, 'n_obs': int(ols_model.nobs)})

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
    results.append({'Y_variable': y_col, 'Treatment': d_col, 'Model': 'Entity FE',
        'theta': fe_model.params[d_col], 'se': fe_model.bse[d_col],
        'pval': fe_model.pvalues[d_col], 'ci_low': ci[0], 'ci_high': ci[1],
        'r2': fe_model.rsquared, 'n_obs': int(fe_model.nobs)})

    # 3. Two-way FE
    time_dummies = pd.get_dummies(df_clean['period'], prefix='time', drop_first=True).astype(float)
    X_twfe_vars = df_demeaned[[d_col] + x_cols].reset_index(drop=True)
    X_twfe_df = pd.concat([X_twfe_vars, time_dummies.reset_index(drop=True)], axis=1)
    y_twfe = df_demeaned[y_col].reset_index(drop=True).astype(float)
    groups_twfe = df_clean['ticker'].reset_index(drop=True)
    twfe_model = sm.OLS(y_twfe, X_twfe_df).fit(cov_type='cluster', cov_kwds={'groups': groups_twfe})
    ci = twfe_model.conf_int().loc[d_col]
    results.append({'Y_variable': y_col, 'Treatment': d_col, 'Model': 'Two-way FE',
        'theta': twfe_model.params[d_col], 'se': twfe_model.bse[d_col],
        'pval': twfe_model.pvalues[d_col], 'ci_low': ci[0], 'ci_high': ci[1],
        'r2': twfe_model.rsquared, 'n_obs': int(twfe_model.nobs)})

    print(f"  ✅ {y_col} ~ {d_col}: Two-way FE θ = {twfe_model.params[d_col]:.6f}, p = {twfe_model.pvalues[d_col]:.6f}")
    return pd.DataFrame(results)

print("="*70)
print("  BASELINE MODELS (OLS, Entity FE, Two-way FE)")
print("="*70)

t0 = time.time()
all_baseline = []
for y_var in ['roa', 'operating_roa']:
    for d_var in ['td_a', 'ibd_a']:
        all_baseline.append(run_regressions(df, y_var, d_var, x_cols))

df_baseline = pd.concat(all_baseline, ignore_index=True)
df_baseline.to_csv('baseline_results.csv', index=False)
print(f"\n⏱️ Baseline hoàn thành trong {time.time()-t0:.1f}s")
print(f"📄 Đã lưu: baseline_results.csv")
df_baseline

import doubleml as dml
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_FOLDS = 5
N_REP = 5

def get_oof_r2(model, key, actual):
    preds = model.predictions[key][:, :, 0]
    r2_list = []
    for r in range(preds.shape[1]):
        u = ((actual - preds[:, r]) ** 2).sum()
        v = ((actual - actual.mean()) ** 2).sum()
        r2_list.append(1 - u / v)
    return np.mean(r2_list)

def run_dml_plr_spec(dml_data, y_actual, d_actual, rf_estimators=100):
    # 1. LassoCV
    lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=10000, random_state=42, n_jobs=-1))
    dml_lasso = dml.DoubleMLPLR(dml_data, ml_l=lasso, ml_m=lasso, n_folds=N_FOLDS, n_rep=N_REP)
    dml_lasso.fit()

    # 2. ElasticNetCV
    enet = make_pipeline(StandardScaler(), ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], max_iter=10000, random_state=42, n_jobs=-1))
    dml_enet = dml.DoubleMLPLR(dml_data, ml_l=enet, ml_m=enet, n_folds=N_FOLDS, n_rep=N_REP)
    dml_enet.fit(store_models=True)

    enet_l1_ratios = []
    for r in range(N_REP):
        for f in range(N_FOLDS):
            est = dml_enet.models['ml_l']['d'][r][f]
            enet_l1_ratios.append(est.named_steps['elasticnetcv'].l1_ratio_)
    mean_l1 = np.mean(enet_l1_ratios)

    # 3. RandomForest (n_jobs=-1 tối đa hóa tài nguyên trên Colab)
    rf_l = RandomForestRegressor(n_estimators=rf_estimators, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_m = RandomForestRegressor(n_estimators=rf_estimators, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
    dml_rf = dml.DoubleMLPLR(dml_data, ml_l=rf_l, ml_m=rf_m, n_folds=N_FOLDS, n_rep=N_REP)
    dml_rf.fit()

    r2_l_lasso = get_oof_r2(dml_lasso, 'ml_l', y_actual)
    r2_m_lasso = get_oof_r2(dml_lasso, 'ml_m', d_actual)
    r2_l_rf = get_oof_r2(dml_rf, 'ml_l', y_actual)
    r2_m_rf = get_oof_r2(dml_rf, 'ml_m', d_actual)

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

def prep(df_sub, y_col, t_name, x_cols):
    gm = df_sub.groupby('ticker')[[y_col,t_name]+x_cols].transform('mean')
    dd = df_sub.copy()
    for c in [y_col,t_name]+x_cols: dd[c] = df_sub[c] - gm[c]
    td = pd.get_dummies(df_sub['period'], prefix='t', drop_first=True).astype(float).values
    X = np.hstack([dd[x_cols].values, td])
    return X, dd[y_col].values, dd[t_name].values, pd.Categorical(df_sub['ticker']).codes, pd.Categorical(df_sub['period']).codes

print("="*80)
print("  DML-PLR PIPELINE (18 mô hình + Seed Stability)")
print("="*80)

y_col = 'roa'
results = []
t0 = time.time()
model_count = 0

for t_name in ['td_a', 'ibd_a']:
    ds = df.dropna(subset=[y_col, t_name]+x_cols).copy()
    X, Y, D, tc, pc = prep(ds, y_col, t_name, x_cols)
    c1 = tc.reshape(-1,1)
    c2 = np.column_stack([tc, pc])

    # 1. One-way
    print(f"\n▶ [{t_name.upper()}] Full Sample (One-way Cluster)...")
    dml_data_1w = dml.DoubleMLData.from_arrays(x=X, y=Y, d=D, cluster_vars=c1)
    res = run_dml_plr_spec(dml_data_1w, Y, D)
    res.update({'Treatment': t_name, 'Model_Specification': 'Full Sample (One-way Cluster)', 'Y_variable': y_col, 'N_obs': len(Y)})
    results.append(res); model_count += 1
    print(f"  ✅ RF θ = {res['RF_theta']:.6f}, p = {res['RF_pval']:.4f}")

    # 2. Two-way
    print(f"▶ [{t_name.upper()}] Full Sample (Two-way Cluster)...")
    dml_data_2w = dml.DoubleMLData.from_arrays(x=X, y=Y, d=D, cluster_vars=c2)
    res = run_dml_plr_spec(dml_data_2w, Y, D)
    res.update({'Treatment': t_name, 'Model_Specification': 'Full Sample (Two-way Cluster)', 'Y_variable': y_col, 'N_obs': len(Y)})
    results.append(res); model_count += 1
    print(f"  ✅ RF θ = {res['RF_theta']:.6f}, p = {res['RF_pval']:.4f}")

    # 3. Lagged
    tl = t_name + '_lag'
    dl = df.dropna(subset=[y_col, tl]+x_cols).copy()
    Xl, Yl, Dl, tcl, _ = prep(dl, y_col, tl, x_cols)
    print(f"▶ [{t_name.upper()}] Lagged Treatment...")
    dml_data_lag = dml.DoubleMLData.from_arrays(x=Xl, y=Yl, d=Dl, cluster_vars=tcl.reshape(-1,1))
    res = run_dml_plr_spec(dml_data_lag, Yl, Dl)
    res.update({'Treatment': tl, 'Model_Specification': 'Lagged Treatment', 'Y_variable': y_col, 'N_obs': len(Yl)})
    results.append(res); model_count += 1
    print(f"  ✅ RF θ = {res['RF_theta']:.6f}, p = {res['RF_pval']:.4f}")

    # 4. Placebo (Set seed 42 để đảm bảo khả năng tái lập)
    print(f"▶ [{t_name.upper()}] Placebo (Permuted D)...")
    np.random.seed(42)
    D_placebo = np.random.permutation(D)
    dml_data_pl = dml.DoubleMLData.from_arrays(x=X, y=Y, d=D_placebo, cluster_vars=c1)
    res = run_dml_plr_spec(dml_data_pl, Y, D_placebo)
    res.update({'Treatment': t_name+'_placebo', 'Model_Specification': 'Placebo (Permuted D)', 'Y_variable': y_col, 'N_obs': len(Y)})
    results.append(res); model_count += 1
    print(f"  ✅ RF θ = {res['RF_theta']:.6f}, p = {res['RF_pval']:.4f}")

    # 5. Excluding UPCoM
    dn = df[df['exchange'].isin(['HOSE','HNX'])].dropna(subset=[y_col,t_name]+x_cols).copy()
    Xn, Yn, Dn, tcn, _ = prep(dn, y_col, t_name, x_cols)
    print(f"▶ [{t_name.upper()}] Excluding UPCoM (N={len(Yn)})...")
    dml_data_nu = dml.DoubleMLData.from_arrays(x=Xn, y=Yn, d=Dn, cluster_vars=tcn.reshape(-1,1))
    res = run_dml_plr_spec(dml_data_nu, Yn, Dn)
    res.update({'Treatment': t_name, 'Model_Specification': 'Excluding UPCoM (HOSE+HNX only)', 'Y_variable': y_col, 'N_obs': len(Yn)})
    results.append(res); model_count += 1
    print(f"  ✅ RF θ = {res['RF_theta']:.6f}, p = {res['RF_pval']:.4f}")

    # 6. Operating ROA
    yo = 'operating_roa'
    do = df.dropna(subset=[yo,t_name]+x_cols).copy()
    Xo, Yo, Do, tco, _ = prep(do, yo, t_name, x_cols)
    print(f"▶ [{t_name.upper()}] Operating ROA (EBIT/Assets)...")
    dml_data_op = dml.DoubleMLData.from_arrays(x=Xo, y=Yo, d=Do, cluster_vars=tco.reshape(-1,1))
    res = run_dml_plr_spec(dml_data_op, Yo, Do)
    res.update({'Treatment': t_name, 'Model_Specification': 'Operating ROA (EBIT/Assets)', 'Y_variable': yo, 'N_obs': len(Yo)})
    results.append(res); model_count += 1
    print(f"  ✅ RF θ = {res['RF_theta']:.6f}, p = {res['RF_pval']:.4f}")

    # 7-8. HTE Subgroups
    for sn, mk in [('Large-Cap (Top 100)', df['large_cap']==1), ('Mid/Small-Cap', df['large_cap']==0)]:
        dg = df[mk].dropna(subset=[y_col,t_name]+x_cols).copy()
        Xg, Yg, Dg, tcg, _ = prep(dg, y_col, t_name, x_cols)
        print(f"▶ [{t_name.upper()}] Subgroup: {sn} (N={len(Yg)})...")
        dml_data_sg = dml.DoubleMLData.from_arrays(x=Xg, y=Yg, d=Dg, cluster_vars=tcg.reshape(-1,1))
        res = run_dml_plr_spec(dml_data_sg, Yg, Dg)
        res.update({'Treatment': t_name, 'Model_Specification': f'Subgroup: {sn}', 'Y_variable': y_col, 'N_obs': len(Yg)})
        results.append(res); model_count += 1
        print(f"  ✅ RF θ = {res['RF_theta']:.6f}, p = {res['RF_pval']:.4f}")

elapsed = time.time() - t0
print(f"\n⏱️ Hoàn thành {model_count} mô hình chính trong {elapsed:.0f}s ({elapsed/60:.1f} phút)")

print("\n" + "="*70)
print("  QUADRATIC DML (D + D²)")
print("="*70)

for t_name in ['td_a', 'ibd_a']:
    tsq = t_name + '_sq'
    dq = df.dropna(subset=[y_col, t_name, tsq] + x_cols).copy()

    gm = dq.groupby('ticker')[[y_col, t_name, tsq] + x_cols].transform('mean')
    dd = dq.copy()
    for c in [y_col, t_name, tsq] + x_cols: 
        dd[c] = dq[c] - gm[c]

    td = pd.get_dummies(dq['period'], prefix='t', drop_first=True).astype(float).values
    Xq = np.hstack([dd[x_cols].values, td])
    Yq = dd[y_col].values
    Dq = dd[[t_name, tsq]].values
    cq = pd.Categorical(dq['ticker']).codes.reshape(-1, 1)

    rf_l = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_m = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
    mq = dml.DoubleMLPLR(dml.DoubleMLData.from_arrays(x=Xq, y=Yq, d=Dq, cluster_vars=cq), ml_l=rf_l, ml_m=rf_m, n_folds=N_FOLDS, n_rep=N_REP)
    mq.fit()

    print(f"  {t_name.upper()}: Linear θ = {mq.coef[0]:+.6f} (p={mq.pval[0]:.4f}), Quadratic θ = {mq.coef[1]:+.6f} (p={mq.pval[1]:.4f})")

    results.append({'Treatment': t_name, 'Model_Specification': 'Quadratic DML (Linear)', 'Y_variable': y_col,
        'RF_theta': mq.coef[0], 'RF_se': mq.se[0], 'RF_pval': mq.pval[0],
        'Lasso_theta': np.nan, 'Lasso_se': np.nan, 'Lasso_pval': np.nan,
        'Enet_theta': np.nan, 'Enet_se': np.nan, 'Enet_pval': np.nan, 'N_obs': len(Yq)})
    results.append({'Treatment': tsq, 'Model_Specification': 'Quadratic DML (Quadratic)', 'Y_variable': y_col,
        'RF_theta': mq.coef[1], 'RF_se': mq.se[1], 'RF_pval': mq.pval[1],
        'Lasso_theta': np.nan, 'Lasso_se': np.nan, 'Lasso_pval': np.nan,
        'Enet_theta': np.nan, 'Enet_se': np.nan, 'Enet_pval': np.nan, 'N_obs': len(Yq)})

# Lưu kết quả DML
df_results = pd.DataFrame(results)
df_results.to_csv('dml_results_expanded.csv', index=False)
print(f"\n📄 Đã lưu: dml_results_expanded.csv ({len(df_results)} mô hình)")

print("\n" + "="*70)
print("  SEED STABILITY TEST (RandomForest, 5 seeds)")
print("="*70)

seeds = [42, 123, 456, 789, 2024]
seed_results = []

for t_name in ['td_a', 'ibd_a']:
    ds = df.dropna(subset=[y_col,t_name]+x_cols).copy()
    X, Y, D, tc, _ = prep(ds, y_col, t_name, x_cols)
    dml_data = dml.DoubleMLData.from_arrays(x=X, y=Y, d=D, cluster_vars=tc.reshape(-1,1))

    for seed in seeds:
        rl = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=seed, n_jobs=-1)
        rm = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=seed, n_jobs=-1)
        m = dml.DoubleMLPLR(dml_data, ml_l=rl, ml_m=rm, n_folds=N_FOLDS, n_rep=N_REP)
        m.fit()
        ci = m.confint()
        seed_results.append({
            'Treatment': t_name, 'seed': seed,
            'theta': m.coef[0], 'se': m.se[0], 'pval': m.pval[0],
            'ci_low': ci.iloc[0, 0], 'ci_high': ci.iloc[0, 1]
        })
        print(f"  {t_name.upper()} | Seed {seed:>4d} | θ = {m.coef[0]:+.6f} | SE = {m.se[0]:.6f} | p = {m.pval[0]:.4f}")

df_seeds = pd.DataFrame(seed_results)
df_seeds.to_csv('dml_seed_stability.csv', index=False)
print(f"\n📄 Đã lưu: dml_seed_stability.csv")

total_time = time.time() - t0
print(f"\n🏁 TỔNG THỜI GIAN CHẠY: {total_time:.0f}s ({total_time/60:.1f} phút)")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#34495e']

for row_idx, t_name in enumerate(['td_a', 'ibd_a']):
    # Left: Forest Plot
    ax1 = axes[row_idx, 0]
    sub_df = df_results[(df_results['Treatment'] == t_name) &
                        (df_results['Model_Specification'].str.contains('Full|Subgroup|Excluding'))]
    y_labels, thetas, ci_lows, ci_highs = [], [], [], []
    for _, row in sub_df.reset_index(drop=True).iterrows():
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
    ax1.set_title(f'Forest Plot: {t_name.upper()} (95% CI)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Right: Seed Sensitivity
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
plt.savefig('convergence_diagnostics_expanded.png', dpi=200, bbox_inches='tight')
plt.show()
print("📄 Đã lưu: convergence_diagnostics_expanded.png")

from google.colab import files

print("Đang tự động chuẩn bị tải xuống các file kết quả...")
files.download('baseline_results.csv')
files.download('dml_results_expanded.csv')
files.download('dml_seed_stability.csv')
files.download('convergence_diagnostics_expanded.png')
print("\n✅ Đã tải xuống thành công 4 file kết quả. Hãy copy chúng vào thư mục D:/draft 2/results/ trên máy của bạn!")

