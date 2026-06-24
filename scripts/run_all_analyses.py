"""
Script tổng hợp: Chạy tất cả phân tích bổ sung theo yêu cầu giảng viên
1. Scatter + LOWESS visualization
2. Baseline Models: Pooled OLS, Entity FE, Two-way FE
3. DML Convergence Diagnostics: theta per fold, per seed, CI
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import doubleml as dml

# ============================================================
# 0. LOAD & PREPARE DATA
# ============================================================
df = pd.read_csv("D:/draft 2/data/processed/master_panel_dataset.csv")

y_col = 'roa'
d_col = 'debt_to_assets'
x_cols = ['current_ratio', 'quick_ratio', 'asset_turnover', 'gross_margin', 'firm_age']

df_clean = df.dropna(subset=[y_col, d_col] + x_cols).copy()
for c in [y_col, d_col] + x_cols:
    df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
df_clean = df_clean.dropna(subset=[y_col, d_col] + x_cols)

print(f"=== DATA LOADED: {df_clean.shape[0]} observations, {df_clean['ticker'].nunique()} firms ===")
print()

# ============================================================
# 1. SCATTER + LOWESS VISUALIZATION
# ============================================================
print("=" * 60)
print("1. SCATTER + LOWESS PLOTS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Bằng chứng trực quan về tính phi tuyến\n(Scatter Plot + LOWESS Smoothing)', 
             fontsize=15, fontweight='bold', y=0.98)

plot_vars = [
    (d_col, 'Đòn bẩy tài chính (D/E)', axes[0, 0]),
    ('current_ratio', 'Khả năng thanh toán hiện thời', axes[0, 1]),
    ('gross_margin', 'Biên lợi nhuận gộp', axes[1, 0]),
    ('asset_turnover', 'Vòng quay tài sản', axes[1, 1]),
]

for var, label, ax in plot_vars:
    x_data = df_clean[var].values
    y_data = df_clean[y_col].values
    
    # Remove extreme outliers for visualization (keep 1st-99th percentile)
    mask = (x_data >= np.percentile(x_data, 1)) & (x_data <= np.percentile(x_data, 99))
    x_plot, y_plot = x_data[mask], y_data[mask]
    
    ax.scatter(x_plot, y_plot, alpha=0.15, s=8, color='steelblue', label='Quan sát')
    
    # LOWESS smoothing
    lowess_result = lowess(y_plot, x_plot, frac=0.4)
    ax.plot(lowess_result[:, 0], lowess_result[:, 1], color='red', linewidth=2.5, label='LOWESS')
    
    # OLS linear fit for comparison
    from numpy.polynomial import polynomial as P
    coefs = np.polyfit(x_plot, y_plot, 1)
    x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
    ax.plot(x_line, np.polyval(coefs, x_line), color='orange', linewidth=1.5, 
            linestyle='--', label='OLS tuyến tính')
    
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('ROA', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('D:/draft 2/results/figures/scatter_lowess_nonlinearity.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: scatter_lowess_nonlinearity.png")
print()

# ============================================================
# 2. BASELINE MODELS: Pooled OLS, Entity FE, Two-way FE
# ============================================================
print("=" * 60)
print("2. BASELINE MODELS")
print("=" * 60)

# --- Pooled OLS ---
X_ols = sm.add_constant(df_clean[[d_col] + x_cols])
y_ols = df_clean[y_col]
ols_model = sm.OLS(y_ols, X_ols).fit(cov_type='cluster', cov_kwds={'groups': df_clean['ticker']})

print("\n--- Pooled OLS (Clustered SE by firm) ---")
print(f"  θ(leverage) = {ols_model.params[d_col]:.6f}")
print(f"  SE          = {ols_model.bse[d_col]:.6f}")
print(f"  p-value     = {ols_model.pvalues[d_col]:.6f}")
ci_ols = ols_model.conf_int().loc[d_col]
print(f"  95% CI      = [{ci_ols[0]:.6f}, {ci_ols[1]:.6f}]")
print(f"  R²          = {ols_model.rsquared:.4f}")

# --- Entity Fixed Effects (Within Transformation) ---
vars_to_demean = [y_col, d_col] + x_cols
grouped_mean = df_clean.groupby('ticker')[vars_to_demean].transform('mean')
df_demeaned = df_clean.copy()
for col in vars_to_demean:
    df_demeaned[col] = df_clean[col] - grouped_mean[col]

X_fe = sm.add_constant(df_demeaned[[d_col] + x_cols])
y_fe = df_demeaned[y_col]
fe_model = sm.OLS(y_fe, X_fe).fit(cov_type='cluster', cov_kwds={'groups': df_clean['ticker']})

print("\n--- Entity Fixed Effects (Clustered SE by firm) ---")
print(f"  θ(leverage) = {fe_model.params[d_col]:.6f}")
print(f"  SE          = {fe_model.bse[d_col]:.6f}")
print(f"  p-value     = {fe_model.pvalues[d_col]:.6f}")
ci_fe = fe_model.conf_int().loc[d_col]
print(f"  95% CI      = [{ci_fe[0]:.6f}, {ci_fe[1]:.6f}]")
print(f"  R² (within) = {fe_model.rsquared:.4f}")

# --- Two-way Fixed Effects (Entity + Time) ---
time_dummies = pd.get_dummies(df_clean['period'], prefix='time', drop_first=True).astype(float)
# Demean both entity and time
df_twfe = df_demeaned.copy()
# Add time dummies (already orthogonal to entity FE after demeaning)
X_twfe_vars = [d_col] + x_cols
X_twfe_df = pd.concat([df_demeaned[X_twfe_vars].reset_index(drop=True), time_dummies.reset_index(drop=True)], axis=1)
X_twfe_df = X_twfe_df.apply(pd.to_numeric, errors='coerce').fillna(0)
X_twfe = sm.add_constant(X_twfe_df)
y_twfe = df_demeaned[y_col].reset_index(drop=True).astype(float)
groups_twfe = df_clean['ticker'].reset_index(drop=True)
twfe_model = sm.OLS(y_twfe, X_twfe).fit(cov_type='cluster', cov_kwds={'groups': groups_twfe})

print("\n--- Two-way FE: Entity + Time (Clustered SE by firm) ---")
print(f"  θ(leverage) = {twfe_model.params[d_col]:.6f}")
print(f"  SE          = {twfe_model.bse[d_col]:.6f}")
print(f"  p-value     = {twfe_model.pvalues[d_col]:.6f}")
ci_twfe = twfe_model.conf_int().loc[d_col]
print(f"  95% CI      = [{ci_twfe[0]:.6f}, {ci_twfe[1]:.6f}]")
print(f"  R² (within) = {twfe_model.rsquared:.4f}")

# ============================================================
# 3. DML CONVERGENCE DIAGNOSTICS
# ============================================================
print("\n" + "=" * 60)
print("3. DML CONVERGENCE DIAGNOSTICS")
print("=" * 60)

# Prepare DML data
X_financial = df_demeaned[x_cols].values
X_time = time_dummies.values.astype(float)
X_all = np.hstack([X_financial, X_time])

scaler = StandardScaler()
X_all[:, :len(x_cols)] = scaler.fit_transform(X_all[:, :len(x_cols)])

Y = df_demeaned[y_col].values
D = df_demeaned[d_col].values

# --- 3a. Theta per learner (with n_rep=1 to get per-fold thetas) ---
print("\n--- 3a. θ per Fold (5 folds, per learner) ---")

learner_configs = {
    'LassoCV': (LassoCV(cv=5, max_iter=10000, random_state=42), 
                LassoCV(cv=5, max_iter=10000, random_state=42)),
    'ElasticNet': (ElasticNetCV(cv=5, max_iter=10000, random_state=42),
                   ElasticNetCV(cv=5, max_iter=10000, random_state=42)),
    'RandomForest': (RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
                     RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)),
}

all_results = {}
for name, (ml_l, ml_m) in learner_configs.items():
    np.random.seed(42)
    dml_data = dml.DoubleMLData.from_arrays(x=X_all, y=Y, d=D)
    model = dml.DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5, n_rep=5)
    model.fit()
    
    # Get per-repetition thetas
    thetas_per_rep = model.all_coef.flatten()
    
    print(f"\n  {name}:")
    print(f"    Final θ = {model.coef[0]:.6f}, SE = {model.se[0]:.6f}, p = {model.pval[0]:.6f}")
    ci = model.confint()
    print(f"    95% CI  = [{ci.iloc[0, 0]:.6f}, {ci.iloc[0, 1]:.6f}]")
    print(f"    θ per rep (5 reps): {[f'{t:.6f}' for t in thetas_per_rep]}")
    print(f"    θ range: [{thetas_per_rep.min():.6f}, {thetas_per_rep.max():.6f}]")
    print(f"    θ std across reps: {thetas_per_rep.std():.6f}")
    
    all_results[name] = {
        'theta': model.coef[0],
        'se': model.se[0],
        'pval': model.pval[0],
        'ci_low': ci.iloc[0, 0],
        'ci_high': ci.iloc[0, 1],
        'thetas_per_rep': thetas_per_rep,
    }

# --- 3b. Theta across different seeds ---
print("\n--- 3b. θ Sensitivity to Random Seed (RandomForest) ---")
seeds = [42, 123, 456, 789, 2024]
theta_by_seed = []

for seed in seeds:
    np.random.seed(seed)
    dml_data = dml.DoubleMLData.from_arrays(x=X_all, y=Y, d=D)
    rf_l = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=seed)
    rf_m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=seed)
    model = dml.DoubleMLPLR(dml_data, ml_l=rf_l, ml_m=rf_m, n_folds=5, n_rep=5)
    model.fit()
    ci = model.confint()
    theta_by_seed.append({
        'seed': seed,
        'theta': model.coef[0],
        'se': model.se[0],
        'pval': model.pval[0],
        'ci_low': ci.iloc[0, 0],
        'ci_high': ci.iloc[0, 1],
    })
    print(f"  Seed {seed:>4d}: θ = {model.coef[0]:+.6f}, SE = {model.se[0]:.6f}, p = {model.pval[0]:.4f}, 95% CI = [{ci.iloc[0,0]:.6f}, {ci.iloc[0,1]:.6f}]")

df_seeds = pd.DataFrame(theta_by_seed)
print(f"\n  θ across seeds: mean = {df_seeds['theta'].mean():.6f}, std = {df_seeds['theta'].std():.6f}")
print(f"  All p-values > 0.05? {'YES' if (df_seeds['pval'] > 0.05).all() else 'NO'}")

# --- 3c. Convergence Forest Plot ---
print("\n--- 3c. Creating Convergence Forest Plot ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: theta by learner
ax1 = axes[0]
learner_names = list(all_results.keys())
thetas = [all_results[n]['theta'] for n in learner_names]
ci_lows = [all_results[n]['ci_low'] for n in learner_names]
ci_highs = [all_results[n]['ci_high'] for n in learner_names]

# Add baseline models
all_names = ['Pooled OLS', 'Entity FE', 'Two-way FE'] + [f'DML-{n}' for n in learner_names]
all_thetas = [ols_model.params[d_col], fe_model.params[d_col], twfe_model.params[d_col]] + thetas
all_ci_lo = [ci_ols[0], ci_fe[0], ci_twfe[0]] + ci_lows
all_ci_hi = [ci_ols[1], ci_fe[1], ci_twfe[1]] + ci_highs
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

y_pos = range(len(all_names))
for i, (name, theta, lo, hi) in enumerate(zip(all_names, all_thetas, all_ci_lo, all_ci_hi)):
    ax1.errorbar(theta, i, xerr=[[theta-lo], [hi-theta]], fmt='o', 
                color=colors[i], markersize=8, capsize=5, linewidth=2)
    ax1.text(hi + 0.002, i, f'θ={theta:.4f}', va='center', fontsize=9)

ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_yticks(list(y_pos))
ax1.set_yticklabels(all_names, fontsize=10)
ax1.set_xlabel('Hệ số θ (Tác động đòn bẩy → ROA)', fontsize=11)
ax1.set_title('So sánh θ: Baseline vs DML (95% CI)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# Right: theta by seed (RF only)
ax2 = axes[1]
for i, row in df_seeds.iterrows():
    ax2.errorbar(row['theta'], i, xerr=[[row['theta']-row['ci_low']], [row['ci_high']-row['theta']]], 
                fmt='s', color='teal', markersize=8, capsize=5, linewidth=2)
    ax2.text(row['ci_high'] + 0.001, i, f"seed={int(row['seed'])}", va='center', fontsize=9)

ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_yticks(range(len(seeds)))
ax2.set_yticklabels([f'Seed {s}' for s in seeds], fontsize=10)
ax2.set_xlabel('Hệ số θ (Random Forest)', fontsize=11)
ax2.set_title('Ổn định θ qua các Random Seeds (95% CI)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('D:/draft 2/results/figures/convergence_diagnostics.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: convergence_diagnostics.png")

# ============================================================
# 4. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 60)
print("4. SUMMARY TABLE FOR PAPER")
print("=" * 60)

summary = pd.DataFrame({
    'Model': all_names,
    'θ': [f'{t:.6f}' for t in all_thetas],
    'SE': [f'{ols_model.bse[d_col]:.6f}', f'{fe_model.bse[d_col]:.6f}', f'{twfe_model.bse[d_col]:.6f}'] + 
          [f'{all_results[n]["se"]:.6f}' for n in learner_names],
    'p-value': [f'{ols_model.pvalues[d_col]:.4f}', f'{fe_model.pvalues[d_col]:.4f}', f'{twfe_model.pvalues[d_col]:.4f}'] +
               [f'{all_results[n]["pval"]:.4f}' for n in learner_names],
    '95% CI': [f'[{lo:.4f}, {hi:.4f}]' for lo, hi in zip(all_ci_lo, all_ci_hi)],
})
print(summary.to_string(index=False))

print("\n\n=== ALL ANALYSES COMPLETED SUCCESSFULLY ===")
