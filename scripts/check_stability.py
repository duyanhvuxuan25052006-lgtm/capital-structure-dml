import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import doubleml as dml
import os
import sys

# Đảm bảo hiển thị đúng font Tiếng Việt trong console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Đường dẫn thư mục dữ liệu
DIR = r"D:\draft 2"
INPUT_FILE = os.path.join(DIR, "master_panel_dataset.csv")
OUTPUT_CSV = os.path.join(DIR, "results", "rf_stability_check.csv")

# 1. Load and prepare data
print("=== ĐANG TẢI DỮ LIỆU ĐỂ KIỂM TRA ĐỘ ỔN ĐỊNH SEED ===")
df = pd.read_csv(INPUT_FILE)
y_col = 'roa'
d_col = 'debt_to_assets'
x_cols = ['current_ratio', 'quick_ratio', 'asset_turnover', 'gross_margin', 'firm_age']

df_clean = df.dropna(subset=[y_col, d_col] + x_cols).copy()
vars_to_demean = [y_col, d_col] + x_cols
grouped_mean = df_clean.groupby('ticker')[vars_to_demean].transform('mean')
df_demeaned = df_clean.copy()
for col in vars_to_demean:
    df_demeaned[col] = df_clean[col] - grouped_mean[col]

time_dummies = pd.get_dummies(df_clean['period'], prefix='time', drop_first=True)
X_financial = df_demeaned[x_cols].values
X_time = time_dummies.values.astype(float)
X_all = np.hstack([X_financial, X_time])

scaler = StandardScaler()
X_all[:, :len(x_cols)] = scaler.fit_transform(X_all[:, :len(x_cols)])

Y = df_demeaned[y_col].values
D = df_demeaned[d_col].values

dml_data = dml.DoubleMLData.from_arrays(x=X_all, y=Y, d=D)

# 2. Run RandomForest DML over 30 seeds
stability_results = []

print("\n--- Đang chạy mô hình DML-RandomForest 30 lần với các Seed từ 1 đến 30 ---")
for seed in range(1, 31):
    # Cố định hạt ngẫu nhiên trước khi khởi tạo dml để khóa fold-splitting
    np.random.seed(seed)
    rf_l = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=seed)
    rf_m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=seed)
    dml_rf = dml.DoubleMLPLR(dml_data, ml_l=rf_l, ml_m=rf_m, n_folds=5, n_rep=5)
    dml_rf.fit()
    
    theta = dml_rf.coef[0]
    se = dml_rf.se[0]
    pval = dml_rf.pval[0]
    ci = dml_rf.confint().values[0]
    
    stability_results.append({
        'seed': seed,
        'theta': theta,
        'se': se,
        'pvalue': pval,
        'ci_lower': ci[0],
        'ci_upper': ci[1]
    })
    print(f"  • Seed {seed:2d}: theta = {theta:+.6f}, p-value = {pval:.4f}")

df_stab = pd.DataFrame(stability_results)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_stab.to_csv(OUTPUT_CSV, index=False)
print(f"\n✔ Đã lưu kết quả kiểm định độ ổn định vào: {OUTPUT_CSV}")

# 3. Tính toán các thống kê mô tả độ ổn định
mean_theta = df_stab['theta'].mean()
std_theta = df_stab['theta'].std()
min_theta = df_stab['theta'].min()
max_theta = df_stab['theta'].max()

min_pval = df_stab['pvalue'].min()
max_pval = df_stab['pvalue'].max()

significant_runs = (df_stab['pvalue'] < 0.05).sum()
pct_significant = (significant_runs / 30) * 100

print("\n" + "="*50)
print("  KẾT QUẢ PHÂN TÍCH ĐỘ ỔN ĐỊNH SENSITIVITY (30 SEEDS)")
print("="*50)
print(f"  • θ trung bình (Mean θ)       : {mean_theta:+.6f}")
print(f"  • Độ lệch chuẩn θ (SD)        : {std_theta:.6f}")
print(f"  • Khoảng θ (Min - Max)        : [{min_theta:+.6f}, {max_theta:+.6f}]")
print(f"  • Khoảng p-value (Min - Max)  : [{min_pval:.4f}, {max_pval:.4f}]")
print(f"  • Tỷ lệ có ý nghĩa (p < 0.05) : {significant_runs}/30 ({pct_significant:.1f}%)")
print("="*50)
