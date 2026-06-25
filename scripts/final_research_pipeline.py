"""
MÔ HÌNH NGHIÊN CỨU DML: CẤU TRÚC VỐN & HIỆU QUẢ HOẠT ĐỘNG
============================================================
Kịch bản nghiên cứu hoàn chỉnh (Final Research Pipeline)
Đề tài: Tác động nhân quả của Đòn bẩy tài chính lên ROA (Doanh nghiệp Niêm yết VN)
Phương pháp: Double Machine Learning (DML-PLR) kiểm soát Fixed Effects
============================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import doubleml as dml

# Đảm bảo hiển thị đúng font Tiếng Việt trong console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Đường dẫn thư mục dữ liệu
DIR = r"D:\draft 2"
INPUT_FILE = os.path.join(DIR, "master_panel_dataset.csv")
OUTPUT_CSV = os.path.join(DIR, "dml_final_report.csv")
OUTPUT_PLOT = os.path.join(DIR, "dml_final_forest_plot.png")

def clean_and_prepare_data(file_path):
    """
    1. Đọc dữ liệu Master đã ghép nối từ các bước trước.
    2. Lọc bỏ các giá trị khuyết thiếu (NaN).
    3. Thực hiện phép biến đổi Fixed Effects (Entity-Demeaning).
    """
    df = pd.read_csv(file_path)
    
    y_col = 'roa'
    d_col = 'debt_to_assets'
    x_cols = ['current_ratio', 'quick_ratio', 'asset_turnover', 'gross_margin', 'firm_age']
    
    # Làm sạch NaN
    df_clean = df.dropna(subset=[y_col, d_col] + x_cols).copy()
    
    # Biến đổi Within-Transformation (Fixed Effects) để triệt tiêu đặc tính cố định của doanh nghiệp
    # Tất cả các biến số được trừ đi giá trị trung bình theo thời gian của chính doanh nghiệp đó
    vars_to_demean = [y_col, d_col] + x_cols
    grouped_mean = df_clean.groupby('ticker')[vars_to_demean].transform('mean')
    
    df_demeaned = df_clean.copy()
    for col in vars_to_demean:
        df_demeaned[col] = df_clean[col] - grouped_mean[col]
        
    return df_clean, df_demeaned, y_col, d_col, x_cols

def run_pipeline():
    print("="*70)
    print("  KÍCH HOẠT HỆ THỐNG NGHIÊN CỨU CASUAL ML (DML-PLR)")
    print("="*70)
    
    # 1. Chuẩn bị dữ liệu
    df_raw, df_demeaned, y_col, d_col, x_cols = clean_and_prepare_data(INPUT_FILE)
    print(f"  • Kích thước mẫu thực tế (N) : {df_raw.shape[0]} quan sát (quý-doanh nghiệp)")
    print(f"  • Số lượng doanh nghiệp (i)  : {df_raw['ticker'].nunique()} công ty")
    print(f"  • Các biến kiểm soát tài chính: {x_cols}")
    
    # 2. Xây dựng ma trận đặc trưng
    # Tạo biến giả thời gian (Time Fixed Effects) để khống chế các cú sốc vĩ mô toàn thị trường
    time_dummies = pd.get_dummies(df_raw['period'], prefix='time', drop_first=True)
    
    # Gộp biến kiểm soát tài chính đã demeaned và biến giả thời gian
    X_financial = df_demeaned[x_cols].values
    X_time = time_dummies.values.astype(float)
    X_all = np.hstack([X_financial, X_time])
    
    # Chuẩn hóa (Scale) các biến kiểm soát tài chính
    scaler = StandardScaler()
    X_all[:, :len(x_cols)] = scaler.fit_transform(X_all[:, :len(x_cols)])
    
    Y = df_demeaned[y_col].values
    D = df_demeaned[d_col].values
    
    # 3. Chạy các mô hình ước lượng
    results = []
    tickers = df_raw['ticker'].values
    
    # Mô hình 1: Pooled OLS (không FE)
    print("\n  • Đang ước lượng mô hình Baseline Pooled OLS (Clustered SE)...")
    Y_raw = df_raw[y_col].values
    D_raw = df_raw[d_col].values
    X_raw_financial = df_raw[x_cols].values
    scaler_raw = StandardScaler()
    X_raw_financial = scaler_raw.fit_transform(X_raw_financial)
    X_pooled = sm.add_constant(np.hstack([D_raw.reshape(-1, 1), X_raw_financial]))
    pooled_model = sm.OLS(Y_raw, X_pooled).fit(cov_type='cluster', cov_kwds={'groups': tickers})
    results.append({
        'Model': 'Pooled OLS (không FE)',
        'Coef': pooled_model.params[1],
        'SE': pooled_model.bse[1],
        'P-value': pooled_model.pvalues[1],
        'CI_lower': pooled_model.conf_int()[1][0],
        'CI_upper': pooled_model.conf_int()[1][1]
    })
    
    # Mô hình 2: Entity FE (FEM)
    print("  • Đang ước lượng mô hình Baseline Entity FE (FEM, Clustered SE)...")
    X_entity_fe = sm.add_constant(np.hstack([D.reshape(-1, 1), scaler.fit_transform(df_demeaned[x_cols].values)]))
    entity_fe_model = sm.OLS(Y, X_entity_fe).fit(cov_type='cluster', cov_kwds={'groups': tickers})
    results.append({
        'Model': 'Entity FE (FEM)',
        'Coef': entity_fe_model.params[1],
        'SE': entity_fe_model.bse[1],
        'P-value': entity_fe_model.pvalues[1],
        'CI_lower': entity_fe_model.conf_int()[1][0],
        'CI_upper': entity_fe_model.conf_int()[1][1]
    })
    
    # Mô hình 3: Two-way FE (Entity & Time FE)
    print("  • Đang ước lượng mô hình Baseline Two-way FE (Entity & Time, Clustered SE)...")
    X_twfe = sm.add_constant(np.hstack([D.reshape(-1, 1), X_all]))
    twfe_model = sm.OLS(Y, X_twfe).fit(cov_type='cluster', cov_kwds={'groups': tickers})
    results.append({
        'Model': 'Two-way FE (Entity & Time)',
        'Coef': twfe_model.params[1],
        'SE': twfe_model.bse[1],
        'P-value': twfe_model.pvalues[1],
        'CI_lower': twfe_model.conf_int()[1][0],
        'CI_upper': twfe_model.conf_int()[1][1]
    })
    
    # Đóng gói dữ liệu cho DoubleML
    dml_data = dml.DoubleMLData.from_arrays(x=X_all, y=Y, d=D)
    
    # Mô hình 4: DML-PLR LassoCV
    print("  • Đang ước lượng mô hình DML-PLR (LassoCV) với 5-Fold Cross-fitting...")
    lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
    np.random.seed(42)
    dml_lasso = dml.DoubleMLPLR(dml_data, ml_l=lasso, ml_m=lasso, n_folds=5, n_rep=5)
    dml_lasso.fit()
    ci_l = dml_lasso.confint().values[0]
    results.append({
        'Model': 'DML-PLR (LassoCV)',
        'Coef': dml_lasso.coef[0],
        'SE': dml_lasso.se[0],
        'P-value': dml_lasso.pval[0],
        'CI_lower': ci_l[0],
        'CI_upper': ci_l[1]
    })
    
    # Mô hình 5: DML-PLR ElasticNetCV
    print("  • Đang ước lượng mô hình DML-PLR (ElasticNetCV) với 5-Fold Cross-fitting...")
    enet = ElasticNetCV(cv=5, max_iter=10000, random_state=42)
    np.random.seed(42)
    dml_enet = dml.DoubleMLPLR(dml_data, ml_l=enet, ml_m=enet, n_folds=5, n_rep=5)
    dml_enet.fit()
    ci_e = dml_enet.confint().values[0]
    results.append({
        'Model': 'DML-PLR (ElasticNetCV)',
        'Coef': dml_enet.coef[0],
        'SE': dml_enet.se[0],
        'P-value': dml_enet.pval[0],
        'CI_lower': ci_e[0],
        'CI_upper': ci_e[1]
    })
    
    # Mô hình 6: DML-PLR RandomForest
    print("  • Đang ước lượng mô hình DML-PLR (RandomForest) với 5-Fold Cross-fitting...")
    from sklearn.ensemble import RandomForestRegressor
    rf_l = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    np.random.seed(42)
    dml_rf = dml.DoubleMLPLR(dml_data, ml_l=rf_l, ml_m=rf_m, n_folds=5, n_rep=5)
    dml_rf.fit()
    ci_rf = dml_rf.confint().values[0]
    results.append({
        'Model': 'DML-PLR (RandomForest)',
        'Coef': dml_rf.coef[0],
        'SE': dml_rf.se[0],
        'P-value': dml_rf.pval[0],
        'CI_lower': ci_rf[0],
        'CI_upper': ci_rf[1]
    })
    
    # 4. Hiển thị bảng tổng hợp báo cáo học thuật
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*70)
    print("  BẢNG KẾT QUẢ ƯỚC LƯỢNG NHÂN QUẢ (θ)")
    print("="*70)
    print(df_results[['Model', 'Coef', 'SE', 'P-value']].to_string(index=False))
    print("="*70)
    
    # 5. Vẽ đồ thị Forest Plot chuyên nghiệp để chèn vào slide báo cáo
    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    models = df_results['Model'].tolist()
    coefs = df_results['Coef'].tolist()
    errors = [df_results['Coef'] - df_results['CI_lower'], df_results['CI_upper'] - df_results['Coef']]
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    for i, (model_name, coef, lo, hi) in enumerate(zip(models, coefs, df_results['CI_lower'], df_results['CI_upper'])):
        plt.errorbar(coef, i, xerr=[[coef - lo], [hi - coef]], fmt='o', color=colors[i % len(colors)],
                     elinewidth=2, capsize=6, markersize=8)
        plt.text(hi + 0.002, i, f'θ={coef:.4f}', va='center', fontsize=9, fontweight='bold')
    
    plt.yticks(range(len(models)), models, fontsize=10, fontweight='bold')
    plt.axvline(0, color='black', linestyle='--', linewidth=1.2)
    plt.title("So Sánh Hệ Số Tác Động Nhân Quả (θ) Giữa Các Mô Hình", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Hệ số hồi quy (θ)", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    
    print(f"  ✔ Đã xuất bảng báo cáo: {OUTPUT_CSV}")
    print(f"  ✔ Đã xuất đồ thị Forest Plot: {OUTPUT_PLOT}")
    print("="*70)

if __name__ == "__main__":
    run_pipeline()
