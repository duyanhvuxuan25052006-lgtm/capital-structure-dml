import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import doubleml as dml

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DIR = r"D:\draft 2"
INPUT_FILE = os.path.join(DIR, "master_panel_dataset.csv")
OUTPUT_CSV = os.path.join(DIR, "dml_results.csv")
OUTPUT_PLOT = os.path.join(DIR, "dml_coefficient_plot.png")

# Các cấu hình mô hình ML
def get_lasso():
    return LassoCV(cv=5, max_iter=10000, random_state=42)

def get_enet():
    return ElasticNetCV(cv=5, max_iter=10000, random_state=42)

try:
    print("1. Đang tải và làm sạch dữ liệu Master...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_FILE}")
        
    df = pd.read_csv(INPUT_FILE)
    print(f"Kích thước ban đầu: {df.shape}")
    
    # Lựa chọn các biến số quan trọng trong mô hình
    # Y = roa (Hiệu quả hoạt động)
    # D = debt_to_equity (Cấu trúc vốn)
    # X = các biến kiểm soát cấp doanh nghiệp
    y_col = 'roa'
    d_col = 'debt_to_equity'
    x_cols = ['current_ratio', 'quick_ratio', 'asset_turnover', 'gross_margin', 'net_margin', 'firm_age']
    
    # Loại bỏ các dòng bị khuyết thiếu (NaN) trong các biến chính
    df_clean = df.dropna(subset=[y_col, d_col] + x_cols).copy()
    print(f"Kích thước sau khi làm sạch NaN: {df_clean.shape}")
    
    # ══════════════════════════════════════════════════
    # BƯỚC 2: BIẾN ĐỔI FIXED EFFECTS (ENTITY-DEMEANING)
    # ══════════════════════════════════════════════════
    # Đây là điểm mấu chốt để kiểm soát các đặc điểm cố định không quan sát được của doanh nghiệp
    # (như văn hóa doanh nghiệp, tài năng CEO) trong dữ liệu bảng.
    print("\n2. Tiến hành biến đổi Fixed Effects (Entity-Demeaning)...")
    
    # Danh sách các biến cần trừ đi giá trị trung bình theo doanh nghiệp
    vars_to_demean = [y_col, d_col] + x_cols
    
    # Tính giá trị trung bình theo từng công ty (ticker)
    grouped_mean = df_clean.groupby('ticker')[vars_to_demean].transform('mean')
    
    # Trừ đi giá trị trung bình (Within-transformation)
    df_demeaned = df_clean.copy()
    for col in vars_to_demean:
        df_demeaned[col] = df_clean[col] - grouped_mean[col]
        
    # ══════════════════════════════════════════════════
    # BƯỚC 3: TẠO BIẾN GIẢ THỜI GIAN VÀ NGÀNH
    # ══════════════════════════════════════════════════
    print("3. Khởi tạo các biến giả cố định thời gian (Time Fixed Effects)...")
    # Tạo biến giả cho cột kỳ báo cáo (period) để kiểm soát các cú sốc thời gian
    time_dummies = pd.get_dummies(df_clean['period'], prefix='time', drop_first=True)
    
    # Gom tất cả các biến kiểm soát X lại
    # X_sc chứa các biến tài chính đã de-meaned + các biến giả thời gian
    X_financial = df_demeaned[x_cols].values
    X_time = time_dummies.values.astype(float)
    X_all = np.hstack([X_financial, X_time])
    
    # Chuẩn hóa (Scale) các biến kiểm soát tài chính
    scaler = StandardScaler()
    X_all[:, :len(x_cols)] = scaler.fit_transform(X_all[:, :len(x_cols)])
    
    Y = df_demeaned[y_col].values
    D = df_demeaned[d_col].values
    
    # ══════════════════════════════════════════════════
    # BƯỚC 4: ƯỚC LƯỢNG MÔ HÌNH DOUBLE MACHINE LEARNING
    # ══════════════════════════════════════════════════
    print("\n4. Đang chạy mô hình Double Machine Learning (DML-PLR)...")
    # Đóng gói dữ liệu chuẩn DoubleML
    dml_data = dml.DoubleMLData.from_arrays(x=X_all, y=Y, d=D)
    
    results = []
    
    # 4.1 Ước lượng bằng thuật toán LassoCV
    print("  • Đang huấn luyện mô hình bằng LassoCV (5-Fold Cross-fitting)...")
    dml_plr_lasso = dml.DoubleMLPLR(dml_data, ml_l=get_lasso(), ml_m=get_lasso(), n_folds=5, n_rep=5)
    dml_plr_lasso.fit()
    
    coef_l = dml_plr_lasso.coef[0]
    se_l = dml_plr_lasso.se[0]
    pval_l = dml_plr_lasso.pval[0]
    ci_l = dml_plr_lasso.confint().values[0]
    results.append({
        'Method': 'DML-PLR (LassoCV)', 'Coefficient': coef_l, 'Std_Error': se_l,
        'p-value': pval_l, 'CI_lower': ci_l[0], 'CI_upper': ci_l[1]
    })
    
    # 4.2 Ước lượng bằng thuật toán ElasticNetCV (Kiểm định tính vững - Robustness Check)
    print("  • Đang huấn luyện mô hình bằng ElasticNetCV (5-Fold Cross-fitting)...")
    dml_plr_enet = dml.DoubleMLPLR(dml_data, ml_l=get_enet(), ml_m=get_enet(), n_folds=5, n_rep=5)
    dml_plr_enet.fit()
    
    coef_e = dml_plr_enet.coef[0]
    se_e = dml_plr_enet.se[0]
    pval_e = dml_plr_enet.pval[0]
    ci_e = dml_plr_enet.confint().values[0]
    results.append({
        'Method': 'DML-PLR (ElasticNetCV)', 'Coefficient': coef_e, 'Std_Error': se_e,
        'p-value': pval_e, 'CI_lower': ci_e[0], 'CI_upper': ci_e[1]
    })
    
    # 4.3 Ước lượng Baseline OLS truyền thống để so sánh chênh lệch
    print("  • Đang chạy mô hình OLS Baseline để so sánh...")
    import statsmodels.api as sm
    X_ols = sm.add_constant(np.hstack([D.reshape(-1, 1), X_all]))
    ols_model = sm.OLS(Y, X_ols).fit(cov_type='HC3')
    coef_o = ols_model.params[1]
    se_o = ols_model.bse[1]
    pval_o = ols_model.pvalues[1]
    ci_o = ols_model.conf_int()[1]
    results.append({
        'Method': 'Traditional OLS (Fixed Effects)', 'Coefficient': coef_o, 'Std_Error': se_o,
        'p-value': pval_o, 'CI_lower': ci_o[0], 'CI_upper': ci_o[1]
    })
    
    # Lưu kết quả ra file CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    
    # ══════════════════════════════════════════════════
    # BƯỚC 5: HIỂN THỊ KẾT QUẢ & VẼ BIỂU ĐỒ HỆ SỐ
    # ══════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  KẾT QUẢ HỒI QUY CẦU TRÚC VỐN LÊN ROA")
    print("="*60)
    for res in results:
        sig = "***" if res['p-value'] < 0.01 else ("**" if res['p-value'] < 0.05 else ("*" if res['p-value'] < 0.1 else "ns"))
        print(f"[{res['Method']}]")
        print(f"  Hệ số tác động (θ)  : {res['Coefficient']:.6f} ({sig})")
        print(f"  Sai số chuẩn (SE)  : {res['Std_Error']:.6f}")
        print(f"  p-value             : {res['p-value']:.6f}")
        print(f"  Khoảng tin cậy 95%  : [{res['CI_lower']:.6f}, {res['CI_upper']:.6f}]")
        print("-" * 50)
        
    # Vẽ biểu đồ so sánh hệ số (Forest Plot)
    plt.figure(figsize=(10, 5))
    methods = [res['Method'] for res in results]
    coefs = [res['Coefficient'] for res in results]
    errors = [res['Coefficient'] - res['CI_lower'] for res in results]
    
    plt.errorbar(coefs, range(len(methods)), xerr=errors, fmt='o', color='navy', 
                 ecolor='red', elinewidth=2, capsize=5, label='Hệ số θ ± 95% CI')
    plt.yticks(range(len(methods)), methods)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("So Sánh Hệ Số Tác Động Nhân Quả (θ): OLS vs Double Machine Learning")
    plt.xlabel("Hệ số tác động (θ)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    
    print(f"\n✅ Lưu bảng kết quả tại  : {OUTPUT_CSV}")
    print(f"✅ Lưu đồ thị hệ số tại  : {OUTPUT_PLOT}")
    print("="*60)

except Exception as e:
    print(f"Lỗi hệ thống: {e}")
    import traceback
    traceback.print_exc()
