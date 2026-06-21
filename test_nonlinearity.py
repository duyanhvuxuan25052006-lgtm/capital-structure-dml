import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DIR = r"D:\draft 2"
INPUT_FILE = os.path.join(DIR, "master_panel_dataset.csv")

try:
    print("1. Đọc dữ liệu Master...")
    df = pd.read_csv(INPUT_FILE)
    
    y_col = 'roa'
    x_cols = ['current_ratio', 'quick_ratio', 'asset_turnover', 'gross_margin', 'net_margin', 'firm_age']
    
    df_clean = df.dropna(subset=[y_col] + x_cols).copy()
    
    # Biến đổi Fixed Effects (Entity-Demeaning) để làm sạch dữ liệu bảng
    grouped_mean = df_clean.groupby('ticker')[[y_col] + x_cols].transform('mean')
    df_demeaned = df_clean.copy()
    for col in [y_col] + x_cols:
        df_demeaned[col] = df_clean[col] - grouped_mean[col]
        
    Y = df_demeaned[y_col].values
    X = df_demeaned[x_cols].values
    
    # ══════════════════════════════════════════════════
    # KIỂM ĐỊNH 1: RAMSEY RESET TEST CHO TÍNH PHI TUYẾN
    # ══════════════════════════════════════════════════
    print("\n2. Đang thực hiện kiểm định Ramsey RESET Test...")
    # Hồi quy tuyến tính OLS
    X_constant = sm.add_constant(X)
    ols_res = sm.OLS(Y, X_constant).fit()
    
    # RESET test thêm các số mũ bậc 2, bậc 3 của giá trị dự báo vào mô hình để kiểm tra tính phi tuyến
    reset_res = linear_reset(ols_res, power=3, use_f=True)
    print(f"  • F-statistic của kiểm định RESET: {reset_res.fvalue:.4f}")
    print(f"  • p-value của kiểm định RESET: {reset_res.pvalue:.8f}")
    
    if reset_res.pvalue < 0.05:
        print("  => KẾT LUẬN: Bác bỏ giả thuyết tuyến tính ở mức ý nghĩa 5%.")
        print("     Mối quan hệ giữa Y (ROA) và các biến kiểm soát X thực sự là PHI TUYẾN TÍNH.")
    else:
        print("  => KẾT LUẬN: Chưa đủ bằng chứng bác bỏ quan hệ tuyến tính.")
        
    # ══════════════════════════════════════════════════
    # KIỂM ĐỊNH 2: SO SÁNH HIỆU QUẢ DỰ BÁO (LINEAR VS ML NON-LINEAR)
    # ══════════════════════════════════════════════════
    print("\n3. So sánh hiệu quả giải thích biến động (R² từ Cross-Validation)...")
    # Linear Model (OLS)
    from sklearn.linear_model import LinearRegression
    r2_linear = cross_val_score(LinearRegression(), X, Y, cv=5, scoring='r2').mean()
    
    # Non-linear Model (Random Forest)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
    r2_rf = cross_val_score(rf, X, Y, cv=5, scoring='r2').mean()
    
    print(f"  • R² trung bình của mô hình tuyến tính (OLS)        : {r2_linear:+.6f}")
    print(f"  • R² trung bình của mô hình phi tuyến (Random Forest): {r2_rf:+.6f}")
    print(f"  • Độ chênh lệch hiệu quả giải thích (ML vs Linear) : {r2_rf - r2_linear:+.6f}")

except Exception as e:
    print(f"Lỗi: {e}")
    import traceback
    traceback.print_exc()
