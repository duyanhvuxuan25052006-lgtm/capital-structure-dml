"""
BƯỚC 3 (AUDITED & FIXED): LÀM SẠCH DỮ LIỆU + DATA VALIDATION
=============================================================
AUDITED FIXES INTEGRATED:
  - R1: Báo cáo chính xác số lượng outliers bị loại (975 âm, 32 cực đoan > 100, 117 missing).
  - R2: Đính chính tài liệu về data_tenure không cần winsorize (chỉ chạy từ 0-8).
  - R3: Tính toán first_year và bộ lọc MIN_PERIODS từ dữ liệu gốc trước khi dropna
        để loại bỏ survivorship bias kép và tránh làm chệch biến data_tenure.
  - R4: Đồng bộ hóa cỡ mẫu thực tế sau dropna.
  - M2: Lọc bỏ triệt để các doanh nghiệp tài chính (Ngân hàng, Bảo hiểm, Chứng khoán, Tài chính khác)
        bao gồm các mã lọt lưới như BVH, BMI, PVI, IPA, OGC, TVC.
=============================================================
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

INPUT_FILE = r"D:\draft 2\data\raw_financial_ratios_all.csv"
MAPPING_FILE = r"D:\draft 2\data\ticker_exchange_mapping.csv"
INDUSTRY_FILE = r"D:\draft 2\data\ticker_industry_mapping.csv"
OUTPUT_FILE = r"D:\draft 2\data\master_panel_all_firms.csv"
REPORT_FILE = r"D:\draft 2\results\data_validation_report.txt"

os.makedirs(r"D:\draft 2\results", exist_ok=True)
os.makedirs(r"D:\draft 2\results\figures", exist_ok=True)

def main():
    print("=" * 70)
    print("  BƯỚC 3 (AUDITED & FIXED): LÀM SẠCH DỮ LIỆU & DATA VALIDATION")
    print("=" * 70)
    
    # 1. Đọc dữ liệu thô
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"Dữ liệu thô: {df_raw.shape[0]} dòng, {df_raw.shape[1]} cột")
    
    # 2. Thống kê chi tiết outliers cấu trúc vốn từ dữ liệu gốc (Point R1)
    de_raw = pd.to_numeric(df_raw['debt_to_equity'], errors='coerce')
    ibd_e_raw = pd.to_numeric(df_raw['debtPerEquity'], errors='coerce')
    
    n_neg = (de_raw < 0).sum()
    n_extreme_pos = (de_raw > 100).sum()
    n_nan = de_raw.isna().sum()
    
    print(f"Thống kê D/E gốc:")
    print(f"  - Quan sát D/E âm (âm vốn chủ sở hữu): {n_neg}")
    print(f"  - Quan sát D/E cực đoan (> 100)      : {n_extreme_pos}")
    print(f"  - Quan sát D/E missing (NaN)        : {n_nan}")
    
    # 3. Lọc bỏ các doanh nghiệp tài chính (Point M2)
    # Lấy thông tin ngành từ file industry mapping
    df_ind = pd.read_csv(INDUSTRY_FILE)
    df_ind['ticker'] = df_ind['ticker'].str.strip()
    df_ind['industry_name'] = df_ind['industry_name'].str.strip()
    
    # Xác định các ngành tài chính cần loại bỏ
    financial_industries = ['Ngân hàng', 'Chứng khoán', 'Bảo hiểm', 'Tài chính khác']
    df_fin_ind = df_ind[df_ind['industry_name'].isin(financial_industries)]
    financial_tickers_from_mapping = set(df_fin_ind['ticker'].unique())
    
    # Danh sách mã tài chính bổ sung (Blacklist tự tạo để tránh sót các mã UPCoM chưa phân loại ngành)
    financial_blacklist = {
        'BVH', 'PVI', 'BMI', 'SSI', 'VCB', 'BID', 'CTG', 'TCB', 'VPB', 'MBB', 
        'ACB', 'SHB', 'HDB', 'VIB', 'TPB', 'MSB', 'LPB', 'OCB', 'NAB', 'SGB', 
        'ABB', 'BAB', 'VBB', 'KLB', 'PGB', 'BVB', 'HCM', 'VCI', 'VND', 'SHS', 
        'MBS', 'FTS', 'CTS', 'BSI', 'AGR', 'BVS', 'TVS', 'VIX', 'ORS', 'PSI', 
        'APG', 'WSS', 'HBS', 'VIG', 'EVS', 'IVS', 'TVB', 'TVC', 'TCI', 'APS', 
        'DSC', 'AAS', 'VFS', 'SBS', 'IPA', 'OGC'
    }
    
    all_financial_tickers = financial_tickers_from_mapping.union(financial_blacklist)
    
    # Loại bỏ khỏi df_raw
    df_raw_clean = df_raw[~df_raw['ticker'].isin(all_financial_tickers)].copy()
    n_removed_financials = df_raw['ticker'].nunique() - df_raw_clean['ticker'].nunique()
    print(f"Đã loại bỏ {n_removed_financials} doanh nghiệp tài chính lọt lưới (bao gồm cả BVH, BMI, PVI, IPA, OGC, TVC).")
    
    # 4. Tính toán first_year và số quý xuất hiện từ dữ liệu phi tài chính thô (Point R3)
    df_raw_clean['year'] = df_raw_clean['period'].str[:4].astype(int)
    first_year_mapping = df_raw_clean.groupby('ticker')['year'].min().to_dict()
    raw_periods_per_firm = df_raw_clean.groupby('ticker').size()
    
    # Doanh nghiệp có ít nhất 8 quý xuất hiện trên thị trường
    MIN_PERIODS = 8
    firms_with_enough_periods = raw_periods_per_firm[raw_periods_per_firm >= MIN_PERIODS].index
    print(f"Số lượng DN phi tài chính có ít nhất {MIN_PERIODS} quý niêm yết: {len(firms_with_enough_periods)}")
    
    # 5. Lọc các cột cần thiết và định nghĩa lại biến số
    target_cols = {
        'ticker': 'ticker',
        'period': 'period',
        'roa': 'roa',
        'roe': 'roe',
        'de': 'debt_to_equity',
        'ibd_e': 'debtPerEquity',
        'current_ratio': 'current_ratio',
        'quick_ratio': 'quick_ratio',
        'asset_turnover': 'asset_turnover',
        'fixed_asset_turnover': 'fixed_asset_turnover',
        'gross_margin': 'gross_margin',
        'ebit_margin': 'ebit_margin',
        'market_cap': 'market_cap'
    }
    
    df_sel = df_raw_clean[[c for c in target_cols.values() if c in df_raw_clean.columns]].copy()
    reverse_map = {v: k for k, v in target_cols.items()}
    df_sel = df_sel.rename(columns=reverse_map)
    
    # Chuyển đổi định dạng số
    numeric_cols = [c for c in df_sel.columns if c not in ['ticker', 'period']]
    for c in numeric_cols:
        df_sel[c] = pd.to_numeric(df_sel[c], errors='coerce')
        
    # Áp dụng bộ lọc đòn bẩy
    invalid_mask = (df_sel['de'] < 0) | (df_sel['de'].isna()) | (df_sel['de'] > 100) | (df_sel['ibd_e'] < 0) | (df_sel['ibd_e'].isna())
    df_sel.loc[invalid_mask, ['de', 'ibd_e']] = np.nan
    
    # Tính đòn bẩy tài sản
    df_sel['td_a'] = df_sel['de'] / (1 + df_sel['de'])
    df_sel['ibd_a'] = df_sel['ibd_e'] / (1 + df_sel['de'])
    
    # Tính Operating ROA (Point A1) & Controls
    df_sel['operating_roa'] = df_sel['ebit_margin'] * df_sel['asset_turnover']
    df_sel['firm_size'] = np.log(df_sel['market_cap'].replace(0, np.nan))
    df_sel['tangibility'] = np.where(df_sel['fixed_asset_turnover'] > 0, 
                                     df_sel['asset_turnover'] / df_sel['fixed_asset_turnover'], 
                                     0.0)
    df_sel['tangibility'] = df_sel['tangibility'].clip(0, 1)
    
    # Gán first_year từ dữ liệu gốc trước lọc (Point R3)
    df_sel['first_year'] = df_sel['ticker'].map(first_year_mapping)
    df_sel['year'] = df_sel['period'].str[:4].astype(int)
    df_sel['data_tenure'] = df_sel['year'] - df_sel['first_year']
    
    # 6. Lọc các DN đủ tiêu chuẩn thời gian từ dữ liệu thô (Point R3)
    df_sel = df_sel[df_sel['ticker'].isin(firms_with_enough_periods)].copy()
    
    # 7. Loại bỏ NaN cho các biến chạy mô hình chính (Point R4)
    all_model_vars = ['roa', 'operating_roa', 'td_a', 'ibd_a', 'current_ratio', 
                      'quick_ratio', 'asset_turnover', 'gross_margin', 'firm_size', 'tangibility']
    
    df_clean = df_sel.dropna(subset=all_model_vars).copy()
    
    # Duplicates
    n_dup = df_clean.duplicated(subset=['ticker', 'period']).sum()
    if n_dup > 0:
        df_clean = df_clean.drop_duplicates(subset=['ticker', 'period'], keep='last')
        
    # 8. Winsorize cấp quý cho các biến liên tục (không winsorize data_tenure - Point R2)
    def winsorize_quarter(group):
        for col in all_model_vars:
            p1 = group[col].quantile(0.01)
            p99 = group[col].quantile(0.99)
            group[col] = group[col].clip(p1, p99)
        return group
        
    df_clean = df_clean.groupby('period', group_keys=False).apply(winsorize_quarter)
    
    # 9. Sắp xếp chuỗi thời gian để tính biến bậc hai và biến trễ
    df_clean = df_clean.sort_values(['ticker', 'period']).reset_index(drop=True)
    
    # Biến bậc hai (Point A2)
    df_clean['td_a_sq'] = df_clean['td_a'] ** 2
    df_clean['ibd_a_sq'] = df_clean['ibd_a'] ** 2
    
    # Biến trễ 1 kỳ (lagged treatment) kiểm soát khoảng hở
    def period_to_qidx(p):
        y, q = p.split('-Q')
        return int(y)*4 + int(q)
        
    df_clean['q_idx'] = df_clean['period'].apply(period_to_qidx)
    
    lag_lookup = df_clean[['ticker', 'q_idx', 'td_a', 'ibd_a']].copy()
    lag_lookup['q_idx'] += 1  # gia tri cua quy X tro thanh "lag" cho quy X+1
    lag_lookup = lag_lookup.rename(columns={'td_a': 'td_a_lag', 'ibd_a': 'ibd_a_lag'})
    
    df_clean = df_clean.merge(lag_lookup, on=['ticker', 'q_idx'], how='left')
    df_clean = df_clean.drop(columns=['q_idx'], errors='ignore')
    
    # 10. Tính nhãn large_cap (Top 100 DN có vốn hóa trung bình lớn nhất)
    avg_mc = df_clean.groupby('ticker')['market_cap'].mean().reset_index()
    avg_mc = avg_mc.sort_values(by='market_cap', ascending=False)
    large_cap_tickers = set(avg_mc.head(100)['ticker'])
    df_clean['large_cap'] = df_clean['ticker'].isin(large_cap_tickers).astype(int)
    
    # 11. Ghép nối sàn giao dịch để phân nhóm robustness (Point A5)
    if os.path.exists(MAPPING_FILE):
        df_map = pd.read_csv(MAPPING_FILE)
        df_map['ticker'] = df_map['ticker'].str.strip()
        df_clean = df_clean.merge(df_map, on='ticker', how='left')
        print("   Da ghep noi thong tin san giao dich tu file mapping.")
    else:
        df_clean['exchange'] = 'UNKNOWN'
        print("   WARNING: Khong tim thay file mapping san giao dich!")
        
    # 12. Lưu kết quả
    df_clean = df_clean.sort_values(['ticker', 'period']).reset_index(drop=True)
    df_clean.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*70}")
    print(f"  KET QUA MASTER PANEL MOI (ĐA LOC TAI CHINH)")
    print(f"{'='*70}")
    print(f"  So DN      : {df_clean['ticker'].nunique()}")
    print(f"  Tong obs   : {len(df_clean)}")
    print(f"  Cot du lieu: {list(df_clean.columns)}")
    
    # Lưu nhật ký làm sạch chi tiết phục vụ báo cáo
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"=== BAO CAO LAM SACH DU LIEU (AUDITED & FIXED) ===\n")
        f.write(f"Raw observations: {df_raw.shape[0]}\n")
        f.write(f"Removed negative D/E observations: {n_neg}\n")
        f.write(f"Removed extreme positive D/E (>100) observations: {n_extreme_pos}\n")
        f.write(f"Removed missing D/E observations: {n_nan}\n")
        f.write(f"Removed leaked financial firms: {n_removed_financials}\n")
        f.write(f"Firms with >= 8 quarters in raw data: {len(firms_with_enough_periods)}\n")
        f.write(f"Final clean observations: {len(df_clean)}\n")
        f.write(f"Final clean firms: {df_clean['ticker'].nunique()}\n")

if __name__ == "__main__":
    main()
