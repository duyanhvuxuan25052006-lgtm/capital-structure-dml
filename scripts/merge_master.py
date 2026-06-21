import sys
import os
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DIR = r"D:\draft 2"

try:
    print("Đang đọc các file dữ liệu...")
    ratios_file = os.path.join(DIR, "step3_financial_ratios.csv")
    profiles_file = os.path.join(DIR, "step2_company_profiles.csv")
    
    if not os.path.exists(ratios_file) or not os.path.exists(profiles_file):
        raise FileNotFoundError("Thiếu file dữ liệu đầu vào.")
        
    df_ratios = pd.read_csv(ratios_file)
    df_profiles = pd.read_csv(profiles_file)
    
    print(f"Kích thước file ratios   : {df_ratios.shape}")
    print(f"Kích thước file profiles : {df_profiles.shape}")
    
    # Ghép nối bằng mã chứng khoán (left_on='ticker', right_on='symbol')
    print("\nĐang thực hiện ghép nối (Merge)...")
    df_master = df_ratios.merge(df_profiles, left_on='ticker', right_on='symbol', how='left')
    
    # Xóa cột symbol trùng lặp từ file profiles
    if 'symbol' in df_master.columns:
        df_master = df_master.drop(columns=['symbol'])
        
    # Tính toán biến Firm Age (Tuổi đời doanh nghiệp từ ngày niêm yết)
    # listing_date có định dạng 'YYYY-MM-DD...'
    print("Đang tính toán các biến phái sinh (Firm Age)...")
    df_master['listing_year'] = pd.to_datetime(df_master['listing_date'], errors='coerce').dt.year
    df_master['report_year'] = df_master['period'].str.split('-').str[0].astype(float)
    df_master['firm_age'] = df_master['report_year'] - df_master['listing_year']
    # Nếu có giá trị âm hoặc rỗng (do lỗi định dạng ngày), điền mặc định bằng 5
    df_master['firm_age'] = df_master['firm_age'].apply(lambda x: x if x >= 0 else 5.0).fillna(5.0)
    
    # Sắp xếp lại dữ liệu bảng
    df_master = df_master.sort_values(by=['ticker', 'period']).reset_index(drop=True)
    
    output_file = os.path.join(DIR, "master_panel_dataset.csv")
    df_master.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("  GHÉP NỐI MASTER HOÀN THÀNH!")
    print("="*50)
    print(f"  Tổng số quan sát (N)  : {len(df_master)}")
    print(f"  Tổng số cột (Biến X,Y): {len(df_master.columns)}")
    print(f"  File Master lưu tại   : {output_file}")
    print("="*50)
    
except Exception as e:
    print(f"Lỗi khi ghép nối: {e}")
    import traceback
    traceback.print_exc()
