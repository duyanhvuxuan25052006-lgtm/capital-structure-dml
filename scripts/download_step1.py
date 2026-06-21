import sys
import os
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUTPUT_DIR = r"D:\draft 2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    print("Khởi tạo vnstock 4.0 Reference API...")
    from vnstock import Reference
    ref = Reference()
    
    print("Đang tải danh sách toàn bộ mã niêm yết...")
    df_all = ref.equity.list()
    print(f"Tổng số mã tải được: {len(df_all)}")
    
    # Chỉ giữ các mã cổ phiếu phổ thông có 3 chữ cái (loại chứng quyền, chứng chỉ quỹ)
    df_filtered = df_all[df_all['symbol'].str.len() == 3].copy()
    print(f"Số mã có 3 chữ cái: {len(df_filtered)}")
    
    # Loại bỏ các ngân hàng, chứng khoán, bảo hiểm, quỹ đầu tư bằng từ khóa trong tên doanh nghiệp
    exclude_keywords = ['ngân hàng', 'chứng khoán', 'bảo hiểm', 'quỹ', 'invest', 'securities', 'bank', 'insurance']
    df_filtered['organ_name_lower'] = df_filtered['organ_name'].str.lower()
    
    df_clean = df_filtered[~df_filtered['organ_name_lower'].apply(
        lambda name: any(kw in str(name) for kw in exclude_keywords)
    )]
    
    # Lưu file CSV bước 1
    output_file = os.path.join(OUTPUT_DIR, "step1_tickers_list.csv")
    df_clean[['symbol', 'organ_name']].to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("  BƯỚC 1 HOÀN THÀNH!")
    print("="*50)
    print(f"  Đã lọc sạch các công ty tài chính.")
    print(f"  Số lượng mã phi tài chính khả dụng: {len(df_clean)}")
    print(f"  File kết quả lưu tại: {output_file}")
    print("="*50)
    
except Exception as e:
    print(f"Gặp lỗi ở Bước 1: {e}")
    import traceback
    traceback.print_exc()
