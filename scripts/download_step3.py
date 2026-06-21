import sys
import os
import time
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

INPUT_FILE = r"D:\draft 2\step2_company_profiles.csv"
OUTPUT_FILE = r"D:\draft 2\step3_financial_ratios.csv"

def reshape_financial_ratios(df_ratio, symbol):
    df_ratio = df_ratio.copy()
    
    # Tìm hàng Năm và Quý để tạo chỉ mục thời gian
    row_year = df_ratio[df_ratio['item_id'] == 'year'].iloc[0]
    row_quarter = df_ratio[df_ratio['item_id'] == 'quarter'].iloc[0]
    
    data_cols = [c for c in df_ratio.columns if c not in ['item', 'item_en', 'item_id']]
    
    periods = []
    for col in data_cols:
        y = str(row_year[col]).strip()
        q = str(row_quarter[col]).strip()
        periods.append(f"{y}-Q{q}")
        
    df_data = df_ratio[~df_ratio['item_id'].isin(['year', 'quarter'])].copy()
    df_data = df_data[['item_id'] + data_cols]
    
    # Chuyển vị
    df_transposed = df_data.set_index('item_id').T
    df_transposed.index = periods
    df_transposed.index.name = 'period'
    
    # Thêm cột mã cổ phiếu
    df_transposed.insert(0, 'ticker', symbol)
    return df_transposed.reset_index()

try:
    print("Đang đọc danh sách mã từ Bước 2...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_FILE}. Vui lòng chạy download_step2.py trước.")
        
    df_profiles = pd.read_csv(INPUT_FILE)
    tickers_to_download = df_profiles['symbol'].tolist()
    print(f"Số lượng mã sẽ tải chỉ số tài chính: {len(tickers_to_download)}")
    
    # Tạo danh sách để lưu dữ liệu của từng mã
    all_data = []
    
    # Nếu file đầu ra đã tồn tại từ trước, đọc lên để lấy danh sách các mã đã tải thành công
    downloaded_tickers = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE)
            downloaded_tickers = set(df_existing['ticker'].unique())
            print(f"Phát hiện file đã tồn tại. Đã có dữ liệu của {len(downloaded_tickers)} mã.")
            # Nạp dữ liệu cũ vào danh sách
            for ticker in downloaded_tickers:
                all_data.append(df_existing[df_existing['ticker'] == ticker])
        except Exception as e:
            print(f"Không đọc được file cũ, sẽ tải lại từ đầu: {e}")
            downloaded_tickers = set()

    print("\nKhởi tạo vnstock 4.0 API...")
    from vnstock.api.financial import Finance
    
    idx = 0
    success_count = len(downloaded_tickers)
    
    while idx < len(tickers_to_download):
        ticker = tickers_to_download[idx]
        
        # Nếu mã này đã được tải ở phiên trước, bỏ qua
        if ticker in downloaded_tickers:
            idx += 1
            continue
            
        print(f"[{idx+1}/{len(tickers_to_download)}] Đang tải chỉ số tài chính: {ticker}...")
        try:
            # Khởi tạo nguồn VCI
            f_vci = Finance(source="vci", symbol=ticker, period="quarter")
            df_raw = f_vci._get_report(report_type="ratio", limit=100, mode="final")
            
            if df_raw.empty or len(df_raw) < 5:
                print(f"  ⚠️ Mã {ticker} không có dữ liệu chỉ số tài chính.")
                idx += 1
                continue
                
            # Định hình lại dữ liệu dọc
            df_reshaped = reshape_financial_ratios(df_raw, ticker)
            
            # Loại bỏ các kỳ báo cáo năm kiểm toán trùng lặp (Q5)
            df_reshaped = df_reshaped[~df_reshaped['period'].str.contains('-Q5')]
            
            all_data.append(df_reshaped)
            downloaded_tickers.add(ticker)
            success_count += 1
            print(f"  ✅ Thành công. Số kỳ: {len(df_reshaped)}")
            
            # Ghi đè file CSV liên tục sau mỗi mã thành công (Incremental Save) để đề phòng lỗi mạng giữa chừng
            df_temp = pd.concat(all_data, ignore_index=True)
            df_temp.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            
            idx += 1
            # Proactively sleep 4.5 seconds to stay under the 20 requests/minute limit
            time.sleep(4.5)
            
        except Exception as ex:
            print(f"  ⚠️ Gặp lỗi hoặc chạm giới hạn API khi tải mã {ticker}: {ex}")
            print("  Chờ 65 giây để reset giới hạn API rồi thử lại...")
            time.sleep(65)
            
    if all_data:
        df_master = pd.concat(all_data, ignore_index=True)
        # Sắp xếp theo ticker và kỳ báo cáo để dữ liệu bảng có thứ tự
        df_master = df_master.sort_values(by=['ticker', 'period']).reset_index(drop=True)
        df_master.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print("  BƯỚC 3 HOÀN THÀNH VÀ ĐÃ LƯU MASTER!")
        print("="*50)
        print(f"  Số mã thành công      : {success_count}/{len(tickers_to_download)}")
        print(f"  Tổng số quan sát (N)  : {len(df_master)}")
        print(f"  File kết quả lưu tại  : {OUTPUT_FILE}")
        print("="*50)
    else:
        print("\n❌ Lỗi: Không tải được chỉ số tài chính của bất kỳ doanh nghiệp nào.")

except Exception as e:
    print(f"Gặp lỗi ở Bước 3: {e}")
    import traceback
    traceback.print_exc()
