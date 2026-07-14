"""
BƯỚC 1: TẢI DỮ LIỆU TÀI CHÍNH TOÀN BỘ DN PHI TÀI CHÍNH NIÊM YẾT
====================================================================
- Nguồn: API vnstock (VCI)
- Đối tượng: ~1.438 DN phi tài chính (HOSE/HNX/UPCoM)
- Dữ liệu: Financial ratios theo quý, tối đa 100 kỳ
- Output: D:/draft 2/data/raw_financial_ratios_all.csv
- Tính năng: Incremental save, auto-retry, resume khi bị ngắt
====================================================================
"""
import sys, os, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

INPUT_FILE = r"D:\draft 2\data\all_nonfinancial_tickers.csv"
OUTPUT_FILE = r"D:\draft 2\data\raw_financial_ratios_all.csv"
LOG_FILE = r"D:\draft 2\data\download_log.txt"

def reshape_financial_ratios(df_ratio, symbol):
    """Chuyển đổi dữ liệu dọc (items x periods) → bảng panel (period x variables)."""
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
    
    # Chuyển vị: rows = periods, columns = financial items
    df_transposed = df_data.set_index('item_id').T
    df_transposed.index = periods
    df_transposed.index.name = 'period'
    
    # Thêm cột mã cổ phiếu
    df_transposed.insert(0, 'ticker', symbol)
    
    # Loại bỏ kỳ kiểm toán (Q5)
    df_transposed = df_transposed.reset_index()
    df_transposed = df_transposed[~df_transposed['period'].str.contains('-Q5')]
    
    return df_transposed

def main():
    print("=" * 70)
    print("  TẢI DỮ LIỆU TÀI CHÍNH TOÀN BỘ DN PHI TÀI CHÍNH NIÊM YẾT")
    print("=" * 70)
    
    # Đọc danh sách mã
    df_tickers = pd.read_csv(INPUT_FILE)
    all_tickers = df_tickers['symbol'].tolist()
    print(f"Tổng số mã cần tải: {len(all_tickers)}")
    
    # Kiểm tra dữ liệu đã tải trước đó (resume)
    downloaded_tickers = set()
    all_data = []
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE)
            downloaded_tickers = set(df_existing['ticker'].unique())
            all_data.append(df_existing)
            print(f"Resume: đã có dữ liệu của {len(downloaded_tickers)} mã từ phiên trước.")
        except Exception as e:
            print(f"Không đọc được file cũ, tải lại từ đầu: {e}")
    
    remaining = [t for t in all_tickers if t not in downloaded_tickers]
    print(f"Còn lại cần tải: {len(remaining)} mã")
    print(f"Ước tính thời gian: ~{len(remaining) * 5 / 60:.0f} phút")
    print()
    
    from vnstock.api.financial import Finance
    
    success_count = len(downloaded_tickers)
    fail_count = 0
    no_data_count = 0
    
    # Mở file log
    log_f = open(LOG_FILE, 'a', encoding='utf-8')
    log_f.write(f"\n{'='*50}\nSession started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}\n")
    
    idx = 0
    save_counter = 0  # Đếm để save mỗi 20 mã
    
    while idx < len(remaining):
        ticker = remaining[idx]
        progress = success_count + no_data_count + fail_count
        total = len(all_tickers)
        pct = progress / total * 100
        
        print(f"[{progress+1}/{total}] ({pct:.1f}%) Tải: {ticker}...", end=" ", flush=True)
        
        try:
            f = Finance(source="vci", symbol=ticker, period="quarter")
            df_raw = f._get_report(report_type="ratio", limit=100, mode="final")
            
            if df_raw.empty or len(df_raw) < 5:
                print(f"THIẾU DỮ LIỆU (rows={len(df_raw)})")
                log_f.write(f"NO_DATA: {ticker} (rows={len(df_raw)})\n")
                no_data_count += 1
                idx += 1
                time.sleep(2)
                continue
            
            # Kiểm tra có year/quarter rows không
            if 'item_id' not in df_raw.columns:
                print("THIẾU item_id")
                log_f.write(f"NO_ITEM_ID: {ticker}\n")
                no_data_count += 1
                idx += 1
                time.sleep(2)
                continue
            
            if not any(df_raw['item_id'] == 'year') or not any(df_raw['item_id'] == 'quarter'):
                print("THIẾU year/quarter")
                log_f.write(f"NO_YEAR_QUARTER: {ticker}\n")
                no_data_count += 1
                idx += 1
                time.sleep(2)
                continue
            
            # Reshape
            df_reshaped = reshape_financial_ratios(df_raw, ticker)
            all_data.append(df_reshaped)
            success_count += 1
            n_periods = len(df_reshaped)
            print(f"OK ({n_periods} quý)")
            log_f.write(f"OK: {ticker} ({n_periods} periods)\n")
            
            idx += 1
            save_counter += 1
            
            # Incremental save mỗi 20 mã
            if save_counter >= 20:
                df_master = pd.concat(all_data, ignore_index=True)
                df_master.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
                save_counter = 0
                print(f"  [SAVED] {success_count} mã → {OUTPUT_FILE}")
                log_f.flush()
            
            time.sleep(4.5)
            
        except Exception as ex:
            error_msg = str(ex)
            if '429' in error_msg or 'rate' in error_msg.lower() or 'limit' in error_msg.lower():
                print(f"RATE-LIMITED → chờ 65s rồi thử lại...")
                log_f.write(f"RATE_LIMIT: {ticker} → retry\n")
                time.sleep(65)
                # Không tăng idx
            else:
                print(f"LỖI: {error_msg[:80]}")
                log_f.write(f"ERROR: {ticker} → {error_msg}\n")
                fail_count += 1
                idx += 1
                time.sleep(5)
    
    # Final save
    if all_data:
        df_master = pd.concat(all_data, ignore_index=True)
        df_master = df_master.sort_values(by=['ticker', 'period']).reset_index(drop=True)
        df_master.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    log_f.write(f"\nSession ended at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.close()
    
    print("\n" + "=" * 70)
    print("  KẾT QUẢ TẢI DỮ LIỆU")
    print("=" * 70)
    print(f"  Thành công      : {success_count}/{len(all_tickers)}")
    print(f"  Thiếu dữ liệu   : {no_data_count}")
    print(f"  Lỗi             : {fail_count}")
    if all_data:
        print(f"  Tổng quan sát   : {len(df_master)}")
        print(f"  Số DN có dữ liệu: {df_master['ticker'].nunique()}")
    print(f"  File output     : {OUTPUT_FILE}")
    print(f"  File log        : {LOG_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()
