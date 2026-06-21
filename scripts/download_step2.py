import sys
import os
import time
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

INPUT_FILE = r"D:\draft 2\step1_tickers_list.csv"
OUTPUT_FILE = r"D:\draft 2\step2_company_profiles.csv"

major_tickers = [
    'HPG', 'FPT', 'VNM', 'MSN', 'VIC', 'VRE', 'PVD', 'HSG', 'MWG', 'PNJ',
    'REE', 'GMD', 'DGC', 'KBC', 'KDH', 'NLG', 'HDG', 'DXG', 'VHC', 'ANV',
    'PC1', 'DBC', 'BMP', 'NTP', 'CSV', 'LIX', 'GIL', 'HAX', 'AAA', 'SMC',
    'PTB', 'DPR', 'PHR', 'TRC', 'TNG', 'MSH', 'TLG', 'PAN', 'DBD', 'DMC',
    'IMP', 'DHT', 'TRA', 'VGC', 'SZC', 'TIP', 'D2D', 'DGW', 'PET', 'FRT'
]

try:
    print("Đang đọc danh sách mã từ Bước 1...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_FILE}. Vui lòng chạy download_step1.py trước.")
        
    df_tickers = pd.read_csv(INPUT_FILE)
    tickers_to_download = [t for t in major_tickers if t in df_tickers['symbol'].values]
    print(f"Số lượng mã sẽ tải thông tin doanh nghiệp: {len(tickers_to_download)}")
    
    print("\nKhởi tạo vnstock 4.0 Company API...")
    from vnstock.api.company import Company
    
    profiles = []
    success_count = 0
    
    idx = 0
    while idx < len(tickers_to_download):
        ticker = tickers_to_download[idx]
        print(f"[{idx+1}/{len(tickers_to_download)}] Đang tải thông tin: {ticker}...")
        try:
            c = Company(source="vci", symbol=ticker, show_log=False)
            df_ov = c.overview()
            
            if df_ov.empty:
                print(f"  ⚠️ Không có thông tin overview cho mã {ticker}.")
                idx += 1
                continue
                
            row = df_ov.iloc[0]
            profile_data = {
                'symbol': ticker,
                'organ_name': row.get('organ_name', ''),
                'sector': row.get('sector', ''),
                'foreigner_percentage': row.get('foreigner_percentage', 0.0),
                'state_percentage': row.get('state_percentage', 0.0),
                'free_float_percentage': row.get('free_float_percentage', 0.0),
                'listing_date': row.get('listing_date', '')
            }
            profiles.append(profile_data)
            success_count += 1
            print(f"  ✅ Thành công. Ngành: {profile_data['sector']} | Nhà nước %: {profile_data['state_percentage']*100:.2f}%")
            
            idx += 1
            # Proactively sleep 4.5 seconds to stay under the 20 requests/minute rate limit
            time.sleep(4.5)
            
        except Exception as ex:
            print(f"  ⚠️ Gặp lỗi hoặc chạm giới hạn API khi tải mã {ticker}: {ex}")
            print("  Chờ 65 giây để reset giới hạn API rồi thử lại...")
            time.sleep(65)
            # Không tăng idx để vòng lặp chạy lại mã này
            
    if profiles:
        # Nếu đã có file từ trước, hãy ghép nối để tránh mất dữ liệu
        if os.path.exists(OUTPUT_FILE):
            try:
                df_old = pd.read_csv(OUTPUT_FILE)
                # Loại bỏ các mã trùng lặp trong df_old
                df_old = df_old[~df_old['symbol'].isin([p['symbol'] for p in profiles])]
                df_new = pd.DataFrame(profiles)
                df_final = pd.concat([df_old, df_new], ignore_index=True)
            except:
                df_final = pd.DataFrame(profiles)
        else:
            df_final = pd.DataFrame(profiles)
            
        df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print("\n" + "="*50)
        print("  BƯỚC 2 HOÀN THÀNH VÀ ĐÃ LƯU!")
        print("="*50)
        print(f"  Số mã thành công      : {success_count}/{len(tickers_to_download)}")
        print(f"  File kết quả lưu tại  : {OUTPUT_FILE}")
        print("="*50)
    else:
        print("\n❌ Lỗi: Không tải được thông tin của bất kỳ doanh nghiệp nào.")

except Exception as e:
    print(f"Gặp lỗi ở Bước 2: {e}")
    import traceback
    traceback.print_exc()
