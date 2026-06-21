# 📚 TÀI LIỆU HƯỚNG DẪN DỮ LIỆU & MÃ NGUỒN (DATA & CODE DOCUMENTATION)
**Dự án:** Đánh giá tác động của Cấu trúc vốn lên Hiệu quả hoạt động doanh nghiệp bằng Double Machine Learning.  
**Thư mục làm việc:** `D:\draft 2`

---

## 🗂️ 1. SƠ ĐỒ THƯ MỤC CẤU TRÚC DỰ ÁN

```text
D:\draft 2\
├── download_step1.py          # Script tải danh sách mã cổ phiếu
├── step1_tickers_list.csv     # Dữ liệu 1,453 mã phi tài chính
├── download_step2.py          # Script tải hồ sơ thông tin doanh nghiệp
├── step2_company_profiles.csv # Dữ liệu hồ sơ của 50 công ty tiêu biểu
├── download_step3.py          # Script tải tỷ số tài chính hàng quý (2018-2026)
├── step3_financial_ratios.csv # Dữ liệu tỷ số tài chính thô của 50 công ty
├── merge_master.py            # Script ghép nối dữ liệu và tạo biến mới
└── master_panel_dataset.csv   # Dữ liệu Master hoàn chỉnh (1,591 dòng, 63 cột)
```

---

## 📊 2. CHI TIẾT CẤU TRÚC BÊN TRONG CÁC FILE DỮ LIỆU (CSV)

### 📄 File 1: `step1_tickers_list.csv`
*   **Mô tả:** Danh sách các mã cổ phiếu phi tài chính đang hoạt động trên 2 sàn giao dịch chứng khoán lớn nhất Việt Nam (HOSE và HNX).
*   **Kích thước (Shape):** 1,453 dòng × 2 cột.
*   **Cấu trúc cột:**
    *   `symbol`: Mã chứng khoán (3 chữ cái, ví dụ: HPG, FPT).
    *   `organ_name`: Tên đầy đủ của doanh nghiệp (ví dụ: Công ty Cổ phần Tập đoàn Hòa Phát).

---

### 📄 File 2: `step2_company_profiles.csv`
*   **Mô tả:** Hồ sơ thông tin cơ bản của 50 doanh nghiệp tiêu biểu để sử dụng làm mẫu nghiên cứu.
*   **Kích thước (Shape):** 50 dòng × 7 cột.
*   **Cấu trúc cột:**
    *   `symbol`: Mã chứng khoán của công ty.
    *   `organ_name`: Tên đầy đủ của doanh nghiệp.
    *   `sector`: Phân ngành cấp 1 (ví dụ: Technology, Basic Resources, Real Estate) - **Biến kiểm soát cố định ngành**.
    *   `foreigner_percentage`: Tỷ lệ sở hữu nước ngoài (giá trị từ `0.0` đến `1.0`).
    *   `state_percentage`: Tỷ lệ sở hữu của Nhà nước (giá trị từ `0.0` đến `1.0`).
    *   `free_float_percentage`: Tỷ lệ cổ phiếu tự do chuyển nhượng trên thị trường.
    *   `listing_date`: Ngày doanh nghiệp chính thức lên sàn giao dịch (định dạng `YYYY-MM-DD...`).

---

### 📄 File 3: `step3_financial_ratios.csv`
*   **Mô tả:** Bảng các chỉ số tài chính hàng quý thô của 50 doanh nghiệp từ quý 1/2018 đến quý 1/2026 sau khi xoay chiều dọc.
*   **Kích thước (Shape):** 1,591 dòng × 54 cột.
*   **Cấu trúc cột:**
    *   `period`: Kỳ báo cáo định dạng `Năm-QQuý` (ví dụ: `2018-Q1`).
    *   `ticker`: Mã chứng khoán của công ty.
    *   Các cột từ 2 đến 53 là **52 chỉ số tài chính và hiệu quả hoạt động**, tiêu biểu gồm:
        *   `roa` / `roe`: Tỷ suất sinh lời trên tài sản / vốn chủ sở hữu ($Y$ chính).
        *   `debt_to_equity`: Tỷ số Nợ trên Vốn chủ sở hữu ($D$ chính).
        *   `financial_leverage`: Hệ số đòn bẩy tài chính tổng tài sản/vốn chủ sở hữu.
        *   `current_ratio` / `quick_ratio`: Khả năng thanh toán hiện hành/thanh toán nhanh.
        *   `asset_turnover`: Vòng quay tổng tài sản (đo lường hiệu suất sử dụng vốn).
        *   `gross_margin` / `net_margin`: Biên lợi nhuận gộp / biên lợi nhuận ròng.

---

### 📄 File 4: `master_panel_dataset.csv`
*   **Mô tả:** Bộ dữ liệu Master cuối cùng được tạo ra bằng cách gộp chỉ số tài chính (File 3) với thông tin doanh nghiệp (File 2), và tạo thêm các biến kiểm soát phái sinh. Đây là file trực tiếp đưa vào các mô hình hồi quy vĩ mô và DML.
*   **Kích thước (Shape):** 1,591 dòng × 63 cột.
*   **Các cột bổ sung quan trọng:**
    *   `sector`: Được ghép từ File 2 để tạo biến giả cố định ngành.
    *   `foreigner_percentage` / `state_percentage`: Tỷ lệ sở hữu làm biến kiểm soát vĩ mô.
    *   `listing_year`: Năm doanh nghiệp bắt đầu niêm yết trên sàn chứng khoán.
    *   `report_year`: Năm của kỳ báo cáo hiện tại (được tách từ cột `period`).
    *   **`firm_age` (Tuổi doanh nghiệp):** Được tính bằng công thức: `report_year` - `listing_year` (Đại diện cho tuổi đời và độ trưởng thành của doanh nghiệp).

---

## 💻 3. GIẢI THÍCH CHI TIẾT DÒNG CODE VÀ KỊCH BẢN PYTHON

### 🛠️ Kịch bản 1: `download_step1.py`
*   **Nhiệm vụ:** Lấy danh sách 1,453 mã chứng khoán phi tài chính.
*   **Giải thích code quan trọng:**
    *   `df_all = ref.equity.list()`: Gọi API của Vnstock để tải về danh sách toàn bộ mã chứng khoán đang giao dịch trên thị trường.
    *   `df_filtered = df_all[df_all['symbol'].str.len() == 3]`: Lọc bỏ các mã phái sinh, chứng quyền, chứng chỉ quỹ (các mã này thường có độ dài 4 ký tự trở lên). Chỉ giữ lại mã cổ phiếu phổ thông (độ dài đúng bằng 3).
    *   `df_clean = df_filtered[~df_filtered['organ_name_lower'].apply(lambda name: any(kw in str(name) for kw in exclude_keywords))]`: Vòng lặp loại bỏ các công ty tài chính có tên chứa các từ khóa nhạy cảm như "ngân hàng", "chứng khoán", "bảo hiểm", "quỹ".

---

### 🛠️ Kịch bản 2: `download_step2.py`
*   **Nhiệm vụ:** Tải hồ sơ thông tin của 50 doanh nghiệp mẫu (xử lý Rate Limit).
*   **Giải thích code quan trọng:**
    *   `c = Company(source="vci", symbol=ticker)`: Khởi tạo đối tượng Company kết nối với API của VCI.
    *   `df_ov = c.overview()`: Tải về bảng dữ liệu tổng quan chứa thông tin ngành nghề và cơ cấu sở hữu.
    *   `time.sleep(4.5)`: **Cơ chế Proactive Rate-limiting**. Nghỉ đúng 4.5 giây sau mỗi lượt gọi để đảm bảo số lượng yêu cầu gửi lên không vượt quá 20 lần/phút (giới hạn của gói Guest).
    *   `time.sleep(65)`: **Cơ chế Phục hồi rủi ro (Retry Logic)**. Nếu trong quá trình tải mạng bị ngắt hoặc chạm trần API và ném ra ngoại lệ `Exception`, chương trình sẽ tự động đứng im trong 65 giây để reset lại băng thông của server rồi tự động tải lại chính mã đó mà không bị crash.

---

### 🛠️ Kịch bản 3: `download_step3.py`
*   **Nhiệm vụ:** Tải dữ liệu chỉ số tài chính, xoay dọc dữ liệu và loại bỏ các kỳ báo cáo năm kiểm toán trùng lắp.
*   **Giải thích code quan trọng:**
    *   `f_vci = Finance(source="vci", symbol=ticker, period="quarter")`: Khởi tạo đối tượng Finance để tải dữ liệu tài chính hàng quý.
    *   `df_raw = f_vci._get_report(report_type="ratio", limit=100, mode="final")`: **Điểm mấu chốt vượt qua giới hạn của thư viện**. Chúng ta trực tiếp gọi phương thức ẩn `_get_report` và truyền tham số `limit=100` để bắt server trả về toàn bộ lịch sử 8 năm (32 kỳ) thay vì mặc định chỉ lấy 4 kỳ gần nhất của thư viện `vnstock`.
    *   `df_reshaped = reshape_financial_ratios(df_raw, ticker)`: 
        *   Tách hàng `year` và hàng `quarter` để ghép lại thành chỉ mục thời gian duy nhất (ví dụ: `2018-Q1`).
        *   Sử dụng phương thức `.set_index().T` để xoay ngang chỉ số thành cột, giúp dữ liệu có cấu trúc dòng là thời gian và cột là biến kiểm soát.
    *   `df_reshaped = df_reshaped[~df_reshaped['period'].str.contains('-Q5')]`: Loại bỏ kỳ `-Q5` vì đây thực tế là giá trị kiểm toán năm, bị lặp lại giá trị của quý 4.
    *   **Incremental Saving (Lưu trữ lũy tiến):** Sau mỗi mã tải về thành công, script tự động ghi đè ngay lập tức vào file `step3_financial_ratios.csv`. Nếu có mất điện hay mất mạng ở mã số 40, bạn chỉ cần chạy lại script và nó sẽ tự động nhận diện các mã đã tải để chạy tiếp từ mã 41 mà không phải tải lại từ đầu.

---

### 🛠️ Kịch bản 4: `merge_master.py`
*   **Nhiệm vụ:** Ghép nối và tính toán biến phái sinh `firm_age`.
*   **Giải thích code quan trọng:**
    *   `df_master = df_ratios.merge(df_profiles, left_on='ticker', right_on='symbol', how='left')`: Thực hiện phép ghép trái (left join) gộp thông tin hồ sơ doanh nghiệp vào bảng chỉ số tài chính theo mã chứng khoán.
    *   `df_master['listing_year'] = pd.to_datetime(df_master['listing_date']).dt.year`: Chuyển chuỗi ngày niêm yết thành kiểu ngày tháng của Pandas để trích xuất năm lên sàn.
    *   `df_master['report_year'] = df_master['period'].str.split('-').str[0].astype(float)`: Tách năm của kỳ báo cáo (ví dụ tách `2018` từ chuỗi `2018-Q1`).
    *   `df_master['firm_age'] = df_master['report_year'] - df_master['listing_year']`: Tính toán tuổi đời doanh nghiệp tại từng thời điểm báo cáo tài chính.
