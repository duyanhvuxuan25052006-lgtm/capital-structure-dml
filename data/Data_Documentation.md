# 📚 TÀI LIỆU HƯỚNG DẪN DỮ LIỆU & MÃ NGUỒN (DATA & CODE DOCUMENTATION)

**Dự án:** Đánh giá mối liên hệ của Cấu trúc vốn lên Hiệu quả hoạt động doanh nghiệp bằng Double Machine Learning.  
**Thư mục làm việc:** `D:\draft 2`

---

## 🗂️ 1. SƠ ĐỒ THƯ MỤC CẤU TRÚC DỰ ÁN

```text
D:\draft 2\
├── data\
│   ├── all_nonfinancial_tickers.csv   # Danh sách 1.438 mã phi tài chính
│   ├── raw_financial_ratios_all.csv   # Dữ liệu chỉ số tài chính thô của 1.432 DN
│   ├── master_panel_all_firms.csv     # Dữ liệu Master sạch (32.029 obs × 19 cột)
│   └── Data_Documentation.md          # Tài liệu này
├── scripts\
│   ├── clean_and_validate.py          # Preprocessing, quarterly winsorization & feature engineering
│   ├── run_baseline_expanded.py       # Ước lượng Baseline OLS/FE/TWFE cho cả td_a và ibd_a
│   ├── run_dml_expanded.py            # Ước lượng DML cho cả hai loại đòn bẩy + HTE + Cluster SE
│   └── run_all_analyses.py            # Script tổng hợp chạy toàn bộ pipeline & vẽ LOWESS
└── results\
    ├── baseline_results.csv           # Kết quả hồi quy Baseline
    ├── dml_results_expanded.csv       # Kết quả hồi quy DML (bao gồm cả HTE)
    ├── dml_seed_stability.csv         # Kết quả kiểm tra seed stability
    └── figures\
        ├── scatter_lowess_nonlinearity.png       # Đồ thị phi tuyến LOWESS
        └── convergence_diagnostics_expanded.png   # Đồ thị Forest Plot & Convergence
```

---

## 📊 2. CHI TIẾT CẤU TRÚC BIẾN SỐ TRONG MASTER DATASET

### 📄 File: `master_panel_all_firms.csv`
*   **Mô tả:** Bộ dữ liệu bảng chính (Master panel dataset) sau khi làm sạch các giá trị D/E âm, winsorize 1-99% theo từng quý, và lọc doanh nghiệp có dưới 8 quý quan sát.
*   **Kích thước (Shape):** 32.029 dòng × 19 cột.
*   **Cấu trúc cột:**
    *   `ticker`: Mã chứng khoán của doanh nghiệp (3 ký tự).
    *   `period`: Kỳ báo cáo định dạng `YYYY-QQuý` (ví dụ: `2018-Q1`).
    *   `roa` ($Y$): Tỷ suất sinh lời trên tài sản (Lợi nhuận sau thuế / Tổng tài sản).
    *   `roe`: Tỷ suất sinh lời trên vốn chủ sở hữu.
    *   `de`: Tỷ số Nợ / Vốn chủ sở hữu gốc ($D/E$).
    *   `ibd_e`: Tỷ số Nợ vay có lãi / Vốn chủ sở hữu gốc ($IBD/E$).
    *   `td_a` ($D_1$): Đòn bẩy tài sản tổng thể ($TD/A = de / (1 + de)$).
    *   `ibd_a` ($D_2$): Đòn bẩy nợ vay có lãi ($IBD/A = ibd\_e / (1 + de)$).
    *   `current_ratio` ($X_1$): Tài sản ngắn hạn / Nợ ngắn hạn.
    *   `quick_ratio` ($X_2$): (Tiền + Khoản phải thu) / Nợ ngắn hạn.
    *   `asset_turnover` ($X_3$): Doanh thu thuần / Tổng tài sản.
    *   `fixed_asset_turnover`: Doanh thu thuần / Tài sản cố định.
    *   `gross_margin` ($X_4$): Lợi nhuận gộp / Doanh thu thuần.
    *   `market_cap`: Vốn hóa thị trường (VND).
    *   `firm_size` ($X_5$): Quy mô doanh nghiệp, tính bằng $\log(\text{market\_cap})$.
    *   `tangibility` ($X_6$): Tỷ lệ tài sản cố định hữu hình, tính bằng $asset\_turnover / fixed\_asset\_turnover$ (giới hạn $[0, 1]$).
    *   `data_tenure` ($X_7$): Số năm quan sát được trong mẫu của doanh nghiệp, tính bằng công thức: `year - first_year_in_data` (Lưu ý: đây là proxy, không phải tuổi đời thực của doanh nghiệp từ khi thành lập).
    *   `year`: Năm báo cáo tài chính.
    *   `large_cap`: Biến giả nhị phân, nhận giá trị 1 nếu doanh nghiệp nằm trong nhóm Top 100 có vốn hóa trung bình lớn nhất, ngược lại nhận giá trị 0.

---

## 🛠️ 3. QUY TRÌNH HỒI QUY DML TRÊN SCRIPTS
1.  **Demean theo doanh nghiệp:** Sử dụng biến đổi trung bình nhóm (within transformation) trên tất cả các đặc trưng tài chính để loại bỏ hoàn toàn Firm Fixed Effects.
2.  **Khống chế đa cộng tuyến:** Sử dụng Lasso/Ridge (qua ElasticNetCV) để tự động hóa lựa chọn các biến giả thời gian cấp quý (32 cột) và các biến kiểm soát tài chính.
3.  **Cross-Fitting:** Chia dữ liệu thành 5-folds ngẫu nhiên để ước lượng các nuisance models ($g_0(W)$, $m_0(W)$) chéo nhau, tránh overfitting bias. Kỹ thuật này được lặp lại 5 lần ($n\_rep = 5$) để đảm bảo tính ổn định.
4.  **Cluster Standard Errors:** Xử lý sai số chuẩn cụm 1 chiều và 2 chiều cấp doanh nghiệp để đảm bảo các kết luận thống kê vững trước các tương tác chuỗi và tương quan chéo hệ thống.
