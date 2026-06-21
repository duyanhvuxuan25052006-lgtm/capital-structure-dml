# Tác động của Cấu trúc Vốn lên Hiệu quả Hoạt động Doanh nghiệp Niêm yết tại Việt Nam: Tiếp cận Double Machine Learning

**Capital Structure and Firm Performance in Vietnam: A Double Machine Learning Approach**

---

## Tổng quan

Nghiên cứu này ước lượng **tác động nhân quả** của cấu trúc vốn (đòn bẩy tài chính) lên hiệu quả hoạt động (ROA) của các doanh nghiệp phi tài chính niêm yết trên sàn HOSE/HNX, giai đoạn 2018–2026.

Phương pháp **Double Machine Learning (DML-PLR)** theo khung lý thuyết của Chernozhukov et al. (2018) được áp dụng, kết hợp với biến đổi Fixed Effects (Entity Demean) và Cross-Fitting 5-fold.

### Kết quả chính

| Chỉ số | Giá trị |
|---|---|
| Mẫu nghiên cứu | 50 doanh nghiệp phi tài chính, 1.591 quan sát quý |
| Giai đoạn | 2018 – 2026 |
| Hệ số θ (DML) | −0,0012 đến +0,0017 (không có ý nghĩa thống kê) |
| R² (Random Forest) | 37,53% (vượt trội so với OLS: 32,76%) |
| Ramsey RESET | F = 20,23; p < 0,001 → xác nhận quan hệ phi tuyến |

> **Kết luận:** Đòn bẩy tài chính không có tác động biên đáng kể đến ROA – phù hợp với dự báo của Trade-off Theory rằng các doanh nghiệp niêm yết lớn tại Việt Nam đã tiệm cận cấu trúc vốn tối ưu.

---

## Cấu trúc thư mục

```
├── README.md                         # File này
├── requirements.txt                  # Các thư viện Python cần thiết
├── .gitignore                        # Các file/thư mục bị loại trừ khỏi Git
│
├── data/                             # Dữ liệu nghiên cứu
│   ├── raw/                          # Dữ liệu thô từ API
│   │   ├── step1_tickers_list.csv    # Danh sách mã cổ phiếu HOSE/HNX
│   │   ├── step2_company_profiles.csv# Thông tin doanh nghiệp (ngành, sở hữu, niêm yết)
│   │   └── step3_financial_ratios.csv# Chỉ số tài chính quý từ VCI API
│   ├── processed/                    # Dữ liệu đã xử lý
│   │   └── master_panel_dataset.csv  # Bộ dữ liệu bảng chính (1.591 obs × 63 biến)
│   └── Data_Documentation.md         # Tài liệu mô tả chi tiết các biến số
│
├── scripts/                          # Mã nguồn Python
│   ├── download_step1.py             # Thu thập danh sách mã cổ phiếu
│   ├── download_step2.py             # Thu thập thông tin doanh nghiệp
│   ├── download_step3.py             # Thu thập chỉ số tài chính quý
│   ├── merge_master.py               # Ghép nối và làm sạch dữ liệu bảng
│   ├── test_nonlinearity.py          # Kiểm định Ramsey RESET (phi tuyến)
│   ├── run_dml_analysis.py           # Chạy mô hình DML-PLR chính
│   └── final_research_pipeline.py    # Pipeline tổng hợp: data → analysis → output
│
├── results/                          # Kết quả phân tích
│   ├── dml_results.csv               # Hệ số ước lượng DML chi tiết
│   ├── dml_final_report.csv          # Bảng tổng hợp kết quả cuối cùng
│   └── figures/                      # Biểu đồ
│       ├── dml_coefficient_plot.png   # Biểu đồ hệ số DML
│       └── dml_final_forest_plot.png  # Forest plot so sánh các mô hình
│
├── paper/                            # Bài nghiên cứu
│   ├── Capital_Structure_DML_Report.md   # Bản Markdown đầy đủ
│   └── Capital_Structure_DML_Report.tex  # Bản LaTeX (để biên dịch PDF)
│
└── references/                       # Tài liệu tham khảo
    └── paper_summary_vietnamese.md    # Tóm tắt bài Nguyen et al. (2023)
```

---

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 2. Thu thập dữ liệu (nếu muốn tái lập từ đầu)

```bash
python scripts/download_step1.py    # Lấy danh sách mã CK
python scripts/download_step2.py    # Lấy thông tin doanh nghiệp
python scripts/download_step3.py    # Lấy chỉ số tài chính quý
python scripts/merge_master.py      # Ghép nối thành master dataset
```

> **Lưu ý:** API VCI có giới hạn 20 requests/phút — các scripts đã tích hợp `time.sleep()` để tránh bị chặn.

### 3. Chạy phân tích

```bash
python scripts/test_nonlinearity.py      # Kiểm định Ramsey RESET
python scripts/run_dml_analysis.py       # Chạy DML-PLR (kết quả lưu vào results/)
```

Hoặc chạy toàn bộ pipeline:

```bash
python scripts/final_research_pipeline.py
```

---

## Nguồn dữ liệu

| Nguồn | Mô tả | Truy cập |
|---|---|---|
| **VCI API** (via `vnstock`) | Báo cáo tài chính quý đã kiểm toán từ HOSE/HNX | [vnstocks.com/docs](https://vnstocks.com/docs) |
| **Vietcap Securities** | Nguồn gốc dữ liệu tài chính | Thông qua vnstock v4 |

---

## Phương pháp nghiên cứu

**Double Machine Learning — Partially Linear Regression (DML-PLR):**

```
Y = θ·D + g(X) + ε
```

- **Y** = ROA (Tỷ suất sinh lời trên tổng tài sản)
- **D** = Financial Leverage (Đòn bẩy tài chính)
- **X** = Controls: Gross Margin, Asset Turnover, Current Ratio, Firm Age, Sector FE, ...
- **g(·)** = Hàm phi tuyến (ước lượng bằng Random Forest / LassoCV / ElasticNet)
- **θ** = Hệ số tác động nhân quả cần ước lượng

**Tham khảo lý thuyết:** Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). *Double/debiased machine learning for treatment and structural parameters.* The Econometrics Journal, 21(1), C1–C68.

---

## Tác giả

**Vũ Xuân Duy Anh**

Lĩnh vực: Tài chính doanh nghiệp / Kinh tế lượng ứng dụng

---

## Giấy phép

Dự án này được phát triển phục vụ mục đích nghiên cứu học thuật.
