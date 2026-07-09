# Tác động của Cấu trúc Vốn lên Hiệu quả Hoạt động Doanh nghiệp Niêm yết tại Việt Nam: Tiếp cận Double Machine Learning

**Capital Structure and Firm Performance in Vietnam: A Double Machine Learning Approach**

---

## Tổng quan

Nghiên cứu này ước lượng tác động của cấu trúc vốn (đòn bẩy tài chính) lên hiệu quả hoạt động (ROA) của các doanh nghiệp phi tài chính niêm yết trên sàn HOSE/HNX, giai đoạn 2018–2026.

Pipeline chính hiện sử dụng toàn bộ doanh nghiệp phi tài chính HOSE/HNX có dữ liệu phù hợp, thay cho danh sách hard-code 50 doanh nghiệp lớn. Phương pháp **Double Machine Learning (DML-PLR)** được kết hợp với Entity Fixed Effects, biến giả thời gian và cross-fitting theo doanh nghiệp.

### Kết quả chính

| Chỉ số | Giá trị |
|---|---|
| Mẫu nghiên cứu | 619 doanh nghiệp phi tài chính đang niêm yết, 18.873 quan sát quý |
| Sàn | 343 HOSE, 276 HNX |
| Giai đoạn mô hình | 2018-Q1 – 2026-Q1, loại 2021-Q2 do coverage thấp |
| Hệ số θ (DML-Random Forest) | −0,04085; SE = 0,00411; p < 0,001 |
| Khoảng tin cậy 95% | [−0,04891; −0,03279] |
| Robustness tối thiểu 16 quý | θ = −0,04084 trên 609 doanh nghiệp |

> **Kết luận cập nhật:** Trên mẫu HOSE/HNX mở rộng, tất cả đặc tả OLS, FE và DML đều cho hệ số âm có ý nghĩa thống kê. Với RandomForest DML, D/A tăng 0,10 gắn với ROA giảm khoảng 0,408 điểm phần trăm. Kết quả này không nên được mô tả là bằng chứng nhân quả tuyệt đối vì DML-PLR chưa xử lý nội sinh từ biến không quan sát nếu không có biến công cụ.

> Pipeline và báo cáo 50 doanh nghiệp cũ được giữ lại để đối chiếu lịch sử, nhưng không còn là kết quả chính của repo.

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
│   │   ├── master_panel_dataset.csv  # Bộ dữ liệu legacy 50 doanh nghiệp
│   │   ├── master_panel_hose_hnx_dataset.csv
│   │   └── analysis_panel_hose_hnx_dataset.csv
│   └── Data_Documentation.md         # Tài liệu mô tả chi tiết các biến số
│
├── scripts/                          # Mã nguồn Python
│   ├── download_step1.py             # Thu thập danh sách mã cổ phiếu
│   ├── download_step2.py             # Thu thập thông tin doanh nghiệp
│   ├── download_step3.py             # Thu thập chỉ số tài chính quý
│   ├── merge_master.py               # Ghép nối và làm sạch dữ liệu bảng
│   ├── test_nonlinearity.py          # Kiểm định Ramsey RESET (phi tuyến)
│   ├── run_dml_analysis.py           # DML-PLR legacy 50 doanh nghiệp
│   ├── final_research_pipeline.py    # Pipeline legacy 50 doanh nghiệp
│   └── expanded_hose_hnx_pipeline.py # Pipeline chính HOSE+HNX mở rộng
│
├── results/                          # Kết quả phân tích
│   ├── dml_final_report.csv          # Kết quả legacy 50 doanh nghiệp
│   ├── expanded_hose_hnx/            # Kết quả chính mẫu mở rộng
│   └── figures/                      # Biểu đồ
│       ├── dml_coefficient_plot.png   # Biểu đồ hệ số DML
│       └── dml_final_forest_plot.png  # Forest plot so sánh các mô hình
│
├── paper/                            # Bài nghiên cứu
│   ├── Capital_Structure_DML_Report.md   # Bản Markdown đầy đủ
│   └── Capital_Structure_DML_Report.tex  # Bản LaTeX (để biên dịch PDF)
├── reports/
│   └── hose_hnx_expanded_dml_report.md   # Báo cáo kỹ thuật kết quả mới
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

### 2. Chạy pipeline HOSE/HNX mở rộng

```bash
python scripts/expanded_hose_hnx_pipeline.py --steps profiles
python scripts/expanded_hose_hnx_pipeline.py --steps ratios --sleep 4.5 --retry-sleep 65
python scripts/expanded_hose_hnx_pipeline.py --steps merge
python scripts/expanded_hose_hnx_pipeline.py --steps analysis
python scripts/expanded_hose_hnx_pipeline.py --steps report
```

> **Lưu ý:** API guest có giới hạn 20 requests/phút. Dùng `--sleep 4.5` khi tải toàn bộ ratios để tránh bị terminate do rate limit. Script hỗ trợ checkpoint và resume.

Hoặc chạy toàn bộ pipeline:

```bash
python scripts/expanded_hose_hnx_pipeline.py --steps all --sleep 4.5 --retry-sleep 65
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
- **X** = Controls: Gross Margin, Asset Turnover, Current Ratio, Quick Ratio, Log Market Cap và Time FE
- **g(·)** = Hàm phi tuyến (ước lượng bằng Random Forest / LassoCV / ElasticNet)
- **θ** = Hệ số mục tiêu của D/A trong đặc tả DML-PLR

**Tham khảo lý thuyết:** Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). *Double/debiased machine learning for treatment and structural parameters.* The Econometrics Journal, 21(1), C1–C68.

---

## Tác giả

**Vũ Xuân Duy Anh**

Lĩnh vực: Tài chính doanh nghiệp / Kinh tế lượng ứng dụng

---

## Giấy phép

Dự án này được phát triển phục vụ mục đích nghiên cứu học thuật.
