# Mối Liên Hệ giữa Cấu Trúc Vốn và Hiệu Quả Hoạt Động Doanh Nghiệp Niêm Yết tại Việt Nam: Tiếp Cận Double Machine Learning

**Capital Structure and Firm Performance in Vietnam: A Double Machine Learning Approach**

---

## 📊 Tổng quan Kết quả Thực nghiệm (Audited & Fixed)

Nghiên cứu này ước lượng **mối liên hệ có điều kiện** (conditional association) dưới giả định unconfoundedness của cấu trúc vốn lên hiệu quả hoạt động (ROA) của các doanh nghiệp phi tài chính niêm yết trên thị trường chứng khoán Việt Nam giai đoạn 2018–2026. 

Quy trình đã được **đánh giá kiểm soát và sửa đổi toàn diện** để tích hợp các chuẩn mực học thuật:
- Sử dụng cả 2 định nghĩa đòn bẩy: Đòn bẩy tài sản tổng thể (`td_a`) và Đòn bẩy nợ vay có lãi (`ibd_a`).
- Bổ sung biến kiểm soát quy mô doanh nghiệp (`firm_size` = log market cap) và tài sản cố định (`tangibility`).
- Winsorize từng quý (1-99%) để kiểm soát cú sốc vĩ mô động.
- Áp dụng sai số chuẩn cụm 1 chiều (One-way cluster theo DN) và cụm 2 chiều (Two-way cluster theo DN + Quý).
- Thực hiện phân tích Heterogeneous Treatment Effects (HTE) trên phân nhóm Large-cap (Top 100) vs. Mid/Small-cap để chứng minh "thiên lệch thành phần mẫu".

---

## 📈 Kết quả ước lượng chính (Mẫu gộp: 32.029 quan sát)

### 1. Mô hình Baseline truyền thống (Two-way FE, 32 Quarter Dummies)
- **td_a (Total Leverage):** $\theta = -0.060157^{***}$ ($SE = 0.006605$)
- **ibd_a (Financial Leverage):** $\theta = -0.085735^{***}$ ($SE = 0.006930$)

### 2. Mô hình Double Machine Learning (DML-PLR, Random Forest)
- **td_a (Total Leverage) - One-way Cluster:** $\theta = -0.038030^{***}$ ($SE = 0.006106$)
- **td_a (Total Leverage) - Two-way Cluster:** $\theta = -0.037864^{**}$ ($SE = 0.011853$, $p = 0.0014$)
- **ibd_a (Financial Leverage) - One-way Cluster:** $\theta = -0.051032^{***}$ ($SE = 0.006128$)
- **ibd_a (Financial Leverage) - Two-way Cluster:** $\theta = -0.051559^{***}$ ($SE = 0.010772$)

### 3. Phân nhóm HTE (Large-cap vs. Mid/Small-cap)
- **Large-Cap (Top 100):** Tác động gần như triệt tiêu và không có ý nghĩa thống kê ($\theta_{TD/A} = +0.0103, p = 0.573$; $\theta_{IBD/A} = -0.0144, p = 0.343$).
- **Mid/Small-Cap:** Tác động âm rất lớn và có ý nghĩa cao ($\theta_{TD/A} = -0.0416^{***}$; $\theta_{IBD/A} = -0.0530^{***}$).
- *Giải thích:* Thiên lệch thành phần mẫu xuất hiện do nhóm doanh nghiệp vừa và nhỏ chiếm ưu thế tuyệt đối (91% quan sát), kéo kết quả toàn mẫu gộp theo chiều âm, trong khi nhóm Blue-chips ít bị ảnh hưởng bởi đòn bẩy.

---

## 🗂️ Cấu trúc thư mục

```
├── README.md                         # File này
├── requirements.txt                  # Các thư viện Python cần thiết
├── walkthrough.md                    # Hướng dẫn chi tiết công việc đã audit
│
├── data/                             # Dữ liệu nghiên cứu
│   ├── all_nonfinancial_tickers.csv  # Danh sách mã phi tài chính
│   ├── raw_financial_ratios_all.csv  # Dữ liệu chỉ số tài chính thô
│   ├── master_panel_all_firms.csv    # Dữ liệu Master sạch (32.029 obs × 19 cột)
│   └── Data_Documentation.md         # Tài liệu mô tả biến số
│
├── scripts/                          # Mã nguồn Python
│   ├── clean_and_validate.py         # Làm sạch, winsorize và tính toán biến phái sinh
│   ├── run_baseline_expanded.py      # Ước lượng OLS, FE, TWFE cho cả td_a và ibd_a
│   ├── run_dml_expanded.py           # Ước lượng DML cho cả hai loại đòn bẩy + HTE + Cluster SE
│   └── run_all_analyses.py           # Script tổng hợp chạy toàn bộ pipeline
│
└── results/                          # Kết quả đầu ra
    ├── baseline_results.csv          # Bảng kết quả baseline
    ├── dml_results_expanded.csv      # Bảng kết quả DML (bao gồm cả HTE)
    ├── dml_seed_stability.csv        # Bảng kết quả chạy seed stability
    └── figures/
        ├── scatter_lowess_nonlinearity.png       # Đồ thị phi tuyến LOWESS
        └── convergence_diagnostics_expanded.png   # Đồ thị Forest plot phân nhóm & seed sensitivity
```

## 🚀 Cách chạy dự án

Đảm bảo bạn đã cài đặt các thư viện cần thiết (`requirements.txt`):
```bash
pip install -r requirements.txt
```

Để chạy toàn bộ pipeline phân tích và tự động vẽ đồ thị, chạy lệnh:
```bash
python scripts/run_all_analyses.py
```
