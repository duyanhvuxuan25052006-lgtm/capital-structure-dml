# Tác Động Của Cấu Trúc Vốn Lên Hiệu Quả Hoạt Động Doanh Nghiệp Tại Việt Nam: Hướng Tiếp Cận Double Machine Learning

**Nghiên cứu ứng dụng Causal Inference (Double Machine Learning) trên dữ liệu bảng khổng lồ của thị trường chứng khoán Việt Nam (HOSE, HNX, UPCoM) giai đoạn 2018–2026.**

---

## Đóng góp cốt lõi của nghiên cứu

Nghiên cứu này vượt ra khỏi các phương pháp kinh tế lượng tuyến tính truyền thống (như OLS hay Fixed Effects), ứng dụng thuật toán **Double Machine Learning (DML)** kết hợp **Random Forest** và **ElasticNet/Lasso** để bóc tách triệt để thiên lệch nội sinh và dạng hàm phi tuyến. 

**3 Phát hiện đột phá:**
1. **Tác động nhân quả ÂM:** Vay nợ có tác động nhân quả làm giảm hiệu quả hoạt động (ROA). Đòn bẩy tài sản (TD_A) tăng 10 điểm phần trăm làm ROA sụt giảm khoảng 0.37 điểm phần trăm.
2. **Bằng chứng vững chắc về Chữ U ngược (Lý thuyết Đánh đổi):** Bằng việc sử dụng Quadratic DML ($Debt + Debt^2$) và đồ thị LOWESS, nghiên cứu chứng minh tồn tại cấu trúc vốn tối ưu. Nợ vay có lợi hoặc trung tính ở mức thấp (tấm chắn thuế), nhưng khi vượt ngưỡng, chi phí kiệt quệ tài chính sẽ tàn phá lợi nhuận (Hệ số $Debt^2$ mang dấu âm có ý nghĩa).
3. **Tính Dị Biệt theo Quy mô (Heterogeneity):** Tác động tiêu cực của nợ giáng đòn nặng nề lên khối doanh nghiệp Vừa và Nhỏ (SMEs), trong khi các "ông lớn" (Large-cap Top 100) có sức chống chịu đòn bẩy xuất sắc (tác động không có ý nghĩa thống kê).

---

## Tổng quan Dữ liệu & Phương pháp

*   **Quy mô dữ liệu:** Toàn bộ doanh nghiệp phi tài chính niêm yết trên 3 sàn (HOSE, HNX, UPCoM). Sau khi làm sạch (Listwise deletion, Winsorization 1-99%), mẫu có **32.029 quan sát (Firm-quarter)**.
*   **Biến số (Variables):**
    *   **Biến kết quả (Y):** `ROA` (Tỷ suất sinh lời trên tài sản).
    *   **Biến can thiệp (D):** `TD_A` (Tổng Nợ/TS) và `IBD_A` (Nợ sinh lãi/TS).
    *   **Biến kiểm soát (X):** Firm Size, Current Ratio, Tangibility, Gross Margin, v.v.
*   **Phương pháp:** Partially Linear Model với DML. Cross-fitting 5-Fold lặp lại 3 lần (N_REP=3) để khử sai số mẫu. Nuisance models sử dụng Random Forest (`max_depth=8`) để tránh overfitting.

---

## Kết quả ước lượng (Final Results)

Bảng kết quả lõi trích xuất từ mô hình DML Random Forest (với One-way/Two-way Cluster Standard Errors):

| Biến Can thiệp (Treatment) | Đặc tả Mô hình (Specification) | Hệ số $\theta$ | P-value | Ý nghĩa thống kê |
| :--- | :--- | :--- | :--- | :--- |
| **Nợ Tổng (td_a)** | Full Sample (Mô hình chính) | **-0.0376** | `< 0.001` | Rất có ý nghĩa (1%) |
| **Nợ Tổng (td_a)** | Loại bỏ sàn UPCoM | **-0.0455** | `< 0.001` | Kết quả bền vững |
| **Nợ Trễ (td_a_lag)** | Chống Nhân quả ngược | **-0.0195** | `< 0.01` | Có ý nghĩa mạnh |
| **Nợ Ảo (Placebo)** | Random permutation | `-0.0020` | `0.288` | **Không có ý nghĩa** (Mô hình không bị nhiễu ảo) |
| **Nợ Tổng (td_a)** | Nhóm Large-Cap | `0.0101` | `0.573` | Không có ý nghĩa |
| **Nợ Tổng (td_a)** | Nhóm Mid/Small-Cap | **-0.0403** | `< 0.001` | Rất có ý nghĩa (1%) |
| **Nợ Sinh lãi (ibd_a)** | Full Sample | **-0.0508** | `< 0.001` | Rất có ý nghĩa (1%) |

*(Test độ ổn định hạt giống - Seed Stability: Chạy lại mô hình RF với 5 random seeds khác nhau đều cho ra $\theta$ dao động cực hẹp trong biên độ `[-0.036, -0.037]`, khẳng định tính bền vững tuyệt đối).*

---

## Cấu trúc thư mục

```text
├── README.md                           # File thông tin dự án
├── results/                            # Báo cáo và Kết quả đầu ra
│   ├── Final_Academic_Report_DML.md    # BÁO CÁO HỌC THUẬT CHI TIẾT (File quan trọng nhất)
│   ├── dml_results_expanded.csv        # Kết quả 20 mô hình DML
│   ├── baseline_results.csv            # Kết quả OLS / Fixed Effects
│   ├── dml_seed_stability.csv          # Kết quả test độ bền Seed
│   └── figures/
│       ├── lowess_tda_roa.png          # Đồ thị LOWESS chứng minh phi tuyến (U ngược)
│       └── convergence_diagnostics_expanded.png # Forest Plot & Seed Stability
├── data/                               # Dữ liệu
│   └── master_panel_all_firms.csv      # File data chính (32.029 obs)
└── scripts/                            # Mã nguồn phân tích (Python)
    └── run_local_safe.py               # Script DML chạy chính thức
```

## Hướng dẫn chạy lại toàn bộ quy trình (Reproducibility)

Để tái lập 100% kết quả nghiên cứu (lưu ý: quá trình huấn luyện >40.000 cây quyết định mất khoảng 3-4 giờ trên CPU thông thường):

1. Cài đặt các thư viện lõi:
```bash
pip install pandas numpy scikit-learn doubleml statsmodels matplotlib seaborn
```

2. Chạy Pipeline chính:
```bash
python scripts/run_local_safe.py
```
Toàn bộ kết quả `.csv` và biểu đồ `.png` sẽ tự động được xuất ra thư mục `results/`.
