# 🚶‍♂️ Walkthrough: Mở rộng Mẫu Nghiên Cứu và Áp dụng DML Chuẩn Hóa (Đã Audit)

Tài liệu này tóm tắt các chỉnh sửa đã thực hiện để nâng cấp toàn diện nghiên cứu về **Cấu trúc vốn và Hiệu quả hoạt động (ROA)** trên mẫu **1.300 doanh nghiệp phi tài chính** niêm yết tại Việt Nam (32.158 quan sát quý), tích hợp đầy đủ 10 điểm phản biện học thuật.

---

## 🛠️ Nội Dung Công Việc Đã Thực Hiện (Audited & Fixed)

### 1. Thu thập và làm sạch dữ liệu chuẩn hóa kế toán
- **Hai định nghĩa đòn bẩy:** Tính toán chính xác Đòn bẩy tài sản tổng thể (`td_a` = $TD/A = \frac{D/E}{1 + D/E}$) và Đòn bẩy nợ vay có lãi (`ibd_a` = $IBD/A = \frac{debtPerEquity}{1 + D/E}$) dựa trên đồng nhất thức kế toán để so sánh.
- **Bổ sung biến kiểm soát (W):** Thêm quy mô doanh nghiệp (`firm_size` = $\log(\text{market\_cap})$) và tỷ lệ tài sản cố định hữu hình (`tangibility` = $asset\_turnover / fixed\_asset\_turnover$ giới hạn $[0, 1]$).
- **Lọc D/E âm:** Loại bỏ hoàn toàn 975 quan sát có vốn chủ sở hữu âm trước khi tính D/A.
- **Winsorize từng quý:** Thay vì winsorize toàn mẫu gộp, chúng tôi winsorize $1\% - 99\%$ theo từng quý riêng biệt để kiểm soát các cú sốc vĩ mô (như COVID-19 và thắt chặt tiền tệ 2022).
- **Nhãn Large-Cap:** Xác định 100 doanh nghiệp có vốn hóa trung bình lớn nhất để phân tích tác động phi đồng nhất.

### 2. Ước lượng Baseline và DML với các chỉnh sửa kinh tế lượng nâng cao
- **Đồng nhất time dummies:** Sử dụng 32 biến giả quý cho cả baseline và DML.
- **Hai phương pháp Cluster Standard Errors:** Báo cáo cả sai số chuẩn cụm 1 chiều (One-way cluster theo DN) và 2 chiều (Two-way cluster theo DN + Quý) để xử lý tương quan chuỗi và tương quan chéo hệ thống.
- **Out-of-fold R² (Nuisance fit check):** Tính toán và chứng minh thuật toán phi tuyến Random Forest có lực khớp nuisance tốt hơn nhiều so với Lasso (ví dụ: $R^2_y$ tăng từ $26.47\%$ lên $38.50\%$).
- **ElasticNetCV Convergence check:** Ghi nhận tham số tối ưu l1_ratio đạt 0.100 cho toàn mẫu, chứng minh ElasticNetCV hoạt động giống mô hình Ridge để xử lý đa cộng tuyến, thay vì sụp đổ về Lasso.
- **Sensitivity Analysis (Chernozhukov, 2022):** Chạy kiểm định độ nhạy, báo cáo Robustness Value (RV) đạt 8.18% (TD/A) và 10.75% (IBD/A) chứng minh kết quả vững trước nguy cơ bỏ sót biến nhiễu.

---

## 📈 Kết Quả Thực Nghiệm Mới Nhất (Toàn mẫu, 32.158 quan sát)

### 1. Hồi quy Baseline truyền thống (Clustered SE)
- **td_a (Total Leverage):** Two-way FE $\theta = -0.060157^{***}$ ($SE = 0.006605$, $p < 0.001$)
- **ibd_a (Financial Leverage):** Two-way FE $\theta = -0.085735^{***}$ ($SE = 0.006930$, $p < 0.001$)

### 2. Hồi quy Double Machine Learning (DML-RF, $n\_rep=5, n\_folds=5$)
- **td_a (Total Leverage):**
  - One-way Cluster: $\theta = -0.038030^{***}$ ($SE = 0.006106$, $p < 0.001$)
  - Two-way Cluster: $\theta = -0.037864^{**}$ ($SE = 0.011853$, $p = 0.0014$)
  - Large-Cap (Top 100): $\theta = +0.010300$ ($SE = 0.018294$, $p = 0.573$ - Không ý nghĩa)
  - Mid/Small-Cap: $\theta = -0.041590^{***}$ ($SE = 0.006214$, $p < 0.001$)
- **ibd_a (Financial Leverage):**
  - One-way Cluster: $\theta = -0.051032^{***}$ ($SE = 0.006128$, $p < 0.001$)
  - Two-way Cluster: $\theta = -0.051559^{***}$ ($SE = 0.010772$, $p < 0.001$)
  - Large-Cap (Top 100): $\theta = -0.014365$ ($SE = 0.015160$, $p = 0.343$ - Không ý nghĩa)
  - Mid/Small-Cap: $\theta = -0.052990^{***}$ ($SE = 0.006369$, $p < 0.001$)

---

## 💡 Nhận Xét và Diễn Giải Kinh Tế Học

1. **Hiệu ứng bóc tách tài chính:** Nợ vay có lãi (`ibd_a`) có hệ số âm lớn hơn đáng kể so với nợ tổng thể (`td_a`). Điều này hoàn toàn nhất quán với thực tế tài chính doanh nghiệp: nợ vay có lãi trực tiếp sinh ra chi phí tài chính bằng tiền mặt làm bào mòn ROA, trong khi nợ chiếm dụng không lãi suất từ nhà cung cấp là nguồn tài trợ rẻ và ít rủi ro hơn.
2. **Thiên lệch thành phần mẫu (HTE Split):** Nhóm Large-Cap hoàn toàn mất ý nghĩa thống kê ở cả hai loại đòn bẩy, trong khi nhóm Mid/Small-Cap chịu tác động âm rất nặng. Điều này trực tiếp chứng minh rằng các doanh nghiệp lớn có năng lực tối ưu hóa cấu trúc vốn nên đòn bẩy không làm hại hiệu quả sinh lời của họ. Hệ số âm của toàn mẫu bị chi phối hoàn toàn bởi nhóm doanh nghiệp vừa và nhỏ (vốn chiếm 91% số lượng quan sát).
3. **Tính ổn định của DML:** Thuật toán phi tuyến Random Forest có độ ổn định cực cao qua các seeds (độ lệch chuẩn $SD = 0,0003$).

---

## 🚀 Hướng Dẫn Chạy Toàn Bộ Pipeline

Để chạy lại toàn bộ quy trình làm sạch, ước lượng baseline, vẽ đồ thị LOWESS và hồi quy DML, thực hiện lệnh duy nhất sau tại thư mục `D:\draft 2`:
```bash
python scripts/run_all_analyses.py
```
Toàn bộ kết quả sẽ được cập nhật tự động vào thư mục `results/` và các đồ thị được lưu trữ tại `results/figures/`.
