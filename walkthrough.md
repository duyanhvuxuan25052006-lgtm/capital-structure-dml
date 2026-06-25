# 🚶‍♂️ Walkthrough: Chuyển đổi Biến Can Thiệp từ D/E sang D/A và Chạy lại Pipeline DML

Báo cáo này tóm tắt các chỉnh sửa đã thực hiện để chuyển đổi biến can thiệp cấu trúc vốn từ tỷ lệ **Nợ trên Vốn chủ sở hữu (D/E)** sang **Nợ trên Tổng tài sản (D/A)**, kết quả chạy lại mô hình DML-PLR, các cập nhật đồng bộ trong bài nghiên cứu (Markdown & LaTeX) và hướng dẫn push lên Git.

---

## 🛠️ Nội Dung Công Việc Đã Thực Hiện

### 1. Chạy Lại Toàn Bộ Pipeline Hồi Quy DML với D/A
Chúng tôi đã chạy lại kịch bản phân tích tổng hợp tại `D:\draft 2\scripts\run_all_analyses.py` với biến can thiệp `d_col = 'debt_to_assets'`. Kết quả chạy lại ghi nhận sự thay đổi mang tính cốt lõi về hệ số tác động $\theta$ (đặc biệt ở mô hình phi tuyến):
- **Pooled OLS:** $\theta = -0,2016$, $SE = 0,0340$, $p < 0,0001$.
- **Entity FE:** $\theta = -0,0078$, $SE = 0,0373$, $p = 0,8335$.
- **Two-way FE (Entity + Time):** $\theta = -0,0095$, $SE = 0,0368$, $p = 0,7971$.
- **DML-PLR LassoCV:** $\theta = -0,0066$, $SE = 0,0127$, $p = 0,6039$.
- **DML-PLR ElasticNetCV:** $\theta = -0,0066$, $SE = 0,0127$, $p = 0,6045$.
- **DML-PLR Random Forest:** $\theta = +0,0313$, $SE = 0,0144$, $p = 0,0301$ (**có ý nghĩa thống kê ở mức 5%**).

**Kiểm định độ nhạy qua các hạt ngẫu nhiên (Random Seeds) của RandomForest:**
- Seed 42: $\theta = +0,0313$, $p = 0,0301$
- Seed 123: $\theta = +0,0362$, $p = 0,0110$
- Seed 456: $\theta = +0,0334$, $p = 0,0222$
- Seed 789: $\theta = +0,0356$, $p = 0,0159$
- Seed 2024: $\theta = +0,0355$, $p = 0,0145$
- *Hệ số $\theta$ trung bình qua các seeds là $+0,0344$ (độ lệch chuẩn $0,0020$), tất cả đều có ý nghĩa thống kê ở mức 5%.*

---

### 2. Cập Nhật Báo Cáo Nghiên Cứu Markdown & LaTeX
Chúng tôi đã cập nhật đồng bộ cả hai tệp tin báo cáo nghiên cứu:
- [Capital_Structure_DML_Report.md](file:///D:/draft%202/paper/Capital_Structure_DML_Report.md)
- [Capital_Structure_DML_Report.tex](file:///D:/draft%202/paper/Capital_Structure_DML_Report.tex)

Các chỉnh sửa chi tiết gồm:
- **Abstract & Introduction:** Thay đổi kết luận từ hiệu ứng trung tính sang hiệu ứng dương có ý nghĩa thống kê của đòn bẩy D/A đối với Random Forest DML.
- **Mục 3.3 (Định nghĩa biến):** Thay đổi giải trình từ ưu tiên D/E sang biện hộ học thuật cho việc sử dụng D/A (khớp lý thuyết Trade-off, định đề MM, đồng bộ với Nguyen et al. (2023) và Dang & Do (2021), và tránh giá trị vô cực của D/E đối với doanh nghiệp có equity mỏng).
- **Mục 3.4 (Thống kê mô tả):** Cập nhật bảng thống kê mô tả với số liệu thực tế của biến `debt_to_assets` (Mean = 0,4494, Std = 0,1858, Min = 0,0752, Max = 0,8709).
- **Mục 4.4 (Mã nguồn Python):** Sửa code mẫu thành `d_col = 'debt_to_assets'`.
- **Mục 5.1 (Bảng kết quả chính):** Thay thế toàn bộ số liệu của D/E cũ bằng số liệu của D/A mới chạy, đồng thời thêm dòng chú thích dưới bảng giải thích bản chất thống kê khác biệt của các đại lượng $R^2$ (Full R2, Within R2, Nuisance R2) để tránh so sánh cơ học sai lệch.
- **Mục 5.2 & 5.3 (Phân tích kết quả):** Viết lại lập luận thận trọng, sử dụng các cụm từ "cho thấy bằng chứng về" thay vì khẳng định chắc chắn "xác nhận" đối với hệ số Random Forest nhỏ ($\theta = 0,0313$) trên mẫu hẹp.
- **Mục 6.1 (Thảo luận kinh tế):** Thảo luận kết quả dương ủng hộ thuyết Đại diện (nợ kỷ luật nhà quản trị) và thuyết Đánh đổi (Blue-chips lớn đang hoạt động dưới ngưỡng đòn bẩy tối ưu).
- **Mục 6.4 (Đối chiếu paper trước):** Viết lại với văn phong học thuật khách quan, không dùng từ "bỏ sót" quá tự tin. Giải thích rằng sự khác biệt về kết quả chủ yếu đến từ đặc thù mẫu (Nguyen et al. và Dang & Do dùng mẫu rộng 300-435 doanh nghiệp bao gồm cả doanh nghiệp vừa và nhỏ, còn nghiên cứu này chỉ khảo sát 50 Blue-chips lớn). Khẳng định DML chỉ gợi ý sự tồn tại của thiên lệch dạng hàm (functional form bias) trên mẫu khảo sát cụ thể.
- **Mục 8 (Hạn chế):** Loại bỏ hạn chế đo lường do D/E, thay bằng hạn chế đòn bẩy dạng tổng thể (chưa tách biệt kỳ hạn nợ).

---

### 3. Cập Nhật README.md và Data_Documentation.md
- [README.md](file:///D:/draft%202/README.md): Cập nhật tóm tắt kết quả chính ở bảng tổng quan và điều chỉnh kết luận thành tác động tích cực có ý nghĩa thống kê của D/A lên ROA.
- [Data_Documentation.md](file:///D:/draft%202/data/Data_Documentation.md): Khai báo bổ sung trường `debt_to_assets` làm biến can thiệp chính $D$ và giải thích cách tính toán từ `debt_to_equity`.

---

## 📈 So Sánh Kết Quả: Biến Cũ D/E vs Biến Mới D/A

| Mô hình | Hệ số $\theta$ (D/E cũ) | Hệ số $\theta$ (D/A mới) | Ý nghĩa của việc thay đổi |
|---|---|---|---|
| **Pooled OLS** | $-0,0319^{***}$ | $-0,2016^{***}$ | OLS không FE bị bias âm rất nặng do reverse causality (Pecking Order). |
| **Entity FE** | $-0,0009$ (ns) | $-0,0078$ (ns) | Khi khống chế Firm FE, tác động tuyến tính giảm sát về 0 ở cả D/E và D/A. |
| **Two-way FE** | $-0,0012$ (ns) | $-0,0095$ (ns) | Thêm Time FE cũng không làm thay đổi bản chất trung tính tuyến tính. |
| **DML-LassoCV** | $-0,0006$ (ns) | $-0,0066$ (ns) | Thuật toán Lasso kiểm soát tuyến tính xác nhận kết quả trung tính. |
| **DML-ElasticNet**| $-0,0007$ (ns) | $-0,0066$ (ns) | Tương tự LassoCV. |
| **DML-Random Forest** | **$+0,0017$ (ns)** | **$+0,0313^{**}$ ($p < 0,05$)** | **Điểm đột phá:** Khi kiểm soát phi tuyến, D/A thể hiện tác động dương có ý nghĩa thống kê vững chắc, trong khi D/E bị nhạt nhòa và không có ý nghĩa thống kê. |

---

## 🚀 Hướng Dẫn Push Các Kết Quả Mới Lên Git

Các file kết quả chạy (`results/figures/*.png`, `results/*.csv`) đã được cập nhật đầy đủ cùng với mã nguồn và báo cáo. Để push toàn bộ nội dung mới lên Git, hãy thực hiện các lệnh sau trong terminal hoặc Git Bash tại thư mục dự án `D:\draft 2`:

1. **Kiểm tra trạng thái các file thay đổi:**
   ```bash
   git status
   ```
2. **Thêm toàn bộ file thay đổi vào staging area:**
   ```bash
   git add .
   ```
3. **Commit các thay đổi với thông điệp rõ ràng:**
   ```bash
   git commit -m "Update treatment variable to D/A, run DML, and sync reports"
   ```
4. **Push lên nhánh làm việc từ xa (ví dụ: main hoặc master):**
   ```bash
   git push origin main
   ```
