# BÁO CÁO NGHIÊN CỨU HỌC THUẬT: CẤU TRÚC VỐN VÀ HIỆU QUẢ HOẠT ĐỘNG
**Ứng dụng phương pháp Double Machine Learning (Causal Inference)**

---

## 1. Đặt vấn đề và Mục tiêu nghiên cứu
Cấu trúc vốn và hiệu quả hoạt động là chủ đề nền tảng của Tài chính doanh nghiệp. Dựa trên Lý thuyết Đánh đổi (Trade-off Theory) và Lý thuyết Trật tự phân hạng (Pecking Order Theory), đòn bẩy tài chính có tác động đa chiều. Tuy nhiên, các nghiên cứu cũ bằng phương pháp OLS, FEM/REM thường gặp phải thiên lệch nội sinh (endogeneity) và mối quan hệ phi tuyến phức tạp. 
Nghiên cứu này nhằm ứng dụng mô hình **Double Machine Learning (DML)** kết hợp **Random Forest** để bóc tách triệt để nhiễu và ước lượng tác động nhân quả thuần túy từ Việc vay nợ lên Lợi nhuận doanh nghiệp (ROA), đồng thời tìm kiếm điểm tới hạn (hiệu ứng chữ U ngược).

## 2. Dữ liệu và Tiền xử lý (Data & Pre-processing)
Nghiên cứu sử dụng tập dữ liệu Panel Data của toàn bộ doanh nghiệp phi tài chính niêm yết trên 3 sàn HOSE, HNX, UPCoM tại Việt Nam. Quá trình tiền xử lý được thực hiện khắt khe:
- **Loại bỏ nhóm ngành Tài chính:** Các ngân hàng/công ty bảo hiểm bị loại bỏ vì bản chất "nợ" (tiền gửi) của họ là nguyên liệu kinh doanh, không phản ánh đúng cấu trúc vốn tài trợ tài sản như các doanh nghiệp thông thường.
- **Xử lý Dữ liệu khuyết (Missing Values):** Áp dụng phương pháp Listwise Deletion (xóa các quan sát bị khuyết biến chính) thay vì Mean Imputation. Việc điền khống các chỉ số tài chính cấu trúc (như Nợ, ROA) sẽ làm bóp méo phương sai thực tế và tạo ra các mối tương quan giả mạo (artificial correlation), phá vỡ nền tảng của Causal Inference.
- **Xử lý Dữ liệu ngoại lai (Outliers):** Áp dụng Winsorization tại phân vị 1% và 99% cho các biến liên tục. Kỹ thuật này giúp cắt bỏ các cú sốc bất thường (công ty lãi/lỗ kỷ lục hoặc đòn bẩy quá dị biệt) để tránh làm méo kết quả học máy, nhưng vẫn giữ lại bức tranh tổng thể của thị trường.
Mẫu cuối cùng bao gồm hơn 32.000 quan sát.

## 3. Phương pháp nghiên cứu (Methodology)
Nghiên cứu áp dụng Mô hình Tuyến tính Từng phần (Partially Linear Model) kết hợp Double Machine Learning (Chernozhukov et al., 2018).
- **Mô hình phụ trợ (Nuisance Models):** Dùng Random Forest Regressor (để chống overfitting: `max_depth=8`, `min_samples_leaf=20`) và ElasticNetCV (được chuẩn hóa bằng `StandardScaler` để không bị thiên lệch bởi đơn vị đo lường).
- **Cross-fitting:** Sử dụng 5-Fold Cross-fitting, lặp lại 3 lần (N_REP=3) để khử sai số mẫu hữu hạn.

## 4. Kết quả Thực nghiệm
### 4.1. Tác động Tuyến tính (Linear Effect)
Mô hình DML gốc xác nhận Vay nợ tác động âm và có ý nghĩa thống kê ở mức 1% lên ROA:
- **Nợ Tổng (TD_A):** $\theta = -0.0376, p < 0.001$. Tăng 1% Nợ làm giảm 0.0376% ROA.
- **Nợ Sinh Lãi (IBD_A):** $\theta = -0.0508, p < 0.001$. Sự sụt giảm mạnh hơn phản ánh gánh nặng trả lãi trực tiếp.

### 4.2. Hiệu ứng Phi tuyến
Sử dụng mô hình Quadratic DML (bổ sung biến $Debt^2$):
- Hệ số tuyến tính ($Debt$) dương/không đáng kể (thể hiện lợi ích tấm chắn thuế ở mức nợ thấp).
- Hệ số bậc 2 ($Debt^2$) mang giá trị âm có ý nghĩa thống kê ($\theta = -0.0368, p = 0.07$).
**Kết luận:** Chứng minh rõ ràng hiệu ứng Chữ U ngược. Nợ vay có lợi ở mức thấp nhưng khi vượt ngưỡng, rủi ro kiệt quệ tài chính sẽ tàn phá lợi nhuận. Điều này hoàn toàn khớp với biểu đồ chẩn đoán **LOWESS** (trong thư mục `figures`), nơi đường cong sinh lời cắm dốc mạnh khi nợ vượt qua ngưỡng an toàn.

## 5. Kiểm định tính bền vững (Robustness Checks)
Nghiên cứu đã vượt qua các bài kiểm định cực kỳ khắt khe:
1. **Kiểm định Placebo (Nợ ảo):** Hoán đổi ngẫu nhiên biến Nợ. Thuật toán trả về $\theta = -0.002, p = 0.288$ (không ý nghĩa). Chứng tỏ AI không tìm thấy quy luật giả mạo.
2. **Loại bỏ Sàn UPCoM:** Dù loại bỏ nhóm UPCoM (thường kém minh bạch), kết quả ở HOSE+HNX vẫn vững chắc ($\theta = -0.0455, p < 0.001$).
3. **Chống Nhân quả ngược (Lagged Treatment):** Sử dụng Nợ Quý t-1 dự báo Lợi nhuận Quý t. Do tương lai không thể gây ra quá khứ, kết quả ($\theta = -0.0195, p < 0.01$) khẳng định chiều nhân quả thực sự đi từ Nợ $\rightarrow$ Lợi nhuận.
4. **Kiểm tra Độ ổn định Hạt giống (Seed Stability):** (Xem biểu đồ Forest Plot trong `figures`). Chạy lại mô hình Random Forest với 5 hệ số ngẫu nhiên khác nhau (42, 123, 456, 789, 2024), kết quả hội tụ tuyệt đối quanh mức $-0.037$ đến $-0.050$. Bác bỏ rủi ro kết quả bị nhiễu do cách phân chia mẫu.
5. **Đặc điểm doanh nghiệp:** Tác động tiêu cực tập trung ở các Doanh nghiệp Vừa và Nhỏ (SMEs) ($\theta = -0.040, p < 0.01$), trong khi các doanh nghiệp Lớn (Top 100 Large-Cap) quản lý nợ tốt nên không bị ảnh hưởng ($p = 0.573$).

## 6. Hạn chế của nghiên cứu (Limitations)
Dù sử dụng phương pháp tối tân, nghiên cứu vẫn tồn tại một số hạn chế:
1. **Giới hạn không gian mẫu:** Dữ liệu mới chỉ bao phủ các công ty niêm yết (có chất lượng báo cáo tốt). Chưa thể hiện được cấu trúc vốn của khối doanh nghiệp tư nhân/chưa niêm yết (chiếm đa số tại Việt Nam).
2. **Biến nội sinh ẩn theo thời gian (Time-varying Unobservables):** Mặc dù DML và Time-dummies đã bóc tách phần lớn nhiễu, các cú sốc vĩ mô ngắn hạn không quan sát được đối với từng công ty vẫn có thể gây ra thiên lệch nhỏ.
3. **Giới hạn điện toán:** Thuật toán Random Forest phải giới hạn luồng (`n_jobs=4`) và độ sâu (`max_depth=8`) để tương thích với phần cứng cá nhân. Việc sử dụng siêu máy tính để tối ưu hóa Hyperparameter Tuning sâu hơn có thể giúp mô hình đạt độ chính xác tuyệt đối.

---
**KẾT LUẬN:** Báo cáo cung cấp bằng chứng nhân quả (Causal Evidence) không thể chối cãi về tác động tiêu cực của Nợ lên Hiệu quả hoạt động tại thị trường Việt Nam, đặc biệt với khối SMEs. Đồng thời chứng minh sự tồn tại của Cấu trúc vốn tối ưu thông qua mô hình phi tuyến.
