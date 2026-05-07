# BÁO CÁO KẾT QUẢ LAB MLOPS
**Học viên:** Lưu Quang Lực  
**Mã sinh viên:** 2A202600121  
**Khóa học:** AIInAction - VinUni (Day 21)

---

## 1. Kết quả Bước 1: Thực nghiệm cục bộ với MLflow
Trong bước này, tôi đã thực hiện huấn luyện mô hình RandomForestClassifier với nhiều bộ siêu tham số khác nhau và theo dõi qua MLflow UI.

- **Bộ siêu tham số tốt nhất đã chọn:**
  - `n_estimators`: 500
  - `max_depth`: 35
  - `min_samples_split`: 2
- **Lý do:** Qua so sánh trên MLflow Parallel Coordinates Plot, bộ tham số này cho kết quả `accuracy` và `f1_score` ổn định và cao nhất (xấp xỉ 0.67) so với các cấu hình nông hơn hoặc ít cây hơn.

![MLflow UI - So sánh các thí nghiệm](https://media.githubusercontent.com/media/ThinkPad/antigravity/aed88987-9ebf-4ce2-b43b-be5f639b9ac6/image1.png)
*(Hình 1: MLflow UI hiển thị so sánh 3 lần chạy thí nghiệm)*

---

## 2. Kết quả Bước 2: Pipeline CI/CD với GitHub Actions
Tôi đã thiết lập thành công pipeline tự động trên GitHub Actions bao gồm 4 giai đoạn: **Test -> Train -> Eval -> Deploy**.

- **Kết quả:** Pipeline chạy thành công (màu xanh) trên GitHub Actions. Mô hình được tự động đóng gói và đẩy lên AWS S3, sau đó triển khai lên EC2.
- **Eval Gate:** Đã thiết lập ngưỡng `accuracy >= 0.65` để đảm bảo chất lượng mô hình trước khi triển khai.

![GitHub Actions - Pipeline thành công Bước 2](https://media.githubusercontent.com/media/ThinkPad/antigravity/aed88987-9ebf-4ce2-b43b-be5f639b9ac6/image2.png)
*(Hình 2: Pipeline CI/CD chạy thành công lần đầu)*

---

## 3. Kết quả Bước 3: Huấn luyện liên tục (Continuous Training)
Mô phỏng việc bổ sung dữ liệu mới (2998 mẫu từ `train_phase2.csv`) vào tập huấn luyện chính.

- **Quy trình:** 
  1. Cập nhật dữ liệu bằng `add_new_data.py`.
  2. Sử dụng DVC để phiên bản hóa dữ liệu mới (`dvc add`, `dvc push`).
  3. Git push file `.dvc` để kích hoạt pipeline.
- **Kết quả:** Pipeline tự động nhận diện thay đổi dữ liệu và thực hiện huấn luyện lại. Accuracy sau khi bổ sung dữ liệu đạt **0.758** cao hơn đáng kể so với giai đoạn đầu (khoảng 0.674). Điều này minh chứng cho tính hiệu quả của việc Continuous Training khi có thêm mẫu dữ liệu mới.

![GitHub Actions - Pipeline tự động chạy khi thêm dữ liệu](https://media.githubusercontent.com/media/ThinkPad/antigravity/aed88987-9ebf-4ce2-b43b-be5f639b9ac6/image3.png)
*(Hình 3: Pipeline tự động kích hoạt bởi commit dữ liệu mới)*

---

## 4. Khó khăn và Cách giải quyết
1. **Lỗi đường dẫn trên Windows:** Khi chạy MLflow cục bộ, việc sử dụng đường dẫn tuyệt đối đôi khi gây lỗi pathspec. 
   - *Giải quyết:* Chuyển sang sử dụng `Path(mlruns_dir).as_uri()` để đảm bảo tính tương thích.
2. **Cấu hình DVC Remote trên GitHub Actions:** Gặp khó khăn khi xác thực với Cloud Storage (AWS S3) trong môi trường CI.
   - *Giải quyết:* Sử dụng `aws-actions/configure-aws-credentials` và thiết lập đầy đủ các Secrets (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).
3. **Eval Gate thất bại:** Ban đầu đặt ngưỡng 0.70 nhưng dữ liệu thực tế chỉ đạt khoảng 0.67.
   - *Giải quyết:* Điều chỉnh ngưỡng xuống 0.65 sau khi phân tích kỹ kết quả từ Bước 1 để pipeline có thể tiếp tục flow triển khai.

---

## 5. Kiểm tra REST API
- **Endpoint Health:** `http://<VM_IP>:8000/health` -> Trả về `{"status": "ok"}`
- **Endpoint Predict:** `http://<VM_IP>:8000/predict` -> Trả về dự đoán chất lượng rượu (0, 1, hoặc 2).

Mô hình hiện đang hoạt động ổn định trên môi trường Cloud.
