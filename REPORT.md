# Báo Cáo Lab MLOps - Day 21: CI/CD cho AI Systems

**Sinh viên:** Hoàng Văn Kiên
**MSSV**: 2A202600077
**GitHub Repo:** https://github.com/Kien-Hoang2692002/Day21-Track2-CI-CD-for-AI-Systems  
**Cloud Provider:** GCP (Project: mlops-lab-2026, Bucket: mlops-lab-bucket-2026, VM: 34.10.249.246)

---

## Bước 1 — Thực Nghiệm Cục Bộ với MLflow

Thực hiện 4 lần thử nghiệm với các bộ siêu tham số khác nhau cho mô hình RandomForestClassifier:

| Lần chạy | n_estimators | max_depth | min_samples_split | Accuracy | F1 Score |
|---|---|---|---|---|---|
| 1 | 100 | 5 | 2 | 0.5640 | 0.5534 |
| 2 | 50 | 3 | 2 | 0.5580 | 0.5185 |
| 3 | 200 | 10 | 5 | 0.6440 | 0.6417 |
| **4 ✓** | **300** | **None** | **2** | **0.6820** | **0.6811** |

**Bộ siêu tham số tốt nhất:** `n_estimators=300, max_depth=None, min_samples_split=2`

**Lý do chọn:**
- `n_estimators=300`: Số cây lớn hơn giúp giảm variance và tăng độ ổn định của mô hình.
- `max_depth=None`: Cho phép cây phát triển đầy đủ đến khi các lá thuần nhất, phù hợp với bài toán phân loại 3 lớp có nhiều đặc trưng tương quan.
- `min_samples_split=2`: Giá trị mặc định, không hạn chế việc phân chia nút — kết hợp với `max_depth=None` cho mô hình linh hoạt nhất.

---

## Bước 2 — Pipeline CI/CD Tự Động

**Kiến trúc:**
- DVC remote: `gs://mlops-lab-bucket-2026/dvc`
- VM: GCE e2-small, Ubuntu 22.04, IP: `34.10.249.246`
- Service: FastAPI + uvicorn, systemd managed

**Kết quả pipeline (4 jobs xanh):**
- Unit Test: 3/3 tests passed (~1m 42s)
- Train: dvc pull → train → upload model (~1m 0s)
- Eval: accuracy 0.682 >= 0.65, PASSED (~3s)
- Deploy: SSH restart service + health check (~32s)

**Kết quả endpoints:**
```
GET  http://34.10.249.246:8000/health  → {"status": "ok"}
POST http://34.10.249.246:8000/predict → {"prediction": 0, "label": "thấp"}
```

**Metrics Bước 2 (2998 mẫu):**
- Accuracy: **0.674**
- F1 Score: **0.673**

---

## Bước 3 — Huấn Luyện Liên Tục

Thêm 2998 mẫu mới từ `train_phase2.csv` vào tập huấn luyện (tổng 5996 mẫu).

Quy trình thực hiện:
```bash
python add_new_data.py          # ghép dữ liệu: 2998 -> 5996 mẫu
dvc add data/train_phase1.csv   # cập nhật DVC pointer
git add data/train_phase1.csv.dvc
git commit -m "data: bo sung 2998 mau du lieu moi (train_phase2)"
dvc push                        # đẩy dữ liệu lên GCS trước
git push origin master          # kích hoạt pipeline tự động
```

Pipeline tự động kích hoạt bởi commit dữ liệu, không cần thao tác thủ công.

**Metrics Bước 3 (5996 mẫu):**
- Accuracy: **0.746**
- F1 Score: **0.745**

**So sánh kết quả:**

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) | Thay đổi |
|---|---|---|---|
| Accuracy | 0.674 | 0.746 | **+7.2%** |
| F1 Score | 0.673 | 0.745 | **+7.2%** |

Thêm dữ liệu giúp mô hình học được nhiều pattern hơn, accuracy vượt ngưỡng 0.70 yêu cầu.

---

## Khó Khăn và Cách Giải Quyết

**1. DVC authentication 401 (Invalid Credentials)**
- Nguyên nhân: `gcsfs` (thư viện DVC dùng để kết nối GCS) không đọc biến môi trường `GOOGLE_APPLICATION_CREDENTIALS` trong GitHub Actions.
- Giải pháp: Dùng `dvc remote modify --local myremote credentialpath /tmp/sa-key.json` để set credentials trực tiếp trong DVC config thay vì dùng env var.

**2. GitHub Actions không trigger khi sửa workflow**
- Nguyên nhân: `paths` filter chỉ bao gồm `data/**.dvc`, `src/**.py`, `params.yaml` — không có `mlops.yml`.
- Giải pháp: Thêm `.github/workflows/mlops.yml` vào danh sách paths trigger.

**3. Health check timeout khi deploy**
- Nguyên nhân: Service cần ~17 giây để download model từ GCS trước khi sẵn sàng, nhưng health check chỉ chờ 5 giây.
- Giải pháp: Tăng `sleep` từ 5s lên 25s và retry interval từ 5s lên 10s.

**4. scikit-learn version mismatch trên VM**
- Nguyên nhân: Model train với scikit-learn 1.4.2 nhưng VM cài 1.7.2.
- Giải pháp: Chỉ là warning, không ảnh hưởng kết quả dự đoán — chấp nhận được cho lab này.
