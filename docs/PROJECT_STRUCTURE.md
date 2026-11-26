# Cấu trúc thư mục dự án (Project Structure)

> Tài liệu này đồng bộ trực tiếp với [plan.md](./plan.md) (Mục 3, 6, 7, 8) — nguồn chuẩn duy nhất của dự án.

```text
Customer-Churn-Prediction-and-Behavioral-Pattern-Mining/
├── plan.md                              # Nguồn chuẩn duy nhất: quyết định D1-D10, quy ước code, phân công
├── problem_definition.md                # Định nghĩa bài toán, mốc thời gian T0, cửa sổ churn, câu hỏi nghiên cứu
├── pattern_mining_approach.md           # Quy trình khai phá mẫu, tham số FP-Growth, kết quả pilot
├── ai_usage_log.md                      # Nhật ký sử dụng AI theo quy tắc R8
├── README.md                            # Giới thiệu dự án và hướng dẫn chạy
├── PROJECT_STRUCTURE.md                 # Sơ đồ và quy ước thư mục (file này)
├── requirements.txt                     # Danh sách thư viện Python (Python 3.10+)
├── .gitignore                           # Bỏ qua cache, venv, checkpoints, model binaries (*.pkl)
├── .gitattributes                       # Chuẩn hóa xuống dòng LF
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dữ liệu gốc (7.043 dòng) - KHÔNG SỬA
│   └── processed/
│       └── clean_data.csv                         # Dữ liệu sau làm sạch (7.043 dòng; lọc tenure > 0 khi phân tích → 7.032 dòng)
│
├── notebooks/                           # Chạy tuần tự từ trên xuống sau khi Restart Kernel
│   ├── 01_data_cleaning.ipynb           # Làm sạch dữ liệu, kiểm tra kiểu và giá trị khuyết
│   ├── 02_eda_demographics_churn.ipynb  # EDA nhân khẩu học, Chi-square, Cramér's V (dùng tập gốc 7.043 dòng)
│   ├── 03_eda_services_contract.ipynb   # EDA dịch vụ, hợp đồng, phương thức thanh toán
│   ├── 04_tms_cluster.ipynb             # Phân cụm khách hàng theo bộ đặc trưng T-M-S
│   ├── 05_pattern_mining.ipynb          # Luật kết hợp FP-Growth, contrast patterns, subgroup discovery
│   ├── 06_models.ipynb                  # Huấn luyện và đánh giá mô hình (LR, DT, RF, XGBoost, LightGBM)
│   └── 07_shap.ipynb                    # Giải thích mô hình bằng SHAP và đánh giá luật làm đặc trưng
│
├── src/                                 # Module Python dùng chung (snake_case)
│   ├── data_loader.py                   # Nạp và làm sạch dữ liệu (RAW_DATA_PATH, PROCESSED_DATA_PATH, load_clean_data)
│   ├── config.py                        # Hằng số, random_state=42, ngưỡng lọc, mốc rời rạc hóa
│   ├── features.py                      # Tạo đặc trưng T-M-S, n_addon, chia train/test (70/30 stratified)
│   ├── mining.py                        # Tạo giao dịch, chạy FP-Growth, lọc minimum improvement, kiểm tra holdout
│   ├── models.py                        # Pipeline huấn luyện, Stratified 5-Fold CV, tuning, tính PR-AUC / Lift@20%
│   └── viz.py                           # Hàm vẽ biểu đồ chuẩn hóa dùng chung
│
├── models/                              # Lưu mô hình đã huấn luyện (*.pkl, *.joblib - bị chặn bởi .gitignore)
│
├── reports/                             # Đầu ra tự động từ code và notebook
│   ├── figures/                         # Hình ảnh biểu đồ (ten_mo_ta.png)
│   └── tables/                          # Bảng kết quả, danh sách luật, bảng so sánh mô hình (ten_mo_ta.csv)
│
└── report/                              # Tài liệu bàn giao cuối kỳ (báo cáo học thuật, slide thuyết trình)
```

---

## Quy ước cốt lõi theo thư mục

1. **`data/`**:
   - `data/raw/` là **Read-Only**.
   - `data/processed/clean_data.csv` giữ 7.043 dòng. Mọi notebook phân tích, khai phá mẫu và mô hình (trừ ngoại lệ mô tả tổng quan ở `02_eda_demographics_churn.ipynb`) bắt buộc lọc `tenure > 0` ngay đầu file để lấy **tập phân tích 7.032 dòng** (`1.869` churn).
2. **`src/` & `notebooks/`**:
   - Không hardcode đường dẫn hoặc hằng số trong notebook; import từ `src/data_loader.py` và `src/config.py`.
   - Mọi bước học từ dữ liệu (scaler, encoder, tam phân vị `MonthlyCharges`, SMOTE, khai phá luật làm đặc trưng) chỉ **fit trên tập Train (70%)**, tuyệt đối không fit trên toàn bộ dữ liệu trước khi chia.
3. **`reports/` vs `report/`**:
   - `reports/figures/` và `reports/tables/` chứa biểu đồ `.png` và bảng `.csv` xuất trực tiếp từ code.
   - `report/` chứa file báo cáo tổng kết và slide thuyết trình của nhóm.
