# plan.md - Customer Churn Prediction and Behavioral Pattern Mining

> **BẮT BUỘC ĐỌC** trước khi viết code, viết báo cáo hoặc làm việc với bất kỳ công cụ AI nào (ChatGPT, Claude, Copilot, Cursor, Gemini...).
> File này là **nguồn chuẩn duy nhất** của dự án. Khi file này mâu thuẫn với tài liệu khác, file này đúng.
> Chỉ sửa file này qua pull request vào `develop` có ít nhất 1 thành viên khác duyệt. Ghi lại thay đổi ở mục 12.

---

## 0. Cách dùng file này

**Với người:** đọc mục 1-8 một lần khi bắt đầu, đọc lại mục 7 (tiến độ) mỗi tuần.

**Với AI:** mỗi phiên làm việc mới, đưa cho AI file này **trước** khi giao việc, kèm tài liệu liên quan. Dùng mẫu prompt ở mục 10.

**Để công cụ AI tự đọc file:** Claude Code đọc `CLAUDE.md`, Cursor / Codex đọc `AGENTS.md`, GitHub Copilot đọc `.github/copilot-instructions.md`. Tạo các file đó với đúng một dòng:

```
Đọc và tuân thủ toàn bộ nội dung file docs/plan.md trước khi làm bất cứ việc gì.
```

---

## 1. Tóm tắt dự án

| Mục | Nội dung |
|---|---|
| Tên đề tài | Customer Churn Prediction and Behavioral Pattern Mining |
| Môn học | Phân tích và khai phá dữ liệu - UET, học kỳ I 2026-2027 |
| Repo | https://github.com/buiquanghung2711/Customer-Churn-Prediction-and-Behavioral-Pattern-Mining |
| Nhóm | 4 thành viên (xem mục 7) |
| Dữ liệu | Telco Customer Churn (IBM sample), 7.043 khách × 21 cột |
| Mục tiêu 1 | **Khai phá hành vi:** tìm tổ hợp đặc điểm gắn với rủi ro rời bỏ cao (luật kết hợp, contrast patterns, subgroup discovery, phân cụm) |
| Mục tiêu 2 | **Dự đoán churn:** mô hình phân loại + giải thích bằng SHAP |
| Đầu ra cuối | Báo cáo, slide, notebook chạy lại được, danh sách khách rủi ro cao kèm đề xuất hành động |
| Khung quy trình | CRISP-DM |

Tài liệu chi tiết (trong thư mục `docs/`):
- `docs/problem_definition.md` - định nghĩa bài toán, mốc thời gian, nhãn churn, câu hỏi nghiên cứu
- `docs/pattern_mining_approach.md` - quy trình khai phá mẫu, tham số, kết quả chạy thử
- `docs/PROJECT_STRUCTURE.md` - sơ đồ và quy ước thư mục dự án
- `docs/ai_usage_log.md` - nhật ký sử dụng AI theo quy tắc R8

---

## 2. Quyết định đã chốt

**Không được thay đổi** các quyết định dưới đây khi chưa họp nhóm. Nếu AI đề xuất làm khác, **từ chối** và giữ theo bảng này.

| Mã | Quyết định | Lý do ngắn |
|---|---|---|
| D1 | Chỉ dùng bộ **Telco Customer Churn** (`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`) | Nhóm đã chốt |
| D2 | Nhãn = cột `Churn` (khách rời đi trong tháng cuối trước ngày trích xuất T0) | Định nghĩa gốc của bộ dữ liệu |
| D3 | **Hai lớp dữ liệu** (xem mục 3): `clean_data.csv` giữ đủ 7.043 dòng; **mọi phân tích, khai phá mẫu và mô hình lọc `tenure > 0`** → 7.032 khách | 11 khách `tenure = 0` mới gia nhập, chưa có lịch sử, không thể có nhãn churn có nghĩa |
| D4 | Chia **ngẫu nhiên có phân tầng 70/30**, CV **stratified 5-fold** trên train | Dữ liệu cắt ngang, không có trục thời gian. **Không** chia theo thời gian |
| D5 | Dùng bộ **T-M-S** (Tenure, MonthlyCharges, số dịch vụ cộng thêm) thay cho RFM | Không có giao dịch để tính Recency, Frequency |
| D6 | `random_state = 42` ở mọi nơi | Tái lập kết quả |
| D7 | Chỉ số mô hình chính: **PR-AUC**; phụ: Recall lớp churn, F1, ROC-AUC, Lift@20% | Nhãn mất cân bằng (26,6%) |
| D8 | Rời rạc hóa: tenure `0-6, 7-12, 13-24, 25-48, 49-72`; MonthlyCharges tam phân vị fit trên train | Xem `docs/pattern_mining_approach.md` mục 3 |
| D9 | Khai phá luật: FP-Growth, `min_support=0.03`, `max_len=4`, `min_confidence=0.5`, `lift≥1.5`, minimum improvement `δ=0.02`, kiểm tra trên holdout | Đã chạy thử, 88% luật giữ vững trên holdout |
| D10 | Không dùng `customerID` làm đặc trưng; không dùng trực tiếp `TotalCharges` cho khai phá mẫu | Định danh / gần trùng với `tenure × MonthlyCharges` |

---

## 3. Luồng dữ liệu

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv        7.043 dòng   KHÔNG sửa file này
        │  src/data_loader.py  →  clean_data()
        │  - TotalCharges: chuỗi → số, 11 ô rỗng điền bằng MonthlyCharges
        │  - bỏ trùng, ép kiểu
        ▼
data/processed/clean_data.csv                         7.043 dòng   lớp làm sạch
        │  lọc tenure > 0   (D3 - bắt buộc ở đầu mọi notebook phân tích / mô hình)
        ▼
Tập phân tích                                         7.032 dòng   1.869 churn (26,58%)
        │  train_test_split(stratify=y, test_size=0.3, random_state=42)
        ▼
Train 70%  /  Test 30%
```

Đoạn code chuẩn ở đầu mọi notebook phân tích (theo cách import của `01_data_cleaning`, notebook chạy từ thư mục `notebooks/`):

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
from data_loader import load_clean_data

df = load_clean_data()
df = df[df["tenure"] > 0].reset_index(drop=True)   # D3
assert len(df) == 7032 and (df["Churn"] == "Yes").sum() == 1869
```

> Ngoại lệ: notebook EDA mô tả toàn bộ dữ liệu (`02_eda_demographics_churn`) được dùng đủ 7.043 dòng, nhưng phải ghi rõ trong notebook là đang dùng tập gốc.

---

## 4. Số liệu chuẩn

Dùng **đúng** các con số này để kiểm tra code và viết báo cáo. Nếu kết quả của bạn khác, pipeline đang có lỗi.

| Số liệu | Giá trị |
|---|---|
| Số dòng gốc / sau làm sạch | 7.043 |
| Số dòng tập phân tích (`tenure > 0`) | **7.032** |
| Số khách churn | 1.869 |
| Tỷ lệ churn nền | **26,58%** (tập phân tích) / 26,54% (tập gốc) |
| Tỷ lệ churn: Month-to-month / One year / Two year | 42,7% / 11,3% / 2,8% |
| Tỷ lệ churn: Fiber optic / DSL / No internet | 41,9% / 19,0% / 7,4% |
| Tỷ lệ churn: Electronic check | 45,3% |
| Tỷ lệ churn: tenure 0-6 tháng | 52,9% |
| Tỷ lệ khách churn có tenure ≤ 12 | 55,5% |

---

## 5. Thuật ngữ thống nhất

Dùng **cùng một từ** cho cùng một khái niệm trong code, báo cáo và slide.

| Tiếng Việt (báo cáo) | Tiếng Anh / tên trong code | Nghĩa |
|---|---|---|
| Rời bỏ | churn, `y = 1` | Khách chấm dứt dịch vụ trong cửa sổ churn |
| Ngày trích xuất | T0, snapshot date | Thời điểm dữ liệu được ghi nhận |
| Cửa sổ quan sát | observation window | Từ ngày gia nhập đến T0, dài bằng `tenure` |
| Cửa sổ churn | churn window, label window | (T0 - 1 tháng, T0] |
| Dữ liệu cắt ngang | cross-sectional | Mỗi khách một dòng, ghi trạng thái tại một thời điểm, không có lịch sử theo tháng |
| Tập phân tích | analysis set | 7.032 khách sau khi lọc `tenure > 0` |
| Rời bỏ sớm / muộn | `early_churn` / `late_churn` | Churn với tenure ≤ 12 / > 12. **Chỉ để phân tích, không làm đặc trưng** |
| Dịch vụ cộng thêm | add-on, `n_addon` | 6 dịch vụ: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Luật kết hợp có lớp đích | class association rule | Luật có vế phải là `Churn=Yes` hoặc `Churn=No` |
| Lọc trùng | minimum improvement | Bỏ luật không tốt hơn luật tổng quát hơn của nó |
| Rò rỉ dữ liệu | data leakage | Dùng thông tin mô hình không được biết lúc dự đoán |

---

## 6. Cấu trúc repo

Đánh dấu: ✓ đã có trên một nhánh nào đó · ○ sẽ tạo.

```
Customer-Churn-Prediction-and-Behavioral-Pattern-Mining/
├── docs/
│   ├── plan.md                          ✓ develop (file này)
│   ├── problem_definition.md            ✓ develop
│   ├── pattern_mining_approach.md       ✓ develop
│   ├── PROJECT_STRUCTURE.md             ✓ develop
│   └── ai_usage_log.md                  ✓ develop
├── README.md                            ✓
├── requirements.txt                     ✓ develop (đã bổ sung đầy đủ thư viện, mục 8)
├── .gitignore                           ✓ develop
├── data/
│   ├── raw/WA_Fn-UseC_-Telco-Customer-Churn.csv     ✓ KHÔNG sửa
│   └── processed/clean_data.csv                     ✓ develop
├── notebooks/
│   ├── 01_data_cleaning.ipynb           ✓ develop
│   ├── 02_eda_demographics_churn.ipynb  ✓ feat/eda-visualization
│   ├── 03_eda_services_contract.ipynb   ○
│   ├── 04_tms_cluster.ipynb             ○
│   ├── 05_pattern_mining.ipynb          ○
│   ├── 06_models.ipynb                  ○
│   └── 07_shap.ipynb                    ○
├── src/
│   ├── data_loader.py                   ✓ develop - load, clean, đường dẫn
│   ├── config.py                        ○ seed, ngưỡng, mốc rời rạc hóa
│   ├── features.py                      ○ T-M-S, n_addon, chia train/test
│   ├── mining.py                        ○ transactions, rules, lọc trùng, holdout
│   ├── models.py                        ○ train, tune, evaluate
│   └── viz.py                           ○ hàm vẽ dùng chung
├── models/                              ✓ mô hình đã huấn luyện (*.pkl đã bị .gitignore)
├── reports/
│   ├── figures/                         ✓ feat/eda-visualization & develop
│   ├── tables/                          ✓ develop - bảng luật, so sánh mô hình
│   └── ...                              ✓ develop - tài liệu báo cáo cuối, slide
└── results/                             ✓ develop - đầu ra thí nghiệm, danh sách khách rủi ro cao
```

Số thứ tự notebook từ 03 trở đi là đề xuất. Ai tạo notebook mới thì dùng số tiếp theo chưa có và cập nhật cây thư mục ở trên.

---

## 7. Phân công và tiến độ

**Ngày bắt đầu tuần 1:** `26/11/2026`

| Thành viên | Vai trò | Nhánh / phần sở hữu |
|---|---|---|
| **Hưng** | Trưởng nhóm & ML Lead | `docs/plan.md`, duyệt PR, `feat/modeling`, `src/models.py`, `06_models.ipynb`, `07_shap.ipynb`, đánh giá kinh doanh & báo cáo tổng |
| **Hiếu** | Data & Feature Engineer | `feat/data-loader-cleaning`, `feat/feature-engineering`, `src/data_loader.py`, `src/config.py`, `src/features.py`, `04_tms_cluster.ipynb` |
| **Hòa** | EDA, Visualization & Slide Lead | `feat/eda-visualization`, `02_eda_demographics_churn.ipynb`, `03_eda_services_contract.ipynb`, `src/viz.py`, chuẩn hóa biểu đồ & Slide |
| **Kiên** | Pattern Mining & Research Lead | `feat/pattern-mining`, `docs/problem_definition.md`, `docs/pattern_mining_approach.md`, `src/mining.py`, `05_pattern_mining.ipynb` |

| Phần việc | Nhánh | Trạng thái |
|---|---|---|
| Khởi tạo repo, cấu trúc thư mục | `main` | Xong |
| Làm sạch dữ liệu, `data_loader.py`, `01_data_cleaning` | `feat/data-loader-cleaning` | Xong, đã gộp vào `develop` |
| EDA nhân khẩu học, kiểm định Chi-square, Cramér's V | `feat/eda-visualization` | Xong, **chưa gộp** vào `develop` |
| Định nghĩa bài toán, cách tiếp cận khai phá mẫu | `feat/pattern-mining` | Xong, đã gộp vào `develop` (`docs/`) |
| `docs/plan.md` & đồng bộ `docs/` | `develop` | Xong, đã gộp vào `develop` (`docs/`) |
| EDA dịch vụ, hợp đồng, thanh toán | | Chưa bắt đầu |
| Đặc trưng T-M-S, chia train/test | `feat/feature-engineering` | Đã tạo nhánh, chưa có commit |
| Khai phá luật, contrast, subgroup discovery | `feat/pattern-mining` | Chưa bắt đầu phần code |
| Phân cụm T-M-S | | Chưa bắt đầu |
| Baseline LR / Decision Tree → RF / XGBoost / LightGBM | | Chưa bắt đầu |
| Đánh giá, SHAP, luật làm đặc trưng (RQ4) | | Chưa bắt đầu |
| Báo cáo, slide, tổng duyệt | | Chưa bắt đầu |

---

## 8. Quy ước code

1. **Python 3.10+**. Thư viện cần có trong `requirements.txt` (bản trên `develop` đã bổ sung đầy đủ thư viện):
   `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `scikit-learn`, `mlxtend`, `pysubgroup`, `imbalanced-learn`, `xgboost`, `lightgbm`, `shap`. Ai thêm thư viện thì cập nhật `requirements.txt` trong cùng PR.
2. **Đường dẫn dữ liệu** lấy từ `src/data_loader.py` (`RAW_DATA_PATH`, `PROCESSED_DATA_PATH`). **Hằng số mới** (seed, ngưỡng, mốc rời rạc hóa) đặt trong `src/config.py`. Không viết cứng trong notebook.
3. **Notebook phải chạy từ trên xuống dưới** sau khi Restart Kernel mà không lỗi. Logic dùng lại được thì chuyển vào `src/`. Đầu notebook phân tích luôn có đoạn code chuẩn ở mục 3.
4. **Chống rò rỉ dữ liệu:**
   - Không tạo đặc trưng nào từ cột `Churn`.
   - Mọi bước học từ dữ liệu (scaler, encoder, phân vị, SMOTE, khai phá luật để làm đặc trưng) chỉ **fit trên train**, rồi áp sang test.
   - SMOTE đặt **bên trong** pipeline CV, không áp trước khi chia.
5. **Đặt tên:** biến và hàm `snake_case`; notebook `NN_ten_viet_thuong.ipynb`; hình `reports/figures/ten_mo_ta.png`; bảng `reports/tables/ten_mo_ta.csv`.
6. **Git flow:**
   - Nhánh mới tách từ `develop`, đặt tên `feat/ten-viec` (ví dụ `feat/modeling`), `fix/ten-loi`, `docs/ten-tai-lieu`.
   - Mở PR vào `develop`, ít nhất 1 người duyệt. **Không push thẳng** lên `develop` hoặc `main`.
   - `main` chỉ nhận gộp từ `develop` khi đến mốc nộp bài.
7. **Commit message:** `[LOẠI] mo ta ngan`. Loại: `FEAT`, `FIX`, `DOCS`, `DATA`, `REFACTOR`, `CHORE`. Ví dụ: `[DOCS] them plan.md`.
8. **Không commit:** file `.env`, mô hình nhị phân (`.gitignore` đã chặn), output notebook quá nặng, file tạm.

---

## 9. Quy tắc khi dùng AI

### 9.1 AI được dùng để

- Gợi ý code, sửa lỗi, tối ưu, viết test.
- Giải thích khái niệm, thuật toán, bài báo.
- Góp ý cấu trúc báo cáo, sửa câu chữ, dịch.
- Rà soát code tìm lỗi rò rỉ dữ liệu.

### 9.2 Quy tắc bắt buộc

| # | Quy tắc | Vì sao |
|---|---|---|
| R1 | **Đưa `docs/plan.md` cho AI trước khi giao việc** | AI không biết các quyết định đã chốt |
| R2 | **Không chấp nhận đề xuất trái với mục 2** (chia theo thời gian, dùng RFM, bỏ bước lọc `tenure > 0`, đổi seed...) | Giữ thống nhất cả nhóm |
| R3 | **Mọi con số trong báo cáo phải đến từ code chạy được trong repo.** Không chép số AI đưa ra mà chưa tự chạy | AI có thể bịa số liệu |
| R4 | **Mọi trích dẫn phải tự mở và kiểm tra bài gốc** (tác giả, năm, tên bài, nơi đăng) | AI có thể bịa tài liệu tham khảo |
| R5 | **Code AI viết phải được chạy và đọc hiểu** trước khi commit. Người commit chịu trách nhiệm | Tránh lỗi ẩn, tránh không giải thích được khi bị hỏi |
| R6 | **Không dán dữ liệu cá nhân hoặc dữ liệu công ty** vào AI. Chỉ dùng bộ Telco công khai | Bảo mật |
| R7 | **Không nộp văn bản AI viết nguyên văn.** Phải viết lại bằng lời của nhóm và hiểu nội dung | Liêm chính học thuật |
| R8 | **Ghi nhật ký** mỗi lần dùng AI cho việc quan trọng vào `docs/ai_usage_log.md` | Minh bạch khi báo cáo, theo yêu cầu môn học |
| R9 | **Dùng đúng thuật ngữ ở mục 5** khi yêu cầu AI viết hoặc sửa nội dung | Báo cáo nhất quán |
| R10 | Khi AI đề xuất **thay đổi quyết định đã chốt** và bạn thấy hợp lý: **không tự sửa**, đưa ra họp nhóm | Tránh mỗi người một hướng |
| R11 | **Không để AI sửa code của thành viên khác** trong PR của mình. Thấy lỗi thì báo người sở hữu (mục 7) | Tránh xung đột khi gộp nhánh |

### 9.3 Mẫu nhật ký `docs/ai_usage_log.md`

```markdown
| Ngày | Người | Công cụ | Việc nhờ AI | Đã kiểm tra / chỉnh sửa gì |
|---|---|---|---|---|
| 2026-11-20 | ____ | Claude | Viết hàm min_improvement_filter | Chạy lại, so số luật với pilot (425), sửa tên biến |
```

---

## 10. Mẫu prompt khi bắt đầu phiên AI

Sao chép, điền vào chỗ trống, đính kèm `docs/plan.md` và tài liệu liên quan:

```
Bối cảnh: Tôi là [tên] - [vai trò] trong dự án ở file docs/plan.md đính kèm.
Hãy đọc kỹ docs/plan.md và tuân thủ mục 2 (quyết định đã chốt), mục 3 (luồng dữ liệu),
mục 5 (thuật ngữ), mục 8 (quy ước code).
Không đề xuất thay đổi quyết định đã chốt. Nếu thấy cần thay đổi, chỉ nêu ra để tôi đưa vào họp nhóm.
Không tự đưa ra số liệu hay trích dẫn - nếu cần, hãy viết code để tôi tự chạy.

Việc cần làm: [mô tả cụ thể]
File liên quan: [đường dẫn trong repo]
Đầu ra mong muốn: [hàm / notebook / đoạn báo cáo / ...]
Tiêu chí đúng: [ví dụ: tập phân tích 7.032 dòng, 1.869 churn]
```

---

## 11. Checklist trước khi mở pull request

- [ ] Nhánh tách từ `develop`, PR mở vào `develop`.
- [ ] Đã đọc lại mục 2, 3 và 8.
- [ ] Notebook chạy lại từ đầu không lỗi; seed = 42.
- [ ] Notebook phân tích / mô hình có bước lọc `tenure > 0` và `assert` số dòng (mục 3).
- [ ] Không có bước nào fit trên toàn bộ dữ liệu trước khi chia train/test.
- [ ] Thư viện mới đã thêm vào `requirements.txt`; hằng số mới đã đưa vào `src/config.py`.
- [ ] Code do AI viết đã được đọc hiểu và ghi vào `docs/ai_usage_log.md`.
- [ ] Hình vào `reports/figures/`, bảng vào `reports/tables/`.

---

## 12. Nhật ký thay đổi và câu hỏi mở

| Ngày | Người | Thay đổi |
|---|---|---|
| 2026-11-18 | ____ | Tạo v1.0 theo hiện trạng repo: quyết định D1-D10, luồng dữ liệu hai lớp, số liệu chuẩn, git flow `feat/*` → `develop`, quy tắc dùng AI |
| 2026-11-26 | Hưng | Cập nhật v1.1: đồng bộ tài liệu vào `docs/`, thống nhất `reports/` và giữ lại `results/`, chốt phân công 4 thành viên (Hưng, Hiếu, Hòa, Kiên) |

**Câu hỏi mở cho buổi họp:**

- [x] Đã chốt thống nhất `report/` → `reports/` (chứa `figures/`, `tables/`, báo cáo cuối & slide) và giữ lại `results/` cho đầu ra thí nghiệm / danh sách khách hàng rủi ro cao.
- [x] Đã chuyển `problem_definition.md`, `pattern_mining_approach.md`, `plan.md`, `PROJECT_STRUCTURE.md`, `ai_usage_log.md` vào thư mục `docs/`.
- [ ] `clean_data()` điền `TotalCharges` rỗng bằng `MonthlyCharges` cho 11 khách `tenure = 0`. Vì nhóm này bị lọc ở bước phân tích (D3) nên không ảnh hưởng kết quả, nhưng nên ghi chú trong notebook `01_data_cleaning`.
- [ ] Gộp `feat/eda-visualization` và `feat/pattern-mining` vào `develop`. Khi gộp sẽ có xung đột ở `requirements.txt` và `README.md`.
