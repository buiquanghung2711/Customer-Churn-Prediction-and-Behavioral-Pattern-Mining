# Định nghĩa bài toán - Customer Behavior Mining & Churn Prediction

> **Trạng thái:** v1.0 - chốt để cả nhóm triển khai
> **Bộ dữ liệu:** Telco Customer Churn (IBM sample data)
> **Tài liệu liên quan:** [`pattern_mining_approach.md`](./pattern_mining_approach.md)

Mọi số liệu trong tài liệu này được tính trực tiếp trên file `Telco-Customer-Churn.csv` (7.043 dòng). Các con số có ghi "đề xuất" là mục tiêu nhóm cần chốt lại trong buổi họp tuần 1.

---

## 1. Bối cảnh và mục tiêu

Một công ty viễn thông cung cấp dịch vụ điện thoại và Internet cho 7.043 khách hàng. Khoảng 1/4 khách hàng đã rời bỏ trong tháng gần nhất. Giữ một khách hàng hiện tại thường rẻ hơn nhiều so với tìm khách mới, nên công ty cần biết **ai sắp rời bỏ** và **vì sao**.

Dự án có hai mục tiêu song song:

| # | Mục tiêu | Loại bài toán | Câu hỏi trả lời |
|---|---|---|---|
| M1 | **Khai phá hành vi** | Unsupervised / descriptive | Nhóm khách nào, với tổ hợp đặc điểm nào, có rủi ro rời bỏ cao? |
| M2 | **Dự đoán churn** | Supervised classification | Với một khách cụ thể, xác suất rời bỏ là bao nhiêu? |

M1 tạo ra insight và đặc trưng cho M2. M2 cho ra danh sách khách cần chăm sóc. Kết quả cuối là đề xuất hành động giữ chân theo từng phân khúc.

---

## 2. Dữ liệu

### 2.1 Nguồn

| Mục | Giá trị |
|---|---|
| Tên | Telco Customer Churn |
| Nguồn gốc | IBM sample dataset |
| Bản dùng trong dự án | Kaggle `blastchar/telco-customer-churn`, hoặc bản mirror trên GitHub `IBM/telco-customer-churn-on-icp4d` (`data/Telco-Customer-Churn.csv`) |
| Kích thước | 7.043 dòng × 21 cột |
| Đơn vị phân tích | 1 dòng = 1 khách hàng (`customerID` duy nhất) |
| Dạng dữ liệu | **Cắt ngang (cross-sectional)**: một ảnh chụp tại một thời điểm, không có lịch sử giao dịch theo thời gian |

### 2.2 Nhóm cột

| Nhóm | Cột | Ghi chú |
|---|---|---|
| Định danh | `customerID` | Không dùng để phân tích |
| Nhân khẩu học | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | `SeniorCitizen` mã hóa 0/1, các cột còn lại là Yes/No |
| Gắn bó | `tenure` | Số tháng đã sử dụng dịch vụ, từ 0 đến 72 |
| Dịch vụ lõi | `PhoneService`, `MultipleLines`, `InternetService` | `InternetService`: DSL / Fiber optic / No |
| Dịch vụ cộng thêm | `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Có giá trị cấu trúc "No internet service" |
| Hợp đồng và thanh toán | `Contract`, `PaperlessBilling`, `PaymentMethod` | `Contract`: Month-to-month / One year / Two year |
| Chi phí | `MonthlyCharges`, `TotalCharges` | `TotalCharges` đang lưu dạng chuỗi |
| **Nhãn** | `Churn` | Yes/No |

### 2.3 Chất lượng dữ liệu đã kiểm tra

| Vấn đề | Chi tiết | Xử lý |
|---|---|---|
| `TotalCharges` rỗng | 11 dòng, **tất cả có `tenure = 0` và `Churn = No`** (khách vừa gia nhập, chưa có hóa đơn) | Loại khỏi tập phân tích (xem 4.3) |
| `TotalCharges` sai kiểu | Lưu dạng chuỗi | Ép sang số |
| Giá trị cấu trúc | "No internet service", "No phone service" là hệ quả trực tiếp của `InternetService = No` / `PhoneService = No` | Khi khai phá mẫu: bỏ các item này để tránh luật hiển nhiên |
| Cột gần trùng | `TotalCharges ≈ tenure × MonthlyCharges` (chênh lệch trung vị 0%, IQR ±2%) | Không dùng trực tiếp; chỉ dùng phần chênh lệch làm đặc trưng "biến động giá" |
| Trùng lặp, thiếu khác | Không có | - |

### 2.4 Bức tranh nền (sau khi loại 11 dòng tenure = 0)

- **7.032 khách**, trong đó **1.869 churn → tỷ lệ nền 26,58%**. Nhãn mất cân bằng ở mức vừa phải (khoảng 1 : 2,8).
- **55,5% khách rời bỏ có tenure ≤ 12 tháng** → churn tập trung ở giai đoạn đầu.

Tỷ lệ churn theo từng biến (đây là "mức đơn biến" mà mọi mẫu khai phá phải vượt qua mới có giá trị):

| Biến | Giá trị | Số khách | Tỷ lệ churn |
|---|---|---:|---:|
| Contract | Month-to-month | 3.875 | **42,7%** |
| | One year | 1.473 | 11,3% |
| | Two year | 1.695 | 2,8% |
| InternetService | Fiber optic | 3.096 | **41,9%** |
| | DSL | 2.421 | 19,0% |
| | No | 1.526 | 7,4% |
| PaymentMethod | Electronic check | 2.365 | **45,3%** |
| | Mailed check | 1.612 | 19,1% |
| | Bank transfer (automatic) | 1.544 | 16,7% |
| | Credit card (automatic) | 1.522 | 15,2% |
| tenure (nhóm) | 0-6 tháng | 1.481 | **52,9%** |
| | 7-12 tháng | 705 | 35,9% |
| | 13-24 tháng | 1.024 | 28,7% |
| | 25-48 tháng | 1.594 | 20,4% |
| | 49-72 tháng | 2.239 | 9,5% |
| SeniorCitizen | Có | 1.142 | 41,7% |
| TechSupport | No | 3.473 | 41,6% |
| OnlineSecurity | No | 3.498 | 41,8% |

*(Số khách ở bảng này tính trên 7.043 dòng gốc, trừ nhóm tenure.)*

---

## 3. Định nghĩa thời gian

### 3.1 Vấn đề của bộ Telco

Bộ Telco là **ảnh chụp tại một thời điểm**, không có ngày giao dịch. Vì vậy không thể dựng cửa sổ quan sát kiểu "lấy 12 tháng hành vi gần nhất" như với dữ liệu giao dịch. Nhóm định nghĩa các mốc thời gian **ở mức khái niệm** dựa trên cách bộ dữ liệu được tạo ra, và ghi rõ giới hạn.

### 3.2 Các mốc

| Ký hiệu | Tên | Định nghĩa |
|---|---|---|
| **T0** | Ngày trích xuất (snapshot date) | Thời điểm dữ liệu được chụp lại. Mọi đặc trưng được ghi nhận tại T0 (hoặc tại thời điểm rời đi với khách đã churn). |
| **W_churn** | Cửa sổ churn (label window) | Khoảng **(T0 - 1 tháng, T0]**. Theo mô tả bộ dữ liệu, `Churn = Yes` nghĩa là khách đã rời đi trong tháng cuối cùng. |
| **W_obs** | Cửa sổ quan sát (observation window) | Khoảng **[T0 - tenure_i, T0]** của từng khách i: từ ngày gia nhập đến T0. Độ dài **thay đổi theo từng khách**, tối đa 72 tháng. |

```
   Ngày gia nhập                                                       Ngày trích xuất
   (T0 - tenure_i)                                    T0 - 1 tháng           T0
        |------------------------------------------------------|---------------|
        |<------------- Cửa sổ quan sát W_obs (độ dài = tenure_i) ------------>|
                                                               |<- W_churn --->|
                                                                 Churn = Yes nếu
                                                                 rời đi ở đây
```

### 3.3 Hệ quả cần nhớ khi phân tích

1. **Cửa sổ quan sát chồng lên cửa sổ churn.** Đặc trưng như `tenure`, `TotalCharges`, trạng thái dịch vụ đều đo đến T0, tức là bao gồm cả tháng khách rời đi. Với dữ liệu thật ta sẽ cắt đặc trưng trước tháng churn; với Telco điều này không làm được. Nhóm **chấp nhận** giới hạn này và ghi vào phần hạn chế của báo cáo.
2. **`tenure` của khách churn là tổng thời gian sống của khách đó.** Luật "tenure 0-6 tháng → churn" phản ánh đúng rằng rủi ro rời bỏ cao ở giai đoạn đầu, nhưng không được diễn giải thành "cứ ở dưới 6 tháng là sẽ rời".
3. **Không có trục thời gian → không chia train/test theo thời gian.** Khác với kế hoạch tổng thể ban đầu, nhóm dùng **chia ngẫu nhiên có phân tầng (stratified) 70/30** và **stratified 5-fold cross-validation** trên tập train.

---

## 4. Ranh giới Churn

### 4.1 Định nghĩa nhãn chính

Với mỗi khách hàng i có `tenure_i ≥ 1`:

```
y_i = 1  (churn)      nếu khách i chấm dứt dịch vụ trong W_churn = (T0 - 1 tháng, T0]
y_i = 0  (ở lại)      nếu khách i vẫn còn hoạt động tại T0
```

Nhãn lấy trực tiếp từ cột `Churn`. Đây là **churn chủ động hay bị động đều tính** (bộ dữ liệu không phân biệt khách tự hủy với khách bị cắt do nợ cước).

### 4.2 Nhãn phụ dùng cho khai phá

Nhãn phụ chỉ dùng để phân tích mô tả, **không dùng làm biến đầu vào** cho mô hình dự đoán.

| Nhãn phụ | Điều kiện | Số khách | Mục đích |
|---|---|---:|---|
| `early_churn` | Churn = Yes và tenure ≤ 12 | khoảng 1.037 (55,5% số churn) | Tìm mẫu "rời bỏ sớm" - thường do onboarding, kỳ vọng giá |
| `late_churn` | Churn = Yes và tenure > 12 | khoảng 832 | Tìm mẫu "rời bỏ sau khi đã gắn bó" - thường do giá, cạnh tranh, chất lượng |

### 4.3 Quy tắc loại trừ

| Quy tắc | Số dòng | Lý do |
|---|---:|---|
| Loại khách có `tenure = 0` | 11 | Gia nhập ngay trong tháng trích xuất → không có cửa sổ quan sát, `TotalCharges` rỗng, không thể có nhãn churn có nghĩa |

**Tập phân tích chính thức: 7.032 khách, 1.869 churn (26,58%).**

### 4.4 Khách "ở lại" là dữ liệu bị kiểm duyệt phải (right-censored)

`Churn = No` chỉ có nghĩa là **chưa rời đi tại T0**, không có nghĩa là sẽ không rời đi. Một khách tenure 2 tháng với `Churn = No` hoàn toàn có thể rời đi tháng sau. Nếu còn thời gian, nhóm có thể làm thêm phân tích sống còn (Kaplan-Meier theo `Contract`) như một phần mở rộng.

---

## 5. Câu hỏi nghiên cứu

| Mã | Câu hỏi | Phụ trách chính | Phương pháp |
|---|---|---|---|
| RQ1 | Tổ hợp đặc điểm nào gắn với tỷ lệ churn **cao bất thường** so với mức nền 26,6%? | TV3 | Luật kết hợp có lớp đích (class association rules) |
| RQ2 | Mẫu nào **phân biệt rõ** nhóm rời bỏ với nhóm ở lại? Mẫu nào gắn với **giữ chân**? | TV3 | Contrast sets / emerging patterns |
| RQ3 | Có những **phân khúc hành vi** nào, rủi ro từng phân khúc ra sao? | TV3, TV1 | Phân cụm theo T-M-S, subgroup discovery |
| RQ4 | Mẫu khai phá được có **cải thiện mô hình** dự đoán không? | TV4 | So sánh mô hình có / không có đặc trưng từ mẫu |
| RQ5 | Mô hình tốt nhất đạt hiệu năng bao nhiêu, và yếu tố nào ảnh hưởng mạnh nhất? | TV4 | LR, RF, XGBoost, LightGBM + SHAP |

### Điều chỉnh RFM cho Telco

Bộ Telco không có giao dịch nên **không tính được Recency và Frequency**. Nhóm thay RFM bằng bộ **T-M-S**:

| Trục | Cột nguồn | Ý nghĩa |
|---|---|---|
| **T**enure | `tenure` | Độ gắn bó |
| **M**onetary | `MonthlyCharges` | Giá trị mỗi tháng |
| **S**ervices | số dịch vụ cộng thêm đang dùng (0-6) | Độ "dính" với hệ sinh thái |

Kiểm tra nhanh trục S cho thấy quan hệ **không đơn điệu**: khách có 1 dịch vụ cộng thêm churn 45,8%, khách có đủ 6 dịch vụ chỉ churn 5,3%, còn khách có 0 dịch vụ churn 21,5% (vì phần lớn là khách không dùng Internet). Đây là ví dụ tốt cho thấy cần khai phá tổ hợp thay vì nhìn từng biến.

---

## 6. Phạm vi

**Trong phạm vi**
- Làm sạch, khám phá dữ liệu, tạo đặc trưng.
- Khai phá mẫu: luật kết hợp, contrast patterns, subgroup discovery, phân cụm.
- Mô hình phân loại churn và giải thích bằng SHAP.
- Đề xuất hành động giữ chân theo phân khúc, ước tính hiệu quả.

**Ngoài phạm vi**
- Khai phá chuỗi tuần tự (sequential pattern mining) - bộ dữ liệu không có trình tự sự kiện.
- Dự đoán thời điểm rời bỏ chính xác (chỉ làm Kaplan-Meier nếu còn thời gian).
- Triển khai production thật.

---

## 7. Tiêu chí thành công (đề xuất)

| Tầng | Chỉ số | Mục tiêu đề xuất |
|---|---|---|
| Khai phá mẫu | Số luật / nhóm con có ý nghĩa thống kê, đã lọc trùng, ≤ 4 điều kiện | ≥ 10 |
| | Độ ổn định: tỷ lệ luật giữ confidence ≥ 0,5 trên tập holdout | ≥ 80% |
| Mô hình | PR-AUC (chỉ số chính, vì nhãn mất cân bằng) | Cao hơn baseline Logistic Regression |
| | Recall lớp churn tại ngưỡng đã chọn | ≥ 0,75 |
| Kinh doanh | Lift tại top 20% khách rủi ro cao nhất | ≥ 2,0 (bắt được ≥ 40% số churn khi chỉ chăm sóc 20% khách) |

---

## 8. Quy tắc chống rò rỉ dữ liệu

1. Không tạo đặc trưng nào từ cột `Churn`, kể cả nhãn phụ ở mục 4.2.
2. Mọi phép biến đổi học từ dữ liệu (ngưỡng rời rạc hóa theo phân vị, scaler, encoder, SMOTE) chỉ **fit trên tập train**, rồi áp sang tập test.
3. Khai phá luật trên tập train, kiểm tra lại trên tập holdout trước khi báo cáo.
4. Seed cố định: `random_state = 42` cho mọi phép chia và mô hình.

---

## 9. Nhật ký quyết định

| Mã | Quyết định | Lý do |
|---|---|---|
| D1 | Dùng bộ Telco Customer Churn | Nhóm đã chốt |
| D2 | Nhãn chính = cột `Churn` (rời đi trong tháng cuối trước T0) | Theo định nghĩa gốc của bộ dữ liệu |
| D3 | Loại 11 khách `tenure = 0` | Không có cửa sổ quan sát |
| D4 | Chia stratified ngẫu nhiên 70/30, CV 5-fold | Không có trục thời gian |
| D5 | RFM → T-M-S | Không có dữ liệu giao dịch |
| D6 | Ngưỡng rời rạc hóa: tenure theo mốc nghiệp vụ, `MonthlyCharges` theo tam phân vị | Chi tiết trong `pattern_mining_approach.md` |

### Câu hỏi còn mở

- [ ] Có báo cáo riêng mẫu cho `early_churn` và `late_churn` không, hay chỉ dùng như phân tích phụ?
- [ ] Chốt ngưỡng support / confidence cuối cùng sau khi chạy thử tuần 2.
- [ ] Có làm thêm phân tích sống còn (Kaplan-Meier) không?

---

## 10. Ánh xạ sang dữ liệu giao dịch thật (cho phần "Hướng phát triển")

Nếu áp dụng cho dữ liệu có lịch sử giao dịch (ví dụ khách lưu trú khu nghỉ dưỡng), cửa sổ thời gian sẽ được dựng tách bạch:

```
|<---- Cửa sổ quan sát (12 tháng) ---->|<- Gap (1 tháng) ->|<---- Cửa sổ churn (6 tháng) ---->|
                                       T0                                                   
  Tính đặc trưng: R, F, M, xu hướng      Thời gian để         Churn = không có booking nào
                                         triển khai chiến dịch
```

- **Gap** là khoảng trống để đội marketing kịp hành động trước khi khách thực sự rời đi.
- Với ngành tần suất mua thấp như du lịch nghỉ dưỡng, cửa sổ churn nên dài (6-12 tháng) thay vì 1 tháng như viễn thông.
- Có thể tạo nhiều snapshot T0 trượt theo tháng để tăng số mẫu và chia train/test theo thời gian.
