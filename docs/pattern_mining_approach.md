# Cách tiếp cận Pattern Mining cho bài toán Churn

> **Trạng thái:** v1.0
> **Phụ thuộc:** [`problem_definition.md`](./problem_definition.md) - định nghĩa nhãn, tập phân tích 7.032 khách, tỷ lệ nền 26,58%
> **Notebook chính:** `notebooks/04_rules.ipynb` - **Module:** `src/mining/`

Tài liệu này trả lời ba câu hỏi: nhóm sẽ dùng những kỹ thuật khai phá mẫu nào, chạy theo các bước ra sao với tham số nào, và kết quả bàn giao cho ai. Mục 6 có kết quả chạy thử thật trên bộ Telco để nhóm thấy trước đầu ra trông thế nào.

---

## 1. Vai trò của khai phá mẫu trong dự án

Mô hình dự đoán (XGBoost, LightGBM...) trả lời **"khách này có rời đi không"**, nhưng khó trả lời **"nhóm khách nào rời đi, vì tổ hợp lý do gì"**. Khai phá mẫu lấp khoảng trống đó bằng các mẫu dạng:

```
NẾU  InternetService = Fiber optic  VÀ  PaymentMethod = Electronic check  VÀ  tenure 0-6 tháng
THÌ  Churn = Yes   (confidence 76,6%, gấp 2,9 lần mức nền)
```

Mẫu dạng này có ba công dụng:

| Đầu ra | Người nhận | Dùng để |
|---|---|---|
| Luật và nhóm con rủi ro cao | TV1 | Viết chân dung phân khúc, đề xuất hành động giữ chân |
| Luật dưới dạng đặc trưng nhị phân | TV4 | Thêm vào mô hình, kiểm tra RQ4 |
| Biểu đồ, bảng luật | TV3, TV1 | Báo cáo và slide |

---

## 2. Tổng quan tài liệu

### 2.1 Các họ phương pháp liên quan

| Họ phương pháp | Ý tưởng | Công trình nền tảng | Áp dụng cho Telco |
|---|---|---|---|
| **Frequent itemset & association rules** | Tìm tổ hợp thuộc tính xuất hiện cùng nhau thường xuyên, sinh luật X → Y | Agrawal & Srikant (1994) - Apriori; Han, Pei & Yin (2000) - FP-Growth | Bước nền tảng. Dùng FP-Growth vì nhanh hơn Apriori trên dữ liệu dày |
| **Class association rules (CAR)** | Chỉ giữ luật có vế phải là nhãn lớp (`Churn = Yes`), có thể dùng luật để phân loại | Liu, Hsu & Ma (1998) - CBA; Li, Han & Pei (2001) - CMAR | **Phương pháp chính** cho RQ1 |
| **Contrast sets / emerging patterns** | Tìm mẫu có tần suất khác biệt lớn giữa hai nhóm | Bay & Pazzani (2001) - STUCCO; Dong & Li (1999) - emerging patterns | RQ2: mẫu phân biệt nhóm churn với nhóm ở lại, và mẫu giữ chân |
| **Subgroup discovery** | Tìm nhóm con có phân bố nhãn khác thường, cân bằng giữa kích thước và độ lệch | Wrobel (1997); Lavrač và cộng sự (2004) - CN2-SD | RQ3: nhóm con đủ lớn để làm chiến dịch |
| **Significant pattern mining** | Kiểm định thống kê để loại mẫu xuất hiện do ngẫu nhiên khi thử hàng nghìn tổ hợp | Webb (2007); Bayardo, Agrawal & Gunopulos (2000) - minimum improvement | Bước lọc bắt buộc trước khi báo cáo |
| **Churn trong viễn thông** | Đánh giá mô hình churn theo lợi nhuận, không chỉ theo độ chính xác | Verbeke và cộng sự (2012) | Khung đánh giá kinh doanh cho TV1 |

### 2.2 Những điểm rút ra từ tài liệu

1. **Luật nhiều không có nghĩa là tốt.** Với 51 item, khai phá ở support 3% đã sinh hơn 69.000 tập phổ biến. Phần lớn luật sinh ra là biến thể của nhau. Cần lọc trùng (minimum improvement) và kiểm định thống kê.
2. **Confidence một mình gây hiểu lầm.** Một luật confidence 30% nghe thấp nhưng vẫn cao hơn mức nền 26,6%. Luôn so với mức nền bằng **lift** hoặc **leverage**.
3. **Kiểm định trên dữ liệu độc lập.** Webb (2007) đề xuất khai phá trên một phần dữ liệu và kiểm tra trên phần còn lại (holdout), thay vì chỉ hiệu chỉnh p-value.
4. **Mẫu là tương quan, không phải nhân quả.** "Electronic check → churn" không có nghĩa là đổi phương thức thanh toán sẽ giữ được khách.

---

## 3. Chuẩn bị dữ liệu giao dịch

Thuật toán khai phá luật làm việc trên **tập item**, nên mỗi khách hàng được chuyển thành một "giao dịch" gồm các item dạng `thuộc_tính=giá_trị`.

```
Khách 7590-VHVEG  →  { gender=Female, Partner=Yes, InternetService=DSL, OnlineBackup=Yes,
                       Contract=Month-to-month, PaymentMethod=Electronic check,
                       tenure_bin=0-6, mc_bin=low, addon_bin=1-2, Churn=No, ... }
```

### 3.1 Rời rạc hóa biến số

| Biến | Cách chia | Các nhóm | Lý do |
|---|---|---|---|
| `tenure` | Mốc nghiệp vụ | 0-6, 7-12, 13-24, 25-48, 49-72 tháng | Khớp với chu kỳ hợp đồng; tách giai đoạn onboarding |
| `MonthlyCharges` | Tam phân vị (fit trên train) | low ≤ 50,4; mid ≤ 83,9; high > 83,9 | Mỗi nhóm khoảng 1/3 khách, tránh nhóm quá nhỏ |
| `TotalCharges` | **Bỏ** | - | Gần trùng với `tenure × MonthlyCharges` |
| Số dịch vụ cộng thêm (mới) | Theo ý nghĩa | 0, 1-2, 3+ | Trục S của T-M-S |

**Kiểm tra độ nhạy:** chạy lại với một cách chia khác (ví dụ tenure chia theo tứ phân vị) và báo cáo các luật vẫn giữ nguyên. Luật chỉ xuất hiện với một cách chia là luật yếu.

### 3.2 Làm sạch item

| Việc | Chi tiết |
|---|---|
| Bỏ item cấu trúc | Bỏ `*=No internet service` và `MultipleLines=No phone service`. Chúng luôn đi cùng `InternetService=No` / `PhoneService=No` nên sinh luật hiển nhiên |
| Chuẩn hóa nhãn | `SeniorCitizen` 0/1 → No/Yes |
| Item suy diễn | `AutoPay = Yes/No` (gộp 2 phương thức thanh toán tự động) |
| Bỏ cột | `customerID`, `TotalCharges` |

Kết quả: **51 item**, ma trận one-hot dạng boolean.

---

## 4. Quy trình khai phá

```
 Bảng khách hàng ──► [B1] Giao dịch ──► [B2] FP-Growth ──► [B3] Luật → Churn
                                                                  │
          [B6] Bàn giao ◄── [B5] Kiểm định holdout ◄── [B4] Lọc trùng + Fisher
                                                                  │
                         [B7] Contrast & subgroup discovery (song song với B3-B5)
```

| Bước | Việc | Công cụ | Tham số khởi đầu |
|---|---|---|---|
| **B1** | Chia train/holdout stratified 70/30, tạo giao dịch như mục 3 | `pandas` | `random_state=42` |
| **B2** | Khai phá tập phổ biến **trên train** | `mlxtend.frequent_patterns.fpgrowth` | `min_support=0.03` (khoảng 150 khách), `max_len=4` |
| **B3** | Sinh luật có vế phải `Churn=Yes` | `mlxtend.frequent_patterns.association_rules` | `min_confidence=0.5`, `lift ≥ 1.5` |
| **B4a** | Lọc trùng bằng **minimum improvement**: chỉ giữ luật có confidence cao hơn mọi luật tổng quát hơn (bỏ bớt điều kiện) ít nhất δ | Code tự viết | `δ = 0.02` |
| **B4b** | Kiểm định Fisher exact một phía cho từng luật, hiệu chỉnh Bonferroni | `scipy.stats.fisher_exact` | `α = 0.05 / số luật` |
| **B5** | Kiểm tra trên holdout: luật còn giữ confidence ≥ 0,5 với ít nhất 30 khách khớp | Code tự viết | - |
| **B6** | Xếp hạng theo lift, gom nhóm luật cùng "lõi", chọn top 10-20 để diễn giải | - | ≤ 4 điều kiện mỗi luật |
| **B7a** | **Luật giữ chân**: vế phải `Churn=No`, confidence ≥ 0,97 | như B3 | Cũng phải lọc trùng như B4a |
| **B7b** | **Subgroup discovery**: tìm nhóm con tối đa WRAcc với đích `Churn=Yes` | `pysubgroup` | beam search, `depth=3`, top 20 |
| **B7c** | **Tách theo nhãn phụ**: chạy B2-B5 riêng cho `early_churn` và `late_churn` | như trên | Tùy kết quả họp nhóm |

### 4.1 Các độ đo dùng trong báo cáo

| Độ đo | Công thức | Đọc như thế nào |
|---|---|---|
| Support | P(X ∧ Churn) | Luật bao phủ bao nhiêu % khách |
| Confidence | P(Churn \| X) | Trong nhóm X, bao nhiêu % rời bỏ |
| Lift | Confidence / P(Churn) | Gấp bao nhiêu lần mức nền 26,6% |
| Leverage | P(X ∧ Churn) - P(X)·P(Churn) | Số khách churn "dôi ra" so với kỳ vọng |
| WRAcc | P(X) · (P(Churn \| X) - P(Churn)) | Cân bằng kích thước nhóm và độ lệch - dùng cho subgroup discovery |
| Growth rate | Supp(X \| churn) / Supp(X \| ở lại) | Dùng cho contrast / emerging patterns, ngưỡng đề xuất ≥ 3 |

### 4.2 Biến mẫu thành đặc trưng cho mô hình (RQ4)

1. Lấy top K luật (K = 10, 20, 30) sau bước B5.
2. Mỗi luật thành một cột nhị phân: `rule_k = 1` nếu khách khớp mọi điều kiện của luật.
3. TV4 so sánh 3 cấu hình trên cùng các fold CV: (a) đặc trưng gốc, (b) đặc trưng gốc + đặc trưng luật, (c) chỉ đặc trưng luật.
4. **Luật phải được khai phá chỉ trên fold train** của từng fold để tránh rò rỉ.

---

## 5. Cấu trúc code

```
src/mining/
├── transactions.py    # build_transactions(df, cut_points) -> DataFrame bool  (TV2)
├── rules.py           # mine_rules, min_improvement_filter, fisher_filter       (TV3)
├── validate.py        # holdout_check, sensitivity_check                       (TV3)
└── features.py        # rules_to_features(df, rules)                           (TV4)
notebooks/
└── 04_rules.ipynb     # chạy B1-B7, xuất bảng luật và biểu đồ
reports/
├── rules_churn.csv
├── rules_retention.csv
└── subgroups.csv
```

Khung code cho bước B2-B3:

```python
from mlxtend.frequent_patterns import fpgrowth, association_rules

itemsets = fpgrowth(T_train, min_support=0.03, use_colnames=True, max_len=4)
rules = association_rules(itemsets, metric="confidence", min_threshold=0.5)
churn_rules = rules[
    (rules["consequents"] == frozenset({"Churn=Yes"})) & (rules["lift"] >= 1.5)
].sort_values("lift", ascending=False)
```

---

## 6. Kết quả chạy thử

Chạy đúng quy trình B1-B5 với tham số ở mục 4 trên tập train 4.962 khách, kiểm tra trên holdout 2.070 khách.

### 6.1 Số lượng luật qua từng bước lọc

| Bước | Số lượng |
|---|---:|
| Item sau làm sạch | 51 |
| Tập phổ biến (support ≥ 3%, ≤ 4 item) | 69.164 |
| Luật → `Churn=Yes`, confidence ≥ 0,5, lift ≥ 1,5 | 887 |
| Sau lọc trùng minimum improvement (δ = 0,02) | **425** (giảm hơn một nửa) |
| Sau kiểm định Fisher + Bonferroni | 425 |
| Giữ confidence ≥ 0,5 trên holdout (≥ 30 khách khớp) | **376 / 425 (88%)** |

Nhận xét: kiểm định Fisher không loại thêm luật nào vì ngưỡng support 3% đã đủ lớn để mọi luật có lift ≥ 1,5 đều có ý nghĩa thống kê. Bước lọc trùng và kiểm tra holdout mới là hai bước có tác dụng thật.

### 6.2 Top 10 luật rời bỏ (xếp theo lift, trên train)

| # | Điều kiện | Số khách | Support | Confidence | Lift |
|---:|---|---:|---:|---:|---:|
| 1 | Fiber optic + Electronic check + tenure 0-6 | 242 | 4,9% | 76,6% | 2,89 |
| 2 | Fiber optic + PaperlessBilling + tenure 0-6 | 261 | 5,3% | 76,1% | 2,87 |
| 3 | Fiber optic + tenure 0-6 | 322 | 6,5% | 74,0% | 2,79 |
| 4 | PaperlessBilling + Electronic check + tenure 0-6 | 267 | 5,4% | 73,8% | 2,78 |
| 5 | No OnlineSecurity + Electronic check + tenure 0-6 | 311 | 6,3% | 73,2% | 2,76 |
| 6 | Electronic check + No TechSupport + tenure 0-6 | 312 | 6,3% | 72,2% | 2,72 |
| 7 | No OnlineBackup + Electronic check + tenure 0-6 | 275 | 5,5% | 72,2% | 2,72 |
| 8 | No DeviceProtection + Electronic check + tenure 0-6 | 285 | 5,7% | 71,8% | 2,71 |
| 9 | No OnlineSecurity + PaperlessBilling + tenure 0-6 | 339 | 6,8% | 70,8% | 2,65 |
| 10 | PaperlessBilling + No TechSupport + tenure 0-6 | 336 | 6,8% | 70,3% | 2,65 |

**Cách đọc:** 10 luật này thực chất xoay quanh **một lõi** là "khách mới (0-6 tháng) + Fiber optic hoặc Electronic check + thiếu dịch vụ hỗ trợ". Bước B6 cần gom chúng thành một phân khúc thay vì báo cáo 10 luật riêng lẻ. Đây là lý do cần làm thêm subgroup discovery (B7b): nó tự động chọn nhóm con đại diện thay vì liệt kê biến thể.

### 6.3 Luật giữ chân - ví dụ về luật trùng chưa lọc

| Điều kiện | Support | Confidence | Lift |
|---|---:|---:|---:|
| Contract = Two year | 23,7% | 97,1% | 1,32 |
| Two year + SeniorCitizen = No | 21,6% | 97,2% | 1,32 |
| Two year + PhoneService = Yes | 21,3% | 97,1% | 1,32 |

Hai luật dưới không thêm thông tin gì so với luật đầu (confidence gần như không đổi). Đây chính là loại luật mà minimum improvement sẽ loại. Kết luận thật chỉ có một: **hợp đồng 2 năm gần như loại bỏ rủi ro rời bỏ** (chỉ 2,8% churn).

---

## 7. Phân công

| Thành viên | Việc trong mảng khai phá mẫu | Hạn |
|---|---|---|
| TV1 | Duyệt định nghĩa item, gom luật thành phân khúc, viết chân dung và đề xuất hành động | Tuần 4 |
| TV2 | `transactions.py`: rời rạc hóa (fit trên train), làm sạch item, test không rò rỉ | Giữa tuần 2 |
| TV3 | `rules.py`, `validate.py`, notebook `04_rules`: chạy B2-B7, kiểm tra độ nhạy, xuất bảng và biểu đồ | Tuần 3-4 |
| TV4 | `features.py`, thí nghiệm RQ4 với khai phá luật trong từng fold | Tuần 4-5 |

### Tiêu chí hoàn thành

- [ ] Notebook `04_rules` chạy lại từ đầu ra cùng kết quả (seed 42).
- [ ] Mỗi luật báo cáo có đủ: điều kiện, số khách, support, confidence, lift, kết quả holdout.
- [ ] Có bảng kiểm tra độ nhạy với cách rời rạc hóa thứ hai.
- [ ] Top luật đã gom thành 3-6 phân khúc có tên và chân dung.
- [ ] Đã ghi rõ trong báo cáo: mẫu là tương quan, không phải nhân quả.

---

## 8. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Bùng nổ số luật | Hàng nghìn luật, nhiều biến thể | Minimum improvement, giới hạn `max_len=4`, gom theo lõi |
| Luật hiển nhiên | `InternetService=No → TechSupport=No internet service` | Đã bỏ item cấu trúc (mục 3.2) |
| Phụ thuộc cách chia nhóm | Luật biến mất khi đổi mốc tenure | Kiểm tra độ nhạy, chỉ báo cáo luật ổn định |
| Quá khớp với train | Luật mạnh trên train nhưng yếu trên holdout | Bắt buộc bước B5 |
| Diễn giải sai thành nhân quả | "Bỏ Electronic check sẽ giảm churn" | Diễn đạt "gắn với", đề xuất thử nghiệm A/B nếu muốn khẳng định |
| Thiên lệch do tenure | Luật tenure ngắn phản ánh thời điểm rời đi | Đọc mục 3.3 của `problem_definition.md` |

---

## 9. Tài liệu tham khảo

> Thông tin dưới đây đủ để tìm bài gốc. Khi đưa vào báo cáo, cần kiểm tra lại tập, số trang và định dạng trích dẫn theo yêu cầu của môn học.

1. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. *Proceedings of the 20th VLDB Conference*.
2. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. *Proceedings of ACM SIGMOD*.
3. Liu, B., Hsu, W., & Ma, Y. (1998). Integrating classification and association rule mining. *Proceedings of KDD-98*.
4. Li, W., Han, J., & Pei, J. (2001). CMAR: Accurate and efficient classification based on multiple class-association rules. *Proceedings of IEEE ICDM*.
5. Dong, G., & Li, J. (1999). Efficient mining of emerging patterns: Discovering trends and differences. *Proceedings of KDD-99*.
6. Bay, S. D., & Pazzani, M. J. (2001). Detecting group differences: Mining contrast sets. *Data Mining and Knowledge Discovery*.
7. Wrobel, S. (1997). An algorithm for multi-relational discovery of subgroups. *Proceedings of PKDD*.
8. Lavrač, N., Kavšek, B., Flach, P., & Todorovski, L. (2004). Subgroup discovery with CN2-SD. *Journal of Machine Learning Research*.
9. Bayardo, R. J., Agrawal, R., & Gunopulos, D. (2000). Constraint-based rule mining in large, dense databases. *Data Mining and Knowledge Discovery*.
10. Webb, G. I. (2007). Discovering significant patterns. *Machine Learning*.
11. Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research*.
12. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*.
13. Chapman, P., và cộng sự (2000). *CRISP-DM 1.0: Step-by-step data mining guide*.
