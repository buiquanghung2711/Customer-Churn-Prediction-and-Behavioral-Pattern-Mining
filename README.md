# Customer Churn Prediction and Behavioral Pattern Mining

Phân tích hành vi của khách hàng, dự đoán nguy cơ rời bỏ dịch vụ và sử dụng kết quả để đề xuất chiến lược giữ chân khách hàng trên bộ dữ liệu Telco Customer Churn.

---

## 1. Phân tích dữ liệu khám phá (EDA)

### Mục tiêu
- Thống kê và tìm hiểu tổng quan dữ liệu.
- Kiểm tra tỷ lệ khách hàng rời bỏ dịch vụ (Churn).
- Phân tích mối liên hệ giữa các đặc điểm nhân khẩu học và Churn.

### Nội dung thực hiện
- Thống kê mô tả dữ liệu trên tập làm sạch (`7.043` khách hàng, `21` cột).
- Phân tích tỷ lệ khách hàng Churn và không Churn.
- Phân tích Churn theo 4 đặc điểm nhân khẩu học (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) và tổ hợp đa chiều.
- Đánh giá ý nghĩa thống kê bằng kiểm định Chi-square, Cramér's V, hiệu chỉnh đa kiểm định (Benjamini–Hochberg, Holm–Bonferroni) và các chỉ số rủi ro (Risk Difference, Relative Risk, Odds Ratio).

### Kết quả
- Các biểu đồ được lưu trong thư mục `reports/figures/`:
  - `numerical_distributions.png`
  - `churn_distribution.png`
  - `churn_by_gender.png`
  - `churn_by_seniorcitizen.png`
  - `churn_by_partner.png`
  - `churn_by_dependents.png`
  - `churn_contribution_demographics.png`
  - `churn_by_demographic_groups.png`
  - `churn_heatmap_senior_partner.png`
- Các bảng tổng hợp được lưu trong thư mục `reports/tables/`:
  - `eda_demographics_summary.csv`
  - `eda_demographics_stats.csv`
  - `eda_demographics_risk.csv`
  - `eda_demographics_groups.csv`

### Notebook
- `notebooks/02_eda_demographics_churn.ipynb` (kèm phần tổng kết và nhận xét ở cuối notebook).
