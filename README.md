## 1. Phân tích dữ liệu (EDA)

### Mục tiêu
- Thống kê và tìm hiểu tổng quan dữ liệu.
- Kiểm tra tỷ lệ khách hàng rời bỏ dịch vụ (Churn).
- Phân tích mối liên hệ giữa các đặc điểm nhân khẩu học và Churn.

### Nội dung thực hiện
- Thống kê mô tả dữ liệu.
- Phân tích tỷ lệ khách hàng Churn và không Churn.
- Phân tích Churn theo 4 đặc điểm:
  - Gender: Giới tính.
  - SeniorCitizen: Khách hàng cao tuổi.
  - Partner: Có bạn đời hay không.
  - Dependents: Có người phụ thuộc hay không.
- Sử dụng kiểm định Chi-square và Cramér's V để đánh giá mối liên hệ giữa các đặc điểm nhân khẩu học và Churn.

### Kết quả
Các biểu đồ được lưu trong thư mục `reports/figures/`, bao gồm:
- `churn_distribution.png`
- `churn_by_gender.png`
- `churn_by_seniorcitizen.png`
- `churn_by_partner.png`
- `churn_by_dependents.png`

### Notebook
`notebooks/02_eda_demographics_churn.ipynb`
