import pandas as pd
import numpy as np
from pathlib import Path


RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "clean_data.csv"

CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "Churn",
]

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

DATA_DICTIONARY = {
    "customerID": "Mã định danh duy nhất cho mỗi khách hàng",
    "gender": "Giới tính khách hàng (Male / Female)",
    "SeniorCitizen": "Khách hàng có phải người cao tuổi không (1: Có, 0: Không)",
    "Partner": "Khách hàng có đối tác/bạn đời không (Yes / No)",
    "Dependents": "Khách hàng có người phụ thuộc không (Yes / No)",
    "tenure": "Số tháng khách hàng đã sử dụng dịch vụ",
    "PhoneService": "Khách hàng có đăng ký dịch vụ điện thoại không (Yes / No)",
    "MultipleLines": "Khách hàng có nhiều đường dây không (Yes / No / No phone service)",
    "InternetService": "Loại dịch vụ internet (DSL / Fiber optic / No)",
    "OnlineSecurity": "Có đăng ký bảo mật trực tuyến không (Yes / No / No internet service)",
    "OnlineBackup": "Có đăng ký sao lưu trực tuyến không (Yes / No / No internet service)",
    "DeviceProtection": "Có đăng ký bảo vệ thiết bị không (Yes / No / No internet service)",
    "TechSupport": "Có đăng ký hỗ trợ kỹ thuật không (Yes / No / No internet service)",
    "StreamingTV": "Có đăng ký truyền hình trực tuyến không (Yes / No / No internet service)",
    "StreamingMovies": "Có đăng ký phim trực tuyến không (Yes / No / No internet service)",
    "Contract": "Loại hợp đồng (Month-to-month / One year / Two year)",
    "PaperlessBilling": "Có sử dụng hóa đơn điện tử không (Yes / No)",
    "PaymentMethod": "Phương thức thanh toán (Electronic check / Mailed check / Bank transfer / Credit card)",
    "MonthlyCharges": "Số tiền thanh toán hàng tháng (USD)",
    "TotalCharges": "Tổng số tiền đã thanh toán (USD)",
    "Churn": "Khách hàng có rời bỏ dịch vụ không (Yes / No) - Biến mục tiêu",
}


def load_raw_data(path=None):
    path = path or RAW_DATA_PATH
    return pd.read_csv(path)


def fix_total_charges(df):
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].str.strip()
    df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    return df


def remove_duplicates(df):
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["customerID"])
    return df


def cast_dtypes(df):
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    return df


def clean_data(df):
    df = fix_total_charges(df)
    df = remove_duplicates(df)
    df = cast_dtypes(df)
    df = df.reset_index(drop=True)
    return df


def save_clean_data(df, path=None):
    path = path or PROCESSED_DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean_data(path=None):
    path = path or PROCESSED_DATA_PATH
    df = pd.read_csv(path)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    return df


def get_data_dictionary():
    return pd.DataFrame(
        [(k, v) for k, v in DATA_DICTIONARY.items()],
        columns=["Column", "Description"],
    )


def get_data_summary(df):
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.notnull().sum(),
        "null": df.isnull().sum(),
        "unique": df.nunique(),
    })
    return summary


def run_pipeline():
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    output_path = save_clean_data(clean_df)
    return clean_df, output_path


if __name__ == "__main__":
    clean_df, output_path = run_pipeline()
    print(f"Raw data shape: {load_raw_data().shape}")
    print(f"Clean data shape: {clean_df.shape}")
    print(f"Saved to: {output_path}")
    print()
    print(get_data_summary(clean_df))

