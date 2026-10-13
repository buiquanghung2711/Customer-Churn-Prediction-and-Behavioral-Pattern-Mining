#Clean data
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])   

df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
df["Partner"] = df["Partner"].map({"No": 0, "Yes": 1})
df["Dependents"] = df["Dependents"].map({"No": 0, "Yes": 1})
df["PhoneService"] = df["PhoneService"].map({"No": 0, "Yes": 1})
df["PaperlessBilling"] = df["PaperlessBilling"].map({"No": 0, "Yes": 1})
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

df_encoded = pd.get_dummies(df, columns=[
    "MultipleLines", 
    "InternetService", 
    "OnlineSecurity", 
    "OnlineBackup", 
    "DeviceProtection", 
    "TechSupport", 
    "StreamingTV", 
    "StreamingMovies", 
    "Contract", 
    "PaymentMethod"], 
    dtype=int)

df_encoded.to_csv('data/processed/WA_Fn-UseC_-Telco-Customer-Churn-processed.csv', index=False)