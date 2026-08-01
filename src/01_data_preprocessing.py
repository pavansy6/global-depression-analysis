import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_nhanes_us_data():
    print("1. Processing US CDC NHANES Dataset...")
    
    dpq = pd.read_sas(os.path.join(RAW_DIR, "DPQ_J.xpt"), format="xport")
    demo = pd.read_sas(os.path.join(RAW_DIR, "DEMO_J.xpt"), format="xport")
    slq = pd.read_sas(os.path.join(RAW_DIR, "SLQ_J.xpt"), format="xport")
    
    for df in [dpq, demo, slq]:
        df["SEQN"] = df["SEQN"].astype(int)
        
    nhanes = dpq.merge(demo, on="SEQN", how="inner").merge(slq, on="SEQN", how="inner")
    
    phq_cols = [f"DPQ0{i}0" for i in range(1, 10)]
    for col in phq_cols:
        nhanes[col] = nhanes[col].replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})
        
    nhanes["phq9_score"] = nhanes[phq_cols].sum(axis=1)
    nhanes["depression_risk"] = (nhanes["phq9_score"] >= 10).astype(int)
    
    us_clean = pd.DataFrame({
        "SEQN": nhanes["SEQN"],
        "Age": nhanes["RIDAGEYR"],
        "Gender": nhanes["RIAGENDR"].map({1: 0, 2: 1}), # 0 = Male, 1 = Female
        "Sleep_Hours": nhanes["SLD012"].replace({77: np.nan, 99: np.nan}),
        "Sleep_Disorder_History": nhanes["SLQ050"].map({1: 1, 2: 0, 7: np.nan, 9: np.nan}),
        "Poverty_Income_Ratio": nhanes["INDFMPIR"],
        "depression_risk": nhanes["depression_risk"],
        "Country": "USA"
    })
    
    out_path = os.path.join(PROCESSED_DIR, "nhanes_us_clean.csv")
    us_clean.to_csv(out_path, index=False)
    print(f"   -> US NHANES clean data saved to: {out_path} (Shape: {us_clean.shape})")
    return us_clean

def process_india_data():
    print("\n2. Processing India Survey Dataset...")
    
    csv_path = os.path.join(RAW_DIR, "final_depression_dataset_1.csv")
    df = pd.read_csv(csv_path)
    
    target_col = "Depression" if "Depression" in df.columns else df.columns[-1]
    
    mapping = {
        "Yes": 1, "No": 0, "yes": 1, "no": 0,
        "True": 1, "False": 0, True: 1, False: 0,
        "1": 1, "0": 0, 1: 1, 0: 0
    }
    
    df["depression_risk"] = df[target_col].map(mapping)
    if df["depression_risk"].isna().any():
        df["depression_risk"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        
    df["depression_risk"] = df["depression_risk"].astype(int)
        
    df["Country"] = "IND"
    
    out_path = os.path.join(PROCESSED_DIR, "india_survey_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"   -> India clean data saved to: {out_path} (Shape: {df.shape})")
    return df

def process_osmi_tech_data():
    print("\n3. Processing OSMI Tech Survey Dataset (survey.csv)...")
    
    csv_path = os.path.join(RAW_DIR, "survey.csv")
    df = pd.read_csv(csv_path)
    
    df["depression_risk"] = df["treatment"].map({"Yes": 1, "No": 0})
    
    # Drop arbitrary comments/state columns that add noise
    cols_to_drop = ["comments", "state", "Timestamp", "treatment"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Label as Global Tech dataset
    df["Country"] = "GLOBAL_TECH"
    
    out_path = os.path.join(PROCESSED_DIR, "osmi_tech_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"   -> OSMI Tech clean data saved to: {out_path} (Shape: {df.shape})")
    return df

if __name__ == "__main__":
    print("--- STARTING MULTI-REGION DATA PREPROCESSING ---")
    us_data = process_nhanes_us_data()
    india_data = process_india_data()
    tech_data = process_osmi_tech_data()
    print("\nPre-processing completed successfully! Clean CSVs are inside `data/processed/`.")