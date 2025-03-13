import pandas as pd  
import glob 
from tqdm import tqdm
import os 
import numpy as np

source_dir = "data/raw"
extract_dir = "data/processed"

os.makedirs(extract_dir, exist_ok=True)

columns = ['TA_F', 'PA_F', 'WS_F', 'CO2_F_MDS', 'VPD_F'] 

files = glob.glob(source_dir + "/*.csv")
for file in tqdm(files): 
    site = file.split("_")[1]

    df = pd.read_csv(file)
    df = df[(df['TIMESTAMP_START'] >= 201001040000) & (df['TIMESTAMP_START'] < 201312300000)]
    df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"], format="%Y%m%d%H%M")
    df['week'] = df["TIMESTAMP_START"].dt.to_period("W")
    
    corr = df.groupby("week")[columns].apply(lambda x: x.corr())
    determinant = df.groupby("week")[columns].apply(lambda x: np.linalg.det(x.corr())) # compute the determinant of the correlation matrix, and check if it is not NaN
    
    if not determinant.isna().any().any():
        corr.to_csv(extract_dir + f"/{site}_corr.csv") 

    cov = df.groupby("week")[columns].apply(lambda x: x.cov())
    determinant = df.groupby("week")[columns].apply(lambda x: np.linalg.det(x.cov())) # compute the determinant of the covariance matrix, and check if it is not NaN

    if not determinant.isna().any().any() and np.min(determinant) > 0:
        cov.to_csv(extract_dir + f"/{site}_cov.csv")

