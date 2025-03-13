"""
This script unzips the zip files in the data/zip folder and extracts the *_HH_*.csv files to the data/raw folder. Meanwhile, for each csv file, we also preprocess the data by selecting certain columns and rows and then save the processed data to the data/raw folder. Visit https://fluxnet.org/data/fluxnet2015-dataset/subset-data-product/ for more information about the dataset.
"""
import os
import zipfile
import glob
import pandas as pd
from tqdm import tqdm

zip_dir = 'data/zip'
extract_dir = 'data/raw'
cols = ['TA_F', 'PA_F', 'P_F', 'WS_F', 'CO2_F_MDS', 'VPD_F']
selected_columns = ['TIMESTAMP_START'] + cols

zip_files = glob.glob(zip_dir + '/*.zip')

os.makedirs(extract_dir, exist_ok=True)

for zip_file in tqdm(zip_files):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if "_HH_" in file and file.endswith(".csv"):
                with zip_ref.open(file) as f:
                    df = pd.read_csv(f, usecols = selected_columns) 
                    
                    start = df['TIMESTAMP_START'].min()
                    end = df['TIMESTAMP_START'].max() 
                    df = df[(df['TIMESTAMP_START'] >= 201001010000) & (df['TIMESTAMP_START'] <= 201312312330)]
                    missing = min(df[cols].min()) <= -9999
                    if start <= 201001010000 and end >= 201312312330 and not missing: 
                        df.to_csv(extract_dir + '/' + file, index=False) 
                        print(f"Extracted {file} to {extract_dir}")