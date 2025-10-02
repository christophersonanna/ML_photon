# -*- coding: utf-8 -*-

#import awkward as ak
#import pyarrow.parquet as pq
import z_config as config
import pandas as pd #will change later

#loading data
#parquet_file = pq.ParquetFile(config.PROCESSED_DATA_PATH)
#parquet_file = pd.read_parquet(config.PROCESSED_DATA_PATH)
#print(parquet_file.columns)

#for testing the scripts for now
X_test = pd.read_csv(config.PROCESSED_DATA_PATH_TEST, delimiter=',')
X_train = pd.read_csv(config.PROCESSED_DATA_PATH_TRAIN, delimiter=',')
