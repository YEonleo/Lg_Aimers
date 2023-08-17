import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from joblib import Parallel, delayed


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        self.sales_feature_size = 3  # 대분류, 중분류, 판매량
        self.time_feature_size = 3  # 연, 월, 일
            
    def __getitem__(self, index):
    # 판매량 추출
        sales_data = self.X[index, :, 2:3]  # 판매량은 3번째 칼럼에 위치한다고 가정
        # 대분류, 중분류 정보 추출
        product_data = self.X[index, :, :2]  # 대분류와 중분류는 처음 두 칼럼에 위치한다고 가정
        
        # 연, 월, 일 정보 추출
        input_time_features = self.X[index, :, self.sales_feature_size:]
        
        # target_data에서 판매량 및 연, 월, 일 정보 추출
        target_sales = self.Y[index, :, 0]
        target_time_features = self.Y[index, :, 1:]
        
        return product_data.long(), sales_data, target_sales, input_time_features, target_time_features
        
    def __len__(self):
        return len(self.X)



def min_max_scaler(train_data):
    scale_max_dict = {}
    scale_min_dict = {}

    numeric_cols = train_data.columns[2:]

    # 각 행의 최댓값과 최솟값 계산
    min_values = train_data[numeric_cols].min(axis=1)
    max_values = train_data[numeric_cols].max(axis=1)

    # 각 행의 범위(max - min) 계산하고, 범위가 0인 경우 1로 대체
    ranges = max_values - min_values
    ranges[ranges == 0] = 1

    # min-max scaling 수행
    train_data[numeric_cols] = (train_data[numeric_cols].subtract(min_values, axis=0)).div(ranges, axis=0)

    # max와 min 값을 dictionary 형태로 저장
    scale_min_dict = min_values.to_dict()
    scale_max_dict = max_values.to_dict()

    # Label Encoding
    label_encoder = LabelEncoder()
    categorical_columns = ['대분류', '중분류']#, '대분류',]

    for col in categorical_columns:
        label_encoder.fit(train_data[col])
        train_data[col] = label_encoder.transform(train_data[col])

    return train_data, scale_max_dict, scale_min_dict

def make_train_data_optimized(data, train_size, predict_size):
    num_rows = len(data)
    window_size = train_size + predict_size
    
    # 컬럼의 인덱스를 추출하여 numpy array로 변환
    data_np = data.to_numpy()
    sales_indices = np.arange(2, 2 + len(data.columns[data.columns.str.contains('-')]))
    
    input_data = np.empty((num_rows * (len(sales_indices) - window_size + 1), train_size, 6))
    target_data = np.empty((num_rows * (len(sales_indices) - window_size + 1), predict_size, 4))
    
    for i in tqdm(range(num_rows)):
        encode_info = data_np[i, :2]
        
        for j in range(len(sales_indices) - window_size + 1):
            sales_window = data_np[i, sales_indices[j:j+train_size]]
            
            year_window_indices = [idx + 1 for idx in sales_indices[j:j+train_size]]
            month_window_indices = [idx + 2 for idx in sales_indices[j:j+train_size]]
            day_window_indices = [idx + 3 for idx in sales_indices[j:j+train_size]]

            year_window = data_np[i, year_window_indices] - 2022
            month_window = data_np[i, month_window_indices]
            day_window = data_np[i, day_window_indices]
            
            combined_window = np.column_stack((np.tile(encode_info, (train_size, 1)), sales_window, year_window, month_window, day_window))
            input_data[num_rows * i + j] = combined_window

            target_sales = data_np[i, sales_indices[j+train_size:j+window_size]]
            target_year = data_np[i, year_window_indices[1:predict_size+1]] - 2022
            target_month = data_np[i, month_window_indices[1:predict_size+1]]
            target_day = data_np[i, day_window_indices[1:predict_size+1]]
            
            combined_target = np.column_stack((target_sales, target_year, target_month, target_day))
            target_data[num_rows * i + j] = combined_target

    return input_data, target_data



def make_train_data(data, train_size, predict_size):
    '''
    학습 기간 블럭, 예측 기간 블럭의 세트로 데이터를 생성
    data : 일별 판매량
    train_size : 학습에 활용할 기간
    predict_size : 추론할 기간
    '''
    num_rows = len(data)
    window_size = train_size + predict_size
    
    sales_cols = [col for col in data.columns if '-' in col and not any(substr in col for substr in ['_year', '_month', '_day'])]
    
    # Adjust the shape for the increased feature size (encode_info + sales + year + month + day)
    input_data = np.empty((num_rows * (len(sales_cols) - window_size + 1), train_size, 6))
    target_data = np.empty((num_rows * (len(sales_cols) - window_size + 1), predict_size, 4)) # 판매량, 연도, 월, 일을 위한 차원 추가
    
    for i in tqdm(range(num_rows)):
        encode_info = data.iloc[i, :2].values  # '대분류' and '중분류'
        
        for j in range(len(sales_cols) - window_size + 1):
            sales_window = data[sales_cols[j: j + train_size]].iloc[i].values
            year_window = (data[[col + "_year" for col in sales_cols[j: j + train_size]]].iloc[i].values - 2022)
            month_window = data[[col + "_month" for col in sales_cols[j: j + train_size]]].iloc[i].values
            day_window = data[[col + "_day" for col in sales_cols[j: j + train_size]]].iloc[i].values
            
            # Combine the windows
            combined_window = np.column_stack((np.tile(encode_info, (train_size, 1)), sales_window, year_window, month_window, day_window))
            input_data[i * (len(sales_cols) - window_size + 1) + j] = combined_window

            # target 데이터에 대한 연, 월, 일 정보 가져오기
            target_year = (data[[col + "_year" for col in sales_cols[j + train_size: j + window_size]]].iloc[i].values - 2022)
            target_month = data[[col + "_month" for col in sales_cols[j + train_size: j + window_size]]].iloc[i].values
            target_day = data[[col + "_day" for col in sales_cols[j + train_size: j + window_size]]].iloc[i].values
            target_sales = data[sales_cols[j + train_size: j + window_size]].iloc[i].values

            # Combine the target data
            combined_target = np.column_stack((target_sales, target_year, target_month, target_day))
            target_data[i * (len(sales_cols) - window_size + 1) + j] = combined_target
    
    return input_data, target_data


def make_predict_data(data, train_size):
        '''
        평가 데이터(Test Dataset)를 추론하기 위한 Input 데이터를 생성
        data : 일별 판매량
        train_size : 추론을 위해 필요한 일별 판매량 기간 (= 학습에 활용할 기간)
        '''
        num_rows = len(data)
        
        input_data = np.empty((num_rows, train_size, len(data.iloc[0, :1]) + 1))
        
        for i in tqdm(range(num_rows)):
            encode_info = np.array(data.iloc[i, :1])
            sales_data = np.array(data.iloc[i, -train_size:])
            
            window = sales_data[-train_size : ]
            temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window[:train_size]))
            input_data[i] = temp_data
        
        return input_data

def process_row(i, data, sales_cols, train_size, predict_size, window_size, encode_info):
    num_rows = len(data)
    input_data_row = np.empty((len(sales_cols) - window_size + 1, train_size, 6))
    target_data_row = np.empty((len(sales_cols) - window_size + 1, predict_size, 4))
    
    for j in range(len(sales_cols) - window_size + 1):
        sales_window = data[sales_cols[j: j + train_size]].iloc[i].values
        year_window = (data[[col + "_year" for col in sales_cols[j: j + train_size]]].iloc[i].values - 2022)
        month_window = data[[col + "_month" for col in sales_cols[j: j + train_size]]].iloc[i].values
        day_window = data[[col + "_day" for col in sales_cols[j: j + train_size]]].iloc[i].values
        
        combined_window = np.column_stack((np.tile(encode_info, (train_size, 1)), sales_window, year_window, month_window, day_window))
        input_data_row[j] = combined_window

        target_year = (data[[col + "_year" for col in sales_cols[j + train_size: j + window_size]]].iloc[i].values - 2022)
        target_month = data[[col + "_month" for col in sales_cols[j + train_size: j + window_size]]].iloc[i].values
        target_day = data[[col + "_day" for col in sales_cols[j + train_size: j + window_size]]].iloc[i].values
        target_sales = data[sales_cols[j + train_size: j + window_size]].iloc[i].values

        combined_target = np.column_stack((target_sales, target_year, target_month, target_day))
        target_data_row[j] = combined_target

    return input_data_row, target_data_row

def make_train_data_parallel(data, train_size, predict_size):
    num_rows = len(data)
    window_size = train_size + predict_size
    
    sales_cols = [col for col in data.columns if '-' in col and not any(substr in col for substr in ['_year', '_month', '_day'])]
    
    input_data = np.empty((num_rows * (len(sales_cols) - window_size + 1), train_size, 6))
    target_data = np.empty((num_rows * (len(sales_cols) - window_size + 1), predict_size, 4))
    
    results = Parallel(n_jobs=-1)(delayed(process_row)(i, data, sales_cols, train_size, predict_size, window_size, data.iloc[i, :2].values) for i in range(num_rows))
    
    for i, (input_data_row, target_data_row) in enumerate(results):
        input_data[i * (len(sales_cols) - window_size + 1): (i + 1) * (len(sales_cols) - window_size + 1)] = input_data_row
        target_data[i * (len(sales_cols) - window_size + 1): (i + 1) * (len(sales_cols) - window_size + 1)] = target_data_row

    return input_data, target_data
