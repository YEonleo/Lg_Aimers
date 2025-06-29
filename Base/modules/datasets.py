import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
            
    def __getitem__(self, index):
        if self.Y is not None:
            return torch.Tensor(self.X[index]), torch.Tensor(self.Y[index])
        return torch.Tensor(self.X[index])
        
    def __len__(self):
        return len(self.X)

def make_train_data(data, train_size, predict_size):
    '''
    학습 기간 블럭, 예측 기간 블럭의 세트로 데이터를 생성
    data : 일별 판매량
    train_size : 학습에 활용할 기간
    predict_size : 추론할 기간
    '''
    num_rows = len(data)
    window_size = train_size + predict_size
    
    input_data = np.empty((num_rows * (len(data.columns) - window_size + 1), train_size, len(data.iloc[0, :4]) + 1))
    target_data = np.empty((num_rows * (len(data.columns) - window_size + 1), predict_size))
    
    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :4])
        sales_data = np.array(data.iloc[i, 4:])
        
        for j in range(len(sales_data) - window_size + 1):
            window = sales_data[j : j + window_size]
            temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window[:train_size]))
            input_data[i * (len(data.columns) - window_size + 1) + j] = temp_data
            target_data[i * (len(data.columns) - window_size + 1) + j] = window[train_size:]
    
    return input_data, target_data

def make_predict_data(data, train_size):
        '''
        평가 데이터(Test Dataset)를 추론하기 위한 Input 데이터를 생성
        data : 일별 판매량
        train_size : 추론을 위해 필요한 일별 판매량 기간 (= 학습에 활용할 기간)
        '''
        num_rows = len(data)
        
        input_data = np.empty((num_rows, train_size, len(data.iloc[0, :4]) + 1))
        
        for i in tqdm(range(num_rows)):
            encode_info = np.array(data.iloc[i, :4])
            sales_data = np.array(data.iloc[i, -train_size:])
            
            window = sales_data[-train_size : ]
            temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window[:train_size]))
            input_data[i] = temp_data
        
        return input_data

