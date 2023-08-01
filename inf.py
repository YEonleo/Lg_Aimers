import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os

from modules.datasets import make_train_data,make_predict_data,CustomDataset
from modules.model import BaseModel
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timezone, timedelta

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'TRAIN_WINDOW_SIZE':90, # 90일치로 학습
    'PREDICT_SIZE':21, # 21일치 예측
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':9192,
    'SEED':42
}

def inference(model, test_loader, device):
    predictions = []
        
    with torch.no_grad():
        for X in tqdm(iter(test_loader)):
            X = X.to(device)
                
            output = model(X)
                
            # 모델 출력인 output을 CPU로 이동하고 numpy 배열로 변환
            output = output.cpu().numpy()
                
            predictions.extend(output)
        
        return np.array(predictions)

if __name__ == '__main__':
    set_seed(CFG['SEED'], device) #random seed 정수로 고정.
    
    # 폴더 경로 설정
    folder_path = "모델 저장 경로"

    # 폴더 경로가 존재하지 않는 경우 폴더 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    train_data = pd.read_csv('./data/train.csv').drop(columns=['ID', '제품'])

    scale_max_dict = {}
    scale_min_dict = {}

    for idx in tqdm(range(len(train_data))):
        maxi = np.max(train_data.iloc[idx,4:])
        mini = np.min(train_data.iloc[idx,4:])
        
        if maxi == mini :
            train_data.iloc[idx,4:] = 0
        else:
            train_data.iloc[idx,4:] = (train_data.iloc[idx,4:] - mini) / (maxi - mini)
        
        scale_max_dict[idx] = maxi
        scale_min_dict[idx] = mini

    label_encoder = LabelEncoder()
    categorical_columns = ['대분류', '중분류', '소분류', '브랜드']

    for col in categorical_columns:
        label_encoder.fit(train_data[col])
        train_data[col] = label_encoder.transform(train_data[col])
        
    test_input = make_predict_data(train_data,CFG['TRAIN_WINDOW_SIZE']) #test_data부분
    
    test_dataset = CustomDataset(test_input, None)
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    
    infer_model = BaseModel()
    infer_model.load_state_dict(torch.load(folder_path, map_location=device))
    infer_model.to(device)
    infer_model.eval()
    
    pred = inference(infer_model, test_loader, device)

    for idx in range(len(pred)):
        pred[idx, :] = pred[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
        
    # 결과 후처리
    pred = np.round(pred, 0).astype(int)
    submit = pd.read_csv('./data/sample_submission.csv')
    submit.iloc[:,1:] = pred
    submit.to_csv('./baseline_submit.csv', index=False)