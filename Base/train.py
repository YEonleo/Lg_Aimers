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

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'TRAIN_WINDOW_SIZE':90, # 90일치로 학습
    'PREDICT_SIZE':21, # 21일치 예측
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':2048,
    'SEED':42
}

# Train Serial
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    best_loss = 9999999
    best_model = None
        
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        train_mae = []
        for X, Y in tqdm(iter(train_loader)):
            X = X.to(device)
            Y = Y.to(device)
                
            optimizer.zero_grad()
                
            output = model(X)
            loss = criterion(output, Y)
                
            loss.backward()
            optimizer.step()
                
            train_loss.append(loss.item())
            
        val_loss = validation(model, val_loader, criterion, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]')
            
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model
            print('Model Saved')
    return best_model


def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
        
    with torch.no_grad():
        for X, Y in tqdm(iter(val_loader)):
            X = X.to(device)
            Y = Y.to(device)
                
            output = model(X)
            loss = criterion(output, Y)
                
        val_loss.append(loss.item())
    return np.mean(val_loss)

if __name__ == '__main__':
    
    seed_everything(CFG['SEED']) # Seed 고정
    
    # 폴더 경로 설정
    folder_path = "./results/" + train_serial

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

    train_input, train_target = make_train_data(train_data,CFG['TRAIN_WINDOW_SIZE'],CFG['PREDICT_SIZE'])
    #test_input = make_predict_data(train_data,CFG['TRAIN_WINDOW_SIZE']) #test_data부분

    # Train adn test split (시계열 데이터 이므로 앞에서부터 몇퍼센트 짜르냐로 확인 하는 부분)
    data_len = len(train_input)
    val_input = train_input[-int(data_len*0.1):]
    val_target = train_target[-int(data_len*0.1):]
    train_input = train_input[:-int(data_len*0.1)]
    train_target = train_target[:-int(data_len*0.1)]

    train_dataset = CustomDataset(train_input, train_target)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val_input, val_target)
    val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = BaseModel()
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    infer_model = train(model, optimizer, train_loader, val_loader, device)
    torch.save(infer_model.state_dict(), os.path.join(folder_path, "infer_model.pt"))
