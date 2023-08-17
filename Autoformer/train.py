import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import wandb
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from modules.datasets import make_train_data,CustomDataset,min_max_scaler,make_train_data_parallel,make_train_data_optimized
from modules.model import Model
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timezone, timedelta

from transformers import get_linear_schedule_with_warmup

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'TRAIN_WINDOW_SIZE':90, # 90일치로 학습 pred_len
    'PREDICT_SIZE':21, # 21일치 예측 label_len
    'freq':'d',# seasonal, trend 정보를 어떤걸 기준으로 할지
    'enc_in':1,#encoder input size
    'dec_in':1,#decoder input size
    'embed':'fixed',#time features encoding, options:[timeF, fixed, learned]
    'drop_out':0.1,
    'EPOCHS':10,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':512,
    'd_model':512,
    'd_ff':2048,
    'e_layers':2,
    'd_layers':1,
    'n_heads': 8,
    'factor': 1,
    'c_out': 1,
    'SEED':41,
    'EPS':1e-8,
    'output_attention':'store_true'
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

def train(model, optimizer, train_loader, val_loader, device, model_scheduler):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    best_loss = 9999999
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        
        for product_data, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(iter(train_loader)):
            product_data = product_data.to(device)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)
            
            # decoder input
            dec_inp = torch.cat([batch_x, torch.zeros_like(batch_y).unsqueeze(-1)], dim=1).to(device)

            # decoder time info
            dec_mark_inp = torch.cat([batch_x_mark, batch_y_mark], dim=1).to(device)

            output = model(product_data, batch_x, batch_x_mark, dec_inp, dec_mark_inp)[0]
            output = output.squeeze(-1)

            batch_y = batch_y[:, -CFG['PREDICT_SIZE']:].to(device)
            optimizer.zero_grad()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        val_loss = validation(model, val_loader, criterion, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]')
        # Log metrics to the current fold's wandb run
        wandb.log({"Val Loss": val_loss, "Train Loss": np.mean(train_loss)})
        torch.save(model.state_dict(), os.path.join(folder_path, f'{epoch}TF model.pt'))      
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            print('Model Saved')
            
    return best_model


def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    
    with torch.no_grad():
        for product_data, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(iter(train_loader)):
            product_data = product_data.to(device)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)
            
            # decoder input
            dec_inp = torch.cat([batch_x, torch.zeros_like(batch_y).unsqueeze(-1)], dim=1).to(device)

            # decoder time info
            dec_mark_inp = torch.cat([batch_x_mark, batch_y_mark], dim=1).to(device)

            output = model(product_data, batch_x, batch_x_mark, dec_inp, dec_mark_inp)[0]
            output = output.squeeze(-1)
            
            batch_y = batch_y[:, -CFG['PREDICT_SIZE']:].to(device)
            
            loss = criterion(output, batch_y)
            val_loss.append(loss.item())

    return np.mean(val_loss)


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
    run_name = f"{train_serial}_TF_model"
    wandb.init(name=run_name+"Autoformer",    
               config=CFG,
               project="lg_AIMERS",)
    CFG = wandb.config
    seed_everything(CFG['SEED']) # Seed 고정
    
    # 폴더 경로 설정
    folder_path = "./results/" + train_serial

    # 폴더 경로가 존재하지 않는 경우 폴더 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 데이터 로딩 시에 'ID' 열을 인덱스로 설정
    train_data = pd.read_csv('./data/train.csv').drop(columns=['ID','제품','소분류','브랜드'])
    columns_to_remove = [col for col in train_data.columns if '2023-02-23' <= col <= '2023-04-04']
    train_data = train_data.drop(columns=columns_to_remove)

    train_data,scale_max_dict,scale_min_dict = min_max_scaler(train_data)
    
    #날짜 연 월 정보 추출
    date_columns = [col for col in train_data.columns if '-' in col]
    date_info = pd.DataFrame(date_columns, columns=["full_date"])

    # 연, 월, 일로 분리
    date_info["year"] = date_info["full_date"].apply(lambda x: int(x.split("-")[0]))
    date_info["month"] = date_info["full_date"].apply(lambda x: int(x.split("-")[1]))
    date_info["day"] = date_info["full_date"].apply(lambda x: int(x.split("-")[2]))

    # train_df에 연, 월, 일 정보 추가
    for idx, col in enumerate(date_columns):
        train_data.insert(train_data.columns.get_loc(col) + 1, col + "_year", date_info.iloc[idx]["year"])
        train_data.insert(train_data.columns.get_loc(col) + 2, col + "_month", date_info.iloc[idx]["month"])
        train_data.insert(train_data.columns.get_loc(col) + 3, col + "_day", date_info.iloc[idx]["day"])
        
    train_input, train_target = make_train_data_optimized(train_data,CFG['TRAIN_WINDOW_SIZE'],CFG['PREDICT_SIZE'])
    
    
    data_len = len(train_input)
    val_input = train_input[-int(data_len*0.2):]
    val_target = train_target[-int(data_len*0.2):]
    train_input = train_input[:-int(data_len*0.2)]
    train_target = train_target[:-int(data_len*0.2)]
    
    train_dataset = CustomDataset(train_input, train_target)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val_input, val_target)
    val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
        

    model = Model(CFG)

    #하이퍼 파라미터 튜닝 부분
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        model_param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        model_optimizer_grouped_parameters = [
                {'params': [p for n, p in model_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in model_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
            ]
    else:
        model_param_optimizer = list(model.classifier.named_parameters())
        model_optimizer_grouped_parameters = [{"params": [p for n, p in model_param_optimizer]}]

    model_optimizer = torch.optim.AdamW(
            model_optimizer_grouped_parameters,
            lr=CFG["LEARNING_RATE"],
            eps=CFG["EPS"],
        )
        
    epochs = CFG['EPOCHS']
    max_grad_norm = 1.0
    total_steps = epochs * len(train_loader)

    model_scheduler = get_linear_schedule_with_warmup(
            model_optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
    infer_model = train(model, model_optimizer, train_loader, val_loader, device, model_scheduler)#, fold)
    
    wandb.finish()