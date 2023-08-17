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

from modules.datasets import make_train_data,make_predict_data,CustomDataset,min_max_scaler
from modules.model import BaseModel,TFModel,TimeSeriesTransformer,Encoder,Decoder,Seq2Seq
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timezone, timedelta

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'TRAIN_WINDOW_SIZE':90, # 90일치로 학습
    'PREDICT_SIZE':21, # 21일치 예측
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':1024,
    'SEED':42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def inference(model, test_loader, device):
    predictions = []
        
    with torch.no_grad():
        for X in tqdm(iter(test_loader)):
            X = X.to(device)
                
            output = model(X)
            output = output.squeeze(-1)
            # 모델 출력인 output을 CPU로 이동하고 numpy 배열로 변환
            output = output.cpu().numpy()
                
            predictions.extend(output)
        
        return np.array(predictions)

if __name__ == '__main__':
    seed_everything(CFG['SEED']) # Seed 고정
    
    # 폴더 경로 설정
    folder_path = "./results/20230816_112125/10TF model.pt" #앞20 검증 뒤80
    folder_path_2 = "./results/20230816_112125/12TF model.pt" #40 20 40
    folder_path_3 = "./results/20230816_112125/13TF model.pt" #40 20 40
    folder_path_4 = "./results/20230816_112125/14TF model.pt" #40 20 40
    folder_path_5 = "./results/20230816_112125/15TF model.pt" #40 20 40

    
    train_data = pd.read_csv('./data/train_with_cluster_reordered.csv')
    train_data = train_data.drop(columns=['ID', '제품', '브랜드', '소분류','대분류','중분류'])
    columns_to_remove = [col for col in train_data.columns if '2023-02-23' <= col <= '2023-03-28']
    train_data = train_data.drop(columns=columns_to_remove)

    train_data,scale_max_dict,scale_min_dict = min_max_scaler(train_data)
    test_input = make_predict_data(train_data,CFG['TRAIN_WINDOW_SIZE']) #test_data부분
    
    test_dataset = CustomDataset(test_input, None)
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    
        # 모델 초기화 및 학습
    encoder_input_size = 2  # 예시로 설정, 실제 입력 차원에 따라 변경 필요
    encoder_hidden_size = 512
    encoder_num_layers = 1

    decoder_input_size = 1  # 예시로 설정, 실제 입력 차원에 따라 변경 필요
    decoder_hidden_size = 512
    decoder_output_size = 1  # 하루의 예측 출력
    decoder_num_layers = 1

    encoder = Encoder(input_size=encoder_input_size, 
                    hidden_size=encoder_hidden_size, 
                    num_layers=encoder_num_layers)

    decoder = Decoder(input_size=decoder_input_size, 
                    hidden_size=decoder_hidden_size, 
                    output_size=decoder_output_size, 
                    num_layers=decoder_num_layers)
    
    infer_model = Seq2Seq(encoder, decoder,21)
    infer_model.load_state_dict(torch.load(folder_path, map_location=device))
    infer_model.to(device)
    infer_model.eval()
    
    pred = inference(infer_model, test_loader, device)
    
    infer_model = Seq2Seq(encoder, decoder,21)
    infer_model.load_state_dict(torch.load(folder_path_2, map_location=device))
    infer_model.to(device)
    infer_model.eval()
    
    pred2 = inference(infer_model, test_loader, device)
    
    infer_model = Seq2Seq(encoder, decoder,21)
    infer_model.load_state_dict(torch.load(folder_path_3, map_location=device))
    infer_model.to(device)
    infer_model.eval()
    
    pred3 = inference(infer_model, test_loader, device)
    
    infer_model = Seq2Seq(encoder, decoder,21)
    infer_model.load_state_dict(torch.load(folder_path_4, map_location=device))
    infer_model.to(device)
    infer_model.eval()
    
    pred4 = inference(infer_model, test_loader, device)
    
    infer_model = Seq2Seq(encoder, decoder,21)
    infer_model.load_state_dict(torch.load(folder_path_5, map_location=device))
    infer_model.to(device)
    infer_model.eval()
    
    pred5 = inference(infer_model, test_loader, device)
    

    #각 모델의 예측값을 복원 (스케일 복원)
    for idx in range(len(pred)):
        pred[idx, :] = pred[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
        pred2[idx, :] = pred2[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
        pred3[idx, :] = pred3[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
        pred4[idx, :] = pred4[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
        pred5[idx, :] = pred5[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
    
    pred = (pred  + pred2 + pred3 + pred4 + pred5)/5
    pred = np.round(pred, 0).astype(int)
    submit = pd.read_csv('./data/sample_submission.csv')
    submit.iloc[:,1:] = pred
    submit.to_csv('./모두다합쳐4.csv', index=False)
    