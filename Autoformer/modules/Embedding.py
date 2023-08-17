import torch
import math
import torch.nn as nn

def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")
    
    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1
    
    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True
    
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='d'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, major_size,minor_size, embed_type='fixed', freq='d', dropout=0.1):#대분류,중분류 사이즈별 임베딩
        super(DataEmbedding_wo_pos, self).__init__()
        self.major_embedding = nn.Embedding(major_size, d_model)
        self.minor_embedding = nn.Embedding(minor_size, d_model)

        
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, product_data, x, x_mark):
        x_embed = self.value_embedding(x)
        
        major_info = product_data[:, :, 0] # 대분류
        minor_info = product_data[:, :, 1] # 중분류
        
        major_embed = self.major_embedding(major_info)
        minor_embed = self.minor_embedding(minor_info)
        
        product_embed = major_embed + minor_embed
        
        x = x_embed + product_embed + self.temporal_embedding(x_mark)
        
        return self.dropout(x)



class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='d'):  # freq를 'd'로 변경
        super(TemporalEmbedding, self).__init__()

        year_size = 2  # 예시로 임의의 년도 크기 사용 (2022, 2023)
        month_size = 13
        day_size = 32

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 'd':
            self.year_embed = Embed(year_size, d_model)
            self.month_embed = Embed(month_size, d_model)
            self.day_embed = Embed(day_size, d_model)

    def forward(self, x):
        x = x.long()

        year_x = self.year_embed(x[:, :, 0])
        month_x = self.month_embed(x[:, :, 1])
        day_x = self.day_embed(x[:, :, 2])

        return year_x + month_x + day_x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]