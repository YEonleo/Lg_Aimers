import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from modules.Encoder import series_decomp, Encoder, EncoderLayer, my_Layernorm
from modules.Decoder import DecoderLayer, Decoder
from modules.Embedding import DataEmbedding_wo_pos
from modules.AutoCorrelation import AutoCorrelationLayer,AutoCorrelation


class Model(nn.Module):
    def __init__(self,CFG):
        super(Model, self).__init__()
        self.seq_len = CFG['TRAIN_WINDOW_SIZE'] + CFG['PREDICT_SIZE'] # 전체 사용할 데이터 사이즈
        self.label_len = CFG['TRAIN_WINDOW_SIZE']
        self.pred_len = CFG['PREDICT_SIZE']
        self.output_attention = CFG['output_attention']

        #Decomp
        kernel_size = 25 #이동평균 일단 5일로설정
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(CFG['enc_in'], CFG['d_model'], 5, 11, CFG['embed'], CFG['freq'], CFG['drop_out'])
        self.dec_embedding = DataEmbedding_wo_pos(CFG['enc_in'], CFG['d_model'], 5, 11, CFG['embed'], CFG['freq'], CFG['drop_out'])


        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, CFG['factor'], attention_dropout=CFG['drop_out'],
                                        output_attention=CFG['output_attention']),
                        CFG['d_model'], CFG['n_heads']),
                    CFG['d_model'],
                    CFG['d_ff'],
                    moving_avg=25,
                    dropout=0.1,
                    activation='gelu'
                ) for l in range(CFG['e_layers'])
            ],
            norm_layer=my_Layernorm(CFG['d_model'])
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, CFG['factor'], attention_dropout=CFG['drop_out'],
                                        output_attention=False),
                        CFG['d_model'], CFG['n_heads']),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, CFG['factor'], attention_dropout=CFG['drop_out'],
                                        output_attention=False),
                        CFG['d_model'], CFG['n_heads']),
                    CFG['d_model'],
                    CFG['c_out'],
                    CFG['d_ff'],
                    moving_avg=25,
                    dropout=0.1,
                    activation='gelu'
                )
                for l in range(CFG['d_layers'])
            ],
            norm_layer=my_Layernorm(CFG['d_model']),
            projection=nn.Linear(CFG['d_model'], CFG['c_out'], bias=True)
        )

    def forward(self, product_data, x_enc, x_mark_enc, x_dec, x_mark_dec,
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

            # decomp init
            mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            zeros = torch.zeros([x_dec.shape[0], self.pred_len, 1], device=x_enc.device)
            
            seasonal_init, trend_init = self.decomp(x_enc)  # 판매량만 사용해서 구함
            # decoder input
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
            seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
            # enc
            enc_out = self.enc_embedding(product_data, x_enc, x_mark_enc)  # 제품 정보 추가
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

            # dec
            dec_product_data = product_data[:, -1, :].unsqueeze(1).repeat(1, self.seq_len, 1)

            dec_out = self.dec_embedding(dec_product_data, seasonal_init, x_mark_dec)
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                    trend=trend_init)
            # final
            dec_out = trend_part + seasonal_part

            if self.output_attention:
                return dec_out[:, -self.pred_len:, :], attns
            else:
                return dec_out[:, -self.pred_len:, :]
