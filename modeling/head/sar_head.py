'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-12 14:24:24
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-13 16:51:36
FilePath: /SVTR/modeling/head/sar_head.py
Description: 
'''
# coding: utf-8
import math
import torch
from torch import nn
import torch.nn.functional as F


class SAREncoder(nn.Module):
    def __init__(self,
        enc_bi_rnn=False,
        enc_drop_rnn=0.1,
        enc_gru=False,
        d_model=512,
        d_enc=512,
        mask=True,
        **kwargs
        ):
        super().__init__()
        assert isinstance(enc_bi_rnn, bool)
        assert isinstance(enc_drop_rnn, (int, float))
        assert 0 <= enc_drop_rnn < 1.0
        assert isinstance(enc_gru, bool)
        assert isinstance(d_model, int)
        assert isinstance(d_enc, int)
        assert isinstance(mask, bool)

        self.enc_bi_rnn = enc_bi_rnn
        self.enc_drop_rnn = enc_drop_rnn
        self.mask = mask

        # LSTM Encoder
        if enc_bi_rnn:
            bidirectional = True
        else:
            bidirectional = False

        kwargs = dict(
            input_size= d_model,
            hidden_size=d_enc,
            num_layers=2,
            dropout=enc_drop_rnn,
            bidirectional=bidirectional
        )

        if enc_gru:
            self.rnn_encoder = nn.GRU(**kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**kwargs)

        # Global feature transformation
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def forward(self, feat, img_metas=None):
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]

        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]

        h_feat = feat.shape[2]
        feat_v = F.max_pool2d(feat, kernel_size=(h_feat, 1), stride=1, padding=0)
        feat_v = feat_v.squeeze(2)
        feat_v = feat_v.permute([0, 2, 1])

        holistic_feat = self.rnn_encoder(feat_v)[0]
        if valid_ratios is not None:
            valid_hf = []
            T = holistic_feat.shape[1]
            for i in range(len(valid_ratios)):
                valid_step = min(T, math.ceil(T * valid_ratios[i])) - 1
                valid_hf.append(holistic_feat[i, valid_step, :])
            valid_hf = torch.stack(valid_hf, dim=0)
        else:
            valid_hf = holistic_feat[:, -1, :]

        holistic_feat = self.linear(valid_hf)        
        return holistic_feat


class BaseDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self, feat, out_enc, label=None, img_metas=None, train_mode=True):
        self.train_mode = train_mode

        if train_mode:
            return self.forward_train(feat, out_enc, label, img_metas)

        return self.forward_test(feat, out_enc, img_metas)


class ParallelSARDecoder(BaseDecoder):
    def __init__(self,
        out_channel,
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        dec_drop_rnn=0.0,
        dec_gru=False,
        d_model=512,
        d_enc=512,
        d_k=64,
        pred_dropout=0.1,
        max_text_length=30,
        mask=True,
        pred_concat=True,
        **kwargs):
        super().__init__()

        self.num_classes = out_channel
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.start_idx = out_channel - 2
        self.padding_idx = out_channel - 1
        self.max_seq_len = max_text_length
        self.mask = mask
        self.pred_concat = pred_concat

        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)

        # 2D attention layer
        self.conv1x1_1 = nn.Linear(decoder_rnn_out_size, d_k)
        self.conv3x3_1 = nn.Conv2d(d_model, d_k, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Linear(d_k, 1)

        # Decoder RNN layer
        if dec_bi_rnn:
            bidirectional = True
        else:
            bidirectional = False

        kwargs = dict(
            input_size=encoder_rnn_out_size,
            hidden_size=encoder_rnn_out_size,
            num_layers=2,
            dropout=dec_drop_rnn,
            bidirectional=bidirectional
        )

        if dec_gru:
            self.rnn_decoder = nn.GRU(**kwargs)
        else:
            self.rnn_decoder = nn.LSTM(**kwargs)

        # Decoder input embedding
        self.embedding = nn.Embedding(
            self.num_classes,
            encoder_rnn_out_size,
            padding_idx=self.padding_idx
        )

        # Prediction layer
        self.pred_dropout = nn.Dropout(pred_dropout)
        pred_num_classes = self.num_classes - 1
        if pred_concat:
            fc_in_channel = decoder_rnn_out_size + d_model + encoder_rnn_out_size
        else:
            fc_in_channel = d_model

        self.prediction = nn.Linear(fc_in_channel, pred_num_classes)

    def _2d_attention(self, decoder_input, feat, holistic_feat, valid_ratios=None):
        y = self.rnn_decoder(decoder_input)[0]

        attn_query = self.conv1x1_1(y)
        bsz, seq_len, attn_size = attn_query.shape
        attn_query = torch.unsqueeze(attn_query, dim=3)
        attn_query = torch.unsqueeze(attn_query, dim=4)
        
        attn_key = self.conv3x3_1(feat)
        attn_key = attn_key.unsqueeze(1)

        attn_weight = torch.tanh(torch.add(attn_key, attn_query))
        attn_weight = attn_weight.permute([0,1,3,4,2])
        attn_weight = self.conv1x1_2(attn_weight)
        
        bsz, T, h, w, c = attn_weight.shape
        assert c == 1

        if valid_ratios is not None:
            for i in range(len(valid_ratios)):
                valid_width = min(w, math.ceil(w * valid_ratios[i]))
                if valid_width < w:
                    attn_weight[i, :, :, valid_width:, :] = float('-inf')

        attn_weight = torch.reshape(attn_weight, [bsz, T, -1])
        attn_weight = F.softmax(attn_weight, dim=-1)

        attn_weight = torch.reshape(attn_weight, [bsz, T, h, w, c])
        attn_weight = attn_weight.permute([0,1,4,2,3])

        attn_feat = torch.sum(
            torch.multiply(feat.unsqueeze(1), attn_weight), 
            (3, 4), 
            keepdim=False)

        # Linear transformation
        if self.pred_concat:
            hf_c = holistic_feat.shape[-1]
            holistic_feat = holistic_feat.expand([bsz, seq_len, hf_c])
            y = self.prediction(torch.cat((y, attn_feat, holistic_feat), 2))
        else:
            y = self.prediction(attn_feat)

        if self.train_mode:
            y = self.pred_dropout(y)

        return y
    
    def forward_train(self, feat, out_enc, label, img_metas):
        '''
        img_metas: [label, valid_ratio]
        '''
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]

        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]

        lab_embedding = self.embedding(label)
        out_enc = out_enc.unsqueeze(1)

        in_dec = torch.cat((out_enc, lab_embedding), dim=1)
        out_dec = self._2d_attention(in_dec, feat, out_enc, valid_ratios=valid_ratios)
        return out_dec[:, 1:, :]

    def forward_test(self, feat, out_enc, img_metas):
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]

        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]

        seq_len = self.max_seq_len
        bsz = feat.shape[0]

        start_token = torch.full((bsz, ), fill_value=self.start_idx).long()
        start_token = self.embedding(start_token)
        emb_dim = start_token.shape[1]
        start_token = start_token.unsqueeze(1)
        start_token = start_token.expand([bsz, seq_len, emb_dim])
        
        out_enc = out_enc.unsqueeze(1)
        decoder_input = torch.cat((out_enc, start_token), dim=1)

        outputs = []
        for i in range(1, seq_len + 1):
            decoder_output = self._2d_attention(
                decoder_input,
                feat,
                out_enc,
                valid_ratios=valid_ratios
            )

            char_output = decoder_output[:, i, :]
            char_output = F.softmax(char_output, -1)
            outputs.append(char_output)

            max_idx = torch.argmax(char_output, dim=1, keepdim=False)
            char_embedding = self.embedding(max_idx)
            if i < seq_len:
                decoder_input[:, i+1, :] = char_embedding
                
        outputs = torch.stack(outputs, 1)
        return outputs


class SARHead(nn.Module):
    def __init__(self,
        in_channel,
        out_channel,
        enc_dim=512,
        max_text_length=30,
        enc_bi_rnn=False,
        enc_drop_rnn=0.1,
        enc_gru=False,
        dec_bi_rnn=False,
        dec_drop_rnn=0.0,
        dec_gru=False,
        d_k=512,
        pred_dropout=0.1,
        pred_concat=True,
        **kwargs
        ):
        super(SARHead, self).__init__()
        
        # encode module
        self.encoder = SAREncoder(
            enc_bi_rnn=enc_bi_rnn,
            enc_drop_rnn=enc_drop_rnn,
            enc_gru=enc_gru,
            d_model=in_channel,
            d_enc=enc_dim
        )

        # decode module
        self.decoder = ParallelSARDecoder(
            out_channel=out_channel,
            enc_bi_rnn=enc_bi_rnn,
            dec_bi_rnn=dec_bi_rnn,
            dec_drop_rnn=dec_drop_rnn,
            dec_gru=dec_gru,
            d_model=in_channel,
            d_enc=enc_dim,
            d_k=d_k,
            pred_dropout=pred_dropout,
            max_text_length=max_text_length,
            pred_concat=pred_concat
        )

    def forward(self, feat, targets=None):
        '''
        img_metas: [label, valid_ratio]
        '''
        holistic_feat = self.encoder(feat, targets)

        if self.training:
            label = targets[0]
            label = torch.tensor(label).long()
            final_out = self.decoder(
                feat, holistic_feat, label, img_metas=targets
            )
        else:
            final_out = self.decoder(
                feat, holistic_feat, label=None, img_metas=targets, train_mode=False
            )

        return final_out


if __name__ == '__main__':
    sarh = SARHead(512, 6625)
    print(sarh)
    from torchsummary import summary
    summary(sarh, input_size=(512, 6625), batch_size=1)
        