'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-23 14:24:00
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-23 17:47:52
FilePath: /SVTR/modeling/loss/rec_ctc_loss.py
Description: 
'''
import torch
from torch import nn


class CTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        # nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    def forward(self, inputs):
        logits = inputs[0]
        labels = inputs[1]
        label_lengths = inputs[2]

        # logits -> probs
        pred_probs = torch.log_softmax(logits, dim=2)
        
        # input lengths
        batch_size = logits.shape[1]
        time_steps = logits.shape[0]
        # pred_lengths = torch.LongTensor([time_steps] * batch_size)
        pred_lengths = torch.full((batch_size,), time_steps)
        loss = self.loss_func(pred_probs, labels, pred_lengths, label_lengths)

        # focal loss
        if self.use_focal_loss:
            weight = torch.square(1 - torch.exp(-loss))
            loss = torch.multiply(loss, weight)

        loss = torch.mean(loss)
        return loss


if __name__ == '__main__':
    loss_func = CTCLoss()
    # ref: 如何优雅的使用pytorch内置torch.nn.CTCLoss的方法
    # https://zhuanlan.zhihu.com/p/67415439
    # https://github.com/GitYCC/crnn-pytorch/blob/f145a61e20e57a36b8f2d642f93c9ba2890cd042/src/train.py#L15
    logits = torch.randn(50, 16, 20)  # (T, N, C)
    labels = torch.randint(1, 20, (16, 30), dtype=torch.long)        # (N, S) S=len(label)
    # input_lengths = torch.full((16,), 50, dtype=torch.long)        # (N, ) value=T
    label_lengths = torch.randint(10, 30, (16, ), dtype=torch.long)  # (N) value=len(label)
    inputs = [logits, labels, label_lengths]
    loss = loss_func(inputs)
    print(loss)