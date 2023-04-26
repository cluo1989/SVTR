'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-23 14:37:20
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-24 13:45:20
FilePath: /SVTR/modeling/loss/rec_sar_loss.py
Description: 
'''
from torch import nn

class SARLoss(nn.Module):
    def __init__(self):
        super(SARLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, inputs):
        logits = inputs[0][:-1, :, :]  # ignore last index of outputs to be in same seq_len with targets
        labels = inputs[1][:, 1:]      # ignore first index of target in loss calculation

        num_classes = logits.shape[2] # N,T,C
        logits = logits.reshape([-1, num_classes])
        labels = labels.reshape([-1])
        loss = self.loss_func(logits, labels)

        return loss


if __name__ == '__main__':
    loss_func = SARLoss()
    import torch
    logits = torch.randn(31, 16, 20)  # (T, N, C)
    labels = torch.randint(1, 20, (16, 31), dtype=torch.long)  # (N, S)
    intputs = [logits, labels]
    print(loss_func(intputs))
    