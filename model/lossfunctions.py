import torch
import torch.nn as nn


class L1DistanceLoss(nn.Module):
    def __init__(self):
        super(L1DistanceLoss, self).__init__()
        self.word_pair_dims = (1, 2)

    def forward(self, predictions, label_batch, length_batch):
        # 去掉padding部分
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()
        if total_sents > 0:
            loss_per_sent = torch.sum(
                torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims
            )
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device="cuda")
        return batch_loss, total_sents


class L1DepthLoss(nn.Module):
    def __init__(self):
        super(L1DepthLoss, self).__init__()
        self.word_dim = 1

    def forward(self, predictions, label_batch, length_batch):
        total_sents = torch.sum(length_batch != 0).float()
        # 去掉padding部分
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s

        if total_sents > 0:
            loss_per_sent = torch.sum(
                torch.abs(predictions_masked - labels_masked), dim=self.word_dim
            )
            normalized_loss_per_sent = loss_per_sent / length_batch.float()
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device="cuda")
        return batch_loss, total_sents


class ContrasiveLoss(nn.Module):
    # todo: 实现对比损失
    def __init__(self) -> None:
        super(ContrasiveLoss, self).__init__()

    def forward(self, predictions, label_batch, length_batch):

        pass
