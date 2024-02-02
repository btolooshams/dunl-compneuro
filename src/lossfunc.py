"""
Copyright (c) 2020 Bahareh Tolooshams

loss functions for training

:author: Bahareh Tolooshams
"""

import torch


class DUNL1DLoss(torch.nn.Module):
    def __init__(self, model_distribution):
        super(DUNL1DLoss, self).__init__()

        self.model_distribution = model_distribution

    def forward(self, y, Hxa):
        if self.model_distribution == "gaussian":
            loss = torch.nn.functional.mse_loss(y, Hxa, reduction="none")
        elif self.model_distribution == "binomial":
            loss = -torch.mean(y * Hxa, dim=-1) + torch.mean(
                torch.log1p(torch.exp(Hxa)), dim=-1
            )
        elif self.model_distribution == "poisson":
            loss = -torch.mean(y * Hxa, dim=-1) + torch.mean(torch.exp(Hxa), dim=-1)

        return torch.mean(loss)


class Smoothloss(torch.nn.Module):
    def __init__(self):
        super(Smoothloss, self).__init__()

    def forward(self, H):
        loss = (H[:, :, 1:] - H[:, :, :-1]).pow(2).sum() / H.shape[0]
        return loss


class l1loss(torch.nn.Module):
    def __init__(self):
        super(l1loss, self).__init__()

    def forward(self, x):
        loss = torch.mean(torch.abs(x))
        return loss
