import sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from utils.layout import ch_locations_2d
import torch.nn.functional as F
from constants import device
from termcolor import cprint
from einops import rearrange


class Classifier(nn.Module):
    # NOTE: experimental

    def __init__(self, args):
        super(Classifier, self).__init__()

        # NOTE: Do we need to adjust the accuracies for the dataset size?
        self.factor = 1  # self.batch_size / 241

    @torch.no_grad()
    def forward(self, Z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        batch_size = Z.size(0)
        diags = torch.arange(batch_size).to(device)
        x = Z.view(batch_size, -1)
        y = Y.view(batch_size, -1)

        # x_ = rearrange(x, 'b f -> 1 b f')
        # y_ = rearrange(y, 'b f -> b 1 f')
        # similarity = torch.nn.functional.cosine_similarity(x_, y_, dim=-1)  # ( B, B )

        # NOTE: avoid CUDA out of memory like this
        similarity = torch.empty(batch_size, batch_size).to(device)
        for i in range(batch_size):
            for j in range(batch_size):
                similarity[i, j] = x[i] @ y[j] / (x[i].norm() * y[j].norm())

        similarity = similarity.T

        # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
        top1accuracy = (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()
        try:
            top10accuracy = np.mean(
                [label in row for row, label in zip(torch.topk(similarity, 10, dim=1, largest=True)[1], diags)])
        except:
            print(similarity.size())
            raise

        return top1accuracy, top10accuracy
