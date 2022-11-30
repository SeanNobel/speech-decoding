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

        # NOTE: no need to normalize the dot products
        # similarity = torch.matmul(x, y.T)

        x_ = rearrange(x, 'b f -> 1 b f')
        y_ = rearrange(y, 'b f -> b 1 f')
        similarity = torch.nn.functional.cosine_similarity(x_, y_, dim=-1)  # ( B, B )

        # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
        top1accuracy = (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()
        _top1accuracy = (similarity.argmax(axis=0) == diags).to(torch.float).mean().item()
        try:
            top10accuracy = np.mean(
                [label in row for row, label in zip(torch.topk(similarity, 10, dim=1, largest=True)[1], diags)])
            _top10accuracy = np.mean(
                [label in row for row, label in zip(torch.topk(similarity, 10, dim=0, largest=True)[1], diags)])
        except:
            print(similarity.size())
            raise

        return top1accuracy, top10accuracy, _top1accuracy, _top10accuracy
