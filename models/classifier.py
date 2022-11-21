import sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from utils.layout import ch_locations_2d
import torch.nn.functional as F
from constants import device
from termcolor import cprint


class Classifier(nn.Module):
    # NOTE: experimental

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.batch_size = args.batch_size
        self.diags = torch.arange(self.batch_size).to(device)
        self.factor = self.batch_size / 241

    def forward(self, Z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = Z.view(self.batch_size, -1)
            y = Y.view(self.batch_size, -1)
            similarity = torch.matmul(x, y.T)  # NOTE: no need to normalize the dot products
            # similarity = torch.einsum('bft,bft -> bb', X, Y)  # NOTE: no need to normalize the dot products
            # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
            top1accuracy = (similarity.argmax(axis=1) == self.diags).to(torch.float).mean().item()
            top10accuracy = np.mean(
                [label in row for row, label in zip(torch.topk(similarity, 10, dim=1, largest=True)[1], self.diags)])

        # NOTE: this is potenially wrong. For top-10 accuracy we should calculate the probability
        # that the correct label is amout the top 10 highest probabilities. But since our
        # batchsize is smaller than the total number of segments, we "correct" the top-10
        # accuracy by bsz/totalNumberOfSegmentsToGuess
        return top1accuracy * self.factor, top10accuracy * self.factor
