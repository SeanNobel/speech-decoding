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

    def forward(self, Z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        x = Z.view(self.batch_size, -1)
        y = Y.view(self.batch_size, -1)
        similarity = torch.matmul(x, y.T)  # NOTE: no need to normalize the dot products
        # similarity = torch.einsum('bft,bft -> bb', X, Y)  # NOTE: no need to normalize the dot products
        # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
        accuracy = (similarity.argmax(axis=1) == self.diags).to(torch.float).mean()
        return accuracy