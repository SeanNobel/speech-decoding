import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from constants import device


def torch_exp(x: torch.Tensor):  # x: ( N, )
    return torch.exp(x.clamp(max=10))


def torch_log(x: torch.Tensor):
    return torch.log(x.clamp(min=1e-10))


class MSELoss(nn.Module):
    """Takes reduction mean only for batch direction"""

    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss(reduction="none")

    def forward(self, Y, Z):  # Y, Z: both ( B, 512, 256 )
        return self.mse(Y, Z).sum(dim=(-1, -2)).mean()


class CLIPLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = device
        self.compute_similarity = nn.CosineSimilarity(dim=-1)
        self._criterion = nn.CrossEntropyLoss(reduction=args.reduction)
        # self.targets = torch.zeros(size=(batch_size, )).long() # that's for the slow method
        # self.registered_targets = False
        # self.batch_size = args.batch_size
        self.temp = nn.Parameter(torch.tensor([float(args.init_temperature)]))

    def forward(self, x, y, fast=True, return_logits=False):
        batch_size = x.size(0)
        assert batch_size > 1, "Batch size must be greater than 1."
        # if not self.registered_targets:
        #   self.register_buffer('targets', torch.arange(self.batch_size, requires_grad=False).to(self.device))
        #   self.registered_targets = True
        targets = torch.arange(batch_size, requires_grad=False).long().to(self.device)

        if not fast:
            # less efficient way
            x_ = rearrange(x, "b f t -> 1 b (f t)")
            y_ = rearrange(y, "b f t -> b 1 (f t)")
            logits = self.compute_similarity(x_, y_)  # s

            # unnecessary steps for the less efficient way (this might be needed for fancy contrastive losses)
            # positives = torch.diag(similarity_matrix, 0).view(-1,1)
            # negative_mask = torch.logical_not(torch.eye(batch_size).type(torch.bool))
            # negatives = similarity_matrix[negative_mask].view(batch_size, -1)
            # logits = torch.cat([positives, negatives], dim=1)

        else:
            # fast way
            x = x.reshape(batch_size, -1)
            y = y.reshape(batch_size, -1)

            # NOTE: scale the embeddings to unit norm
            x = x / x.norm(dim=-1, keepdim=True)
            y = y / y.norm(dim=-1, keepdim=True)

            # get dot products
            logits = torch.matmul(x, y.T)

            # scale by temperature (learned)
            logits *= torch.exp(self.temp)

            # # NOTE: the old way
            # # I don't know why yet, but normalization seems to be necessary to get sensible similarities (0, 1)
            # logits = logits / (x.norm(dim=-1) * y.norm(dim=-1))
            # loss = self._criterion(logits, targets)

        # NOTE: as in https://arxiv.org/abs/2103.00020
        loss = (self._criterion(logits, targets) + self._criterion(logits.t(), targets)) / 2

        if return_logits:
            return logits, loss
        else:
            return loss


class MyCLIPLikeClassificationLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = device
        self.compute_similarity = nn.CosineSimilarity(dim=-1)
        self._criterion = nn.CrossEntropyLoss(reduction=args.reduction)
        # self.targets = torch.zeros(size=(batch_size, )).long() # that's for the slow method
        # self.registered_targets = False
        # self.batch_size = args.batch_size
        self.temp = nn.Parameter(torch.tensor([float(args.init_temperature)]))

        self.prepare_image_features()
        self.same_category_length = 8

    def prepare_image_features(self):
        sorted_image_features = np.load('./data/GOD/image_features_train.npy')
        self.sorted_image_features = torch.tensor(sorted_image_features, requires_grad=False).to(torch.float).to(device)
        assert len(self.sorted_image_features) == 1200
        self.gt_size = 1200 

    def calculate_smooth_labeling(self, labels:np.ndarray, smmoth_value=0.1):

        # labels: batchsize:64
        targets = torch.zeros([64, 1200],requires_grad=False).to(torch.float).to(device)
        for i, l in enumerate(labels):
            l_mod = l % self.same_category_length
            targets[i, l_mod*self.same_category_length:(l_mod+1)*self.same_category_length] = smmoth_value
            targets[i, l] = 1
        return targets


    def forward(self, x, labels, fast=True, return_logits=False):
        labels = labels-1 # labelsは1始まり
        batch_size = x.size(0)
        y = self.sorted_image_features
        assert batch_size > 1, "Batch size must be greater than 1."
        # if not self.registered_targets:
        #   self.register_buffer('targets', torch.arange(self.batch_size, requires_grad=False).to(self.device))
        #   self.registered_targets = True
        targets = self.calculate_smooth_labeling(labels, smmoth_value=0.1)# torch.arange(batch_size, requires_grad=False).long().to(self.device)

        if not fast:
            # less efficient way
            x_ = rearrange(x, "b f t -> 1 b (f t)")
            y_ = rearrange(y, "b f t -> b 1 (f t)")
            logits = self.compute_similarity(x_, y_)  # s

            # unnecessary steps for the less efficient way (this might be needed for fancy contrastive losses)
            # positives = torch.diag(similarity_matrix, 0).view(-1,1)
            # negative_mask = torch.logical_not(torch.eye(batch_size).type(torch.bool))
            # negatives = similarity_matrix[negative_mask].view(batch_size, -1)
            # logits = torch.cat([positives, negatives], dim=1)

        else:
            # import pdb; pdb.set_trace()
            # fast way
            x = x.reshape(batch_size, -1)
            y = y.reshape(self.gt_size, -1)

            # NOTE: scale the embeddings to unit norm
            x = x / x.norm(dim=-1, keepdim=True)
            y = y / y.norm(dim=-1, keepdim=True)

            # get dot products
            logits = torch.matmul(x, y.T)

            # scale by temperature (learned)
            logits *= torch.exp(self.temp)

            # # NOTE: the old way
            # # I don't know why yet, but normalization seems to be necessary to get sensible similarities (0, 1)
            # logits = logits / (x.norm(dim=-1) * y.norm(dim=-1))
            # loss = self._criterion(logits, targets)

        # NOTE: as in https://arxiv.org/abs/2103.00020
        loss = self._criterion(logits, targets) # + self._criterion(logits.t(), targets.T)) / 2

        if return_logits:
            return logits, loss
        else:
            return loss