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

class SameLabelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, Z, labels):
        """
        Z: ( B, 512 ) torch.tensor
        labels: ( B, ) np.ndarray
        """
        loss = []
        for i,l in enumerate(labels):
            indices = np.where(labels==l)[0]
            anchor_Z = Z[l,:]
            for ind in indices:
                if ind == i:
                    continue
                else:
                    ref_Z = Z[ind,:]
                    loss.append(self.mse(anchor_Z, ref_Z))

        return torch.mean(torch.stack(loss))




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
        if args.criterion == 'crossentropy':
            print('use crossentropy')
            self._criterion = nn.CrossEntropyLoss(reduction=args.reduction)
            self.criterion_mode = 'crossentropy'
            self.smmoth_value=0.1
        elif args.criterion == 'binary_crossentropy':
            print('use binary_cross_entropy')
            self._criterion = nn.BCELoss(reduction=args.reduction)
            self.criterion_mode = 'binary_crossentropy'
            self.smmoth_value=0.5
        elif args.criterion == 'similarity_crossentropy':
            print('use similarity_crossentropy')
            self._criterion = nn.CrossEntropyLoss(reduction=args.reduction)
            self.criterion_mode = 'similarity_crossentropy'
            self.smmoth_value = None
        else:
            raise ValueError()
        # self.targets = torch.zeros(size=(batch_size, )).long() # that's for the slow method
        # self.registered_targets = False
        # self.batch_size = args.batch_size
        if args.temp_trainable:
            self.temp = nn.Parameter(torch.tensor([float(args.init_temperature)]))
        else:
            self.temp = torch.tensor(float(args.init_temperature), requires_grad=False)
        self.normalize_image_features = args.normalize_image_features
        self.prepare_image_features()
        self.same_category_length = 8


    def prepare_image_features(self):
        sorted_image_features = np.load('./data/GOD/image_features_train.npy')
        self.sorted_image_features = torch.tensor(sorted_image_features, requires_grad=False).to(torch.float).to(device)
        assert len(self.sorted_image_features) == 1200
        self.gt_size = 1200

        sorted_image_features_test = np.load('./data/GOD/image_features.npy')
        self.sorted_image_features_test = torch.tensor(sorted_image_features_test, requires_grad=False).to(torch.float).to(device)
        assert len(self.sorted_image_features_test) == 50
        self.gt_size_test = 50

        if self.normalize_image_features:
            self.sorted_image_features = self.normalize_per_unit(self.sorted_image_features)
            self.sorted_image_features_test = self.normalize_per_unit(self.sorted_image_features_test)

        if self.criterion_mode == 'similarity_crossentropy':
            self.similarity_matrix = self.compute_similarity(self.sorted_image_features, self.sorted_image_features)
            self.similarity_matrix_test = self.compute_similarity(self.sorted_image_features_test, self.sorted_image_features_test)

    def normalize_per_unit(self, tensor):
        print('normalize image_feature along unit dim')
        # array: n_samples x n_units(512)
        tensor = tensor - torch.mean(tensor, 0, keepdim=True)
        tensor = tensor / torch.std(tensor, 0,  keepdim=True)
        return tensor

    def calculate_smooth_labeling(self, labels:np.ndarray, smmoth_value=0.1):
        # labels: batchsize:64
        targets = torch.zeros([64, 1200],requires_grad=False).to(torch.float).to(device)

        if self.criterion_mode == 'crossentropy' and self.criterion_mode=='binary_crossentropy':
            for i, l in enumerate(labels):
                l_mod = l % self.same_category_length
                targets[i, l_mod*self.same_category_length:(l_mod+1)*self.same_category_length] = smmoth_value
                targets[i, l] = 1
        elif self.criterion_mode == 'similarity_crossentropy':
            for i, l in enumerate(labels):
                targets[i] = self.similarity_matrix[l]
        return targets


    def forward(self, x, labels, fast=True, return_logits=False, train=True, debug_Y=None):
        labels = labels-1 # labelsは1始まり
        batch_size = x.size(0)

        assert batch_size > 1, "Batch size must be greater than 1."

        if train:
            # smooth_value 0.1 -> 0.5 (binaryentripy使用時)
            targets = self.calculate_smooth_labeling(labels, smmoth_value=self.smmoth_value)# torch.arange(batch_size, requires_grad=False).long().to(self.device)
            y = self.sorted_image_features
            gt_size = self.gt_size
            # print(debug_Y[0]==y[labels[0]])
            # import pdb; pdb.set_trace()
        else:
            targets = labels.to(torch.long) # torch.arange(batch_size, requires_grad=False).long().to(self.device)
            if self.criterion_mode == 'binary_crossentropy':
                targets = F.one_hot(targets, num_classes=50).to(torch.float) # binary cross entropy only can manipulate onehot
            y = self.sorted_image_features_test
            gt_size = self.gt_size_test
            # import pdb; pdb.set_trace()
        if not fast:
            # less efficient way
            x_ = rearrange(x, "b f t -> 1 b (f t)")
            y_ = rearrange(y, "b f t -> b 1 (f t)")
            logits = self.compute_similarity(x_, y_)  # s

        else:
            # fast way
            x = x.reshape(batch_size, -1)
            y = y.reshape(gt_size, -1)

            # NOTE: scale the embeddings to unit norm
            x = x / x.norm(dim=-1, keepdim=True)
            y = y / y.norm(dim=-1, keepdim=True)

            # get dot products
            logits = torch.matmul(x, y.T)

            # scale by temperature (learned)
            logits = logits * torch.exp(self.temp)


        # NOTE: as in https://arxiv.org/abs/2103.00020
        if self.criterion_mode == 'binary_crossentropy':
            # import pdb; pdb.set_trace()
            logits = torch.sigmoid(logits)  # torch.exp(logits) / torch.exp(logits).sum(dim=-1, keepdim=True)
        if self.criterion_mode == 'similarity_crossentropy':
            if train:
                positive_targets = torch.nn.Softmax(1)(targets *torch.exp(self.temp))
                # negative_targets =  torch.nn.Softmax(1)(-targets)
                loss = self._criterion(logits, positive_targets)
            else:
                loss = self._criterion(logits, targets)
        else:
            loss = self._criterion(logits, targets) # + self._criterion(logits.t(), targets.T)) / 2

        if return_logits:
            return logits, loss
        else:
            return loss