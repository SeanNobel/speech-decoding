import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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

    def __init__(self, device, batch_size, reduction="mean"):
        super().__init__()
        self.device = device
        self.compute_similarity = nn.CosineSimilarity(dim=-1)
        self._criterion = nn.CrossEntropyLoss(reduction=reduction)
        # self.targets = torch.zeros(size=(batch_size, )).long() # that's for the slow method
        self.registered_targets = False
        self.batch_size = batch_size

    def forward(self, x, y, fast=True, return_logits=False):
        # batch_size = x.size(0)
        if not self.registered_targets:
            self.register_buffer(
                'targets',
                torch.arange(self.batch_size,
                             requires_grad=False).to(self.device))
            self.registered_targets = True

        if not fast:
            # less efficient way
            x_ = rearrange(x, 'b f t -> 1 b (f t)')
            y_ = rearrange(y, 'b f t -> b 1 (f t)')
            logits = self.compute_similarity(x_, y_)  # s

            # unnecessary steps for the less efficient way (this might be needed for fancy contrastive losses)
            # positives = torch.diag(similarity_matrix, 0).view(-1,1)
            # negative_mask = torch.logical_not(torch.eye(batch_size).type(torch.bool))
            # negatives = similarity_matrix[negative_mask].view(batch_size, -1)
            # logits = torch.cat([positives, negatives], dim=1)

        else:
            # fast way
            x = x.reshape(self.batch_size, -1)
            y = y.reshape(self.batch_size, -1)
            logits = torch.matmul(x, y.T)
            # I don't know why yet, but normalization seems to be necessary to get sensible similarities (0, 1)
            logits = logits / (x.norm(dim=-1) * y.norm(dim=-1))

        # print(f"is_cuda self.targets {self.targets.is_cuda} logits {logits.is_cuda}")
        if return_logits:
            return logits, self._criterion(logits, self.targets)
        else:
            return self._criterion(logits, self.targets)


class CLIPLossVer3(nn.Module):

    def __init__(self, reduction="sum"):
        super().__init__()

        self.reduction = reduction

    def forward(self, Y: torch.Tensor, Z: torch.Tensor):
        """
        Y ( N, F, T ): latent representation of speech sound from wav2vec
        Z ( N, F, T ): latent representation of EEG/MEG from brain encoder
        """
        assert Y.shape == Z.shape
        N = Y.shape[0]

        probs = torch.einsum('bft,nft->bn', Z, Y)  # ( N, N )

        probs = nn.LogSoftmax(dim=1)(probs)

        targets = torch.eye(N).to(device)

        loss = -(targets * probs).sum(dim=1)

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            raise NotImplementedError()


class CLIPLossVer1(nn.Module):

    def __init__(self, reduction="sum"):
        super().__init__()

        self.reduction = reduction

    def forward(self, Y: torch.Tensor, Z: torch.Tensor):
        """
        Y ( N, F, T ): latent representation of speech sound from wav2vec
        Z ( N, F, T ): latent representation of EEG/MEG from brain encoder
        """
        assert Y.shape == Z.shape
        N = Y.shape[0]

        probs = torch.einsum('bft,nft->bn', Z, Y)  # ( N, N )

        # NOTE avoid nan by this
        probs = (probs.T - probs.max(dim=1).values).T

        probs = torch.exp(probs)
        probs = (probs.T / probs.sum(dim=1)).T

        labels = torch.eye(N).to(device)

        # return F.binary_cross_entropy(input=probs, target=labels, reduction=self.reduction)
        return F.cross_entropy(input=probs,
                               target=labels,
                               reduction=self.reduction)


class CLIPLossVer2(nn.Module):

    def __init__(self, reduction="sum"):
        super().__init__()

        self.reduction = reduction

    def forward(self, Y: torch.Tensor, Z: torch.Tensor):
        """
        Y ( N, F, T ): latent representation of speech sound from wav2vec
        Z ( N, F, T ): latent representation of EEG/MEG from brain encoder
        """
        assert Y.shape == Z.shape
        N = Y.shape[0]

        term_1 = -torch.einsum('nft,nft->n', Z, Y)  # ( N, )
        # print(term_1)

        term_2 = []
        for i in range(N):
            _term_2 = torch.einsum('ft,nft->n', Z[i], Y)  # ( N, )
            # _term_2 -= _term_2.max()
            # _term_2 = _term_2 - _term_2.mean()
            _term_2 = torch_exp(_term_2)
            _term_2 = _term_2.sum()

            term_2.append(torch_log(_term_2))

        term_2 = torch.stack(term_2)
        # print(term_2)

        # sys.exit()

        # return (term_1 + term_2).mean()
        return term_1.mean()


if __name__ == "__main__":

    Y = torch.rand(128, 20, 100).cuda()
    Z = torch.rand(128, 20, 100).cuda()

    # loss = clip_loss_(Y, Z)
    # print(loss)

    clip_loss = CLIPLoss()
    loss = clip_loss(Y, Z)
    print(loss)