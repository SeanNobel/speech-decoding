import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
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

        probs = torch.einsum('bft,nft->bn', Z, Y) # ( N, N )
        
        # NOTE avoid nan by this
        probs = (probs.T - probs.max(dim=1).values).T

        probs = torch.exp(probs)
        probs = (probs.T / probs.sum(dim=1)).T
        
        labels = torch.eye(N).cuda()

        # return F.binary_cross_entropy(input=probs, target=labels, reduction=self.reduction)
        return F.cross_entropy(input=probs, target=labels, reduction=self.reduction)


class CLIPLoss2(nn.Module):
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

        logits = torch.einsum('bft,nft->bn', Z, Y) # ( N, N )
        
        labels = torch.eye(N).cuda()

        loss_i = F.binary_cross_entropy(input=logits, target=labels, reduction=self.reduction)
        loss_t = F.binary_cross_entropy(input=logits.T, target=labels, reduction=self.reduction)

        return loss_i + loss_t


# def clip_loss_(Y: torch.Tensor, Z: torch.Tensor):
#     """
#     Y ( N, F, T ): latent representation of speech sound from wav2vec
#     Z ( N, F, T ): latent representation of EEG/MEG from brain encoder
#     """
#     assert Y.shape == Z.shape
#     N = Y.shape[0]

#     loss = 0
#     for j in range(N):
#         # Inner product over both dimensions. Same as torch.sum(Z[j] * Y[j])
#         probs = torch.einsum('ft,nft->n', Z[j], Y) # ( N, )

#         # NOTE avoid nan by this
#         probs -= probs.max()

#         probs = torch.exp(probs)
#         probs /= probs.sum()
        
#         labels = torch.zeros_like(probs)
#         labels[j] = 1

#         loss += nn.CrossEntropyLoss()(input=probs, target=labels)

    
#     return loss


if __name__ == "__main__":

    Y = torch.rand(128, 20, 100)
    Z = torch.rand(128, 20, 100)

    # loss = clip_loss_(Y, Z)
    # print(loss)

    clip_loss = CLIPLoss2()
    loss = clip_loss(Y, Z)
    print(loss)