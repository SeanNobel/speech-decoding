import sys
import numpy as np
import torch
import torch.nn as nn
from time import time
import torch.nn.functional as F
from termcolor import cprint
from einops import rearrange

from speech_decoding.utils.layout import ch_locations_2d
from speech_decoding.constants import device
    

# class BrainMultiAttention(nn.Module):

#     def __init__(self, args, embed_dim=None, num_heads=1):
#         super(BrainMultiAttention, self).__init__()

#         self.num_heads = num_heads
#         self.embed_dim = embed_dim 
#         self.eeg_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=False)

#     def forward(self, X, return_attn=False): # X(B, F, T)=(64, 128, 90)
#         cprint(f"X.shape: {X.shape}", "red")  # ( B, F, T
#         X = torch.permute(X, (2, 0, 1))  # (T, B, F)
#         X, attn = self.eeg_attention(X, X, X)  # was expecting embedding dimension of 128, but got 270
#         X = torch.permute(X, (1, 2, 0))  # (B, F, T)
#         return X, attn
        

class SpatialAttention(nn.Module):
    """Same as SpatialAttentionVer2, but a little more concise"""

    def __init__(self, args, layout_fn):
        super(SpatialAttention, self).__init__()

        # vectorize of k's and l's
        a = []
        for k in range(args.K):
            for l in range(args.K):
                a.append((k, l))
        a = torch.tensor(a).to(device)  # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
        k, l = a[:, 0], a[:, 1]

        # vectorize x- and y-positions of the sensors
        loc = layout_fn(args).to(device)
        x, y = loc[:, 0], loc[:, 1]

        # make a complex-valued parameter, reshape k,l into one dimension
        self.z = nn.Parameter(
            torch.rand(size=(args.D1, args.K**2), dtype=torch.cfloat)
        ).to(device)

        # NOTE: pre-compute the values of cos and sin (they depend on k, l, x and y which repeat)
        phi = (
            2
            * torch.pi
            * (torch.einsum("k,x->kx", k, x) + torch.einsum("l,y->ly", l, y))
        )  # torch.Size([1024, 60]))
        self.cos = torch.cos(phi).to(device)
        self.sin = torch.sin(phi).to(device)

        # self.spatial_dropout = SpatialDropoutX(args)
        self.spatial_dropout = SpatialDropout(loc, args.d_drop)

    def forward(self, X):

        # NOTE: do hadamard product and and sum over l and m (i.e. m, which is l X m)
        re = torch.einsum(
            "jm, me -> je", self.z.real, self.cos
        )  # torch.Size([270, 60])
        im = torch.einsum("jm, me -> je", self.z.imag, self.sin)
        a = (
            re + im
        )  # essentially (unnormalized) weights with which to mix input channels into ouput channels
        # ( D1, num_channels )

        # NOTE: to get the softmax spatial attention weights over input electrodes,
        # we don't compute exp, etc (as in the eq. 5), we take softmax instead:
        SA_wts = F.softmax(a, dim=-1)  # each row sums to 1
        # ( D1, num_channels )

        # NOTE: drop some channels within a d_drop of the sampled channel
        dropped_X = self.spatial_dropout(X)

        # NOTE: each output is a diff weighted sum over each input channel
        return torch.einsum("oi,bit->bot", SA_wts, dropped_X)


class SpatialDropout(nn.Module):
    """Using same drop center for all samples in batch"""

    def __init__(self, loc, d_drop):
        super(SpatialDropout, self).__init__()
        self.loc = loc  # ( num_channels, 2 )
        self.d_drop = d_drop
        self.num_channels = loc.shape[0]

    def forward(self, X):  # ( B, num_channels, seq_len )
        assert X.shape[1] == self.num_channels

        if self.training:
            drop_center = self.loc[np.random.randint(self.num_channels)]  # ( 2, )
            distances = (self.loc - drop_center).norm(dim=-1)  # ( num_channels, )
            mask = torch.where(distances < self.d_drop, 0.0, 1.0).to(
                device
            )  # ( num_channels, )
            return torch.einsum("c,bct->bct", mask, X)
        else:
            return X


class SubjectBlock(nn.Module):
    def __init__(self, args, num_subjects, layout_fn):
        super(SubjectBlock, self).__init__()

        self.num_subjects = num_subjects
        self.D1 = args.D1
        self.K = args.K
        self.spatial_attention = SpatialAttention(args, layout_fn)
        self.conv = nn.Conv1d(
            in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1
        )
        self.use_subject_layer = args.use_subject_layer
        if self.use_subject_layer:
            self.subject_layer = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=self.D1,
                        out_channels=self.D1,
                        kernel_size=1,
                        bias=False,
                        stride=1,
                        device=device,
                    )
                    for _ in range(self.num_subjects)
                ]
            )


        # self.use_attn_instead_of_sbjly = args.use_attn_instead_of_sbjly

        # self.brain_multi_attention = BrainMultiAttention(args, embed_dim=self.D1, num_heads=1)

    def forward(self, X, subject_idxs):
        # subject_idxs: ( B, ),  i = 0 or 1 or 2 in subject_random_split
        X = self.spatial_attention(X)  # ( B, 270, 256 )
        if self.use_subject_layer:
            X = self.conv(X)  # ( B, 270, 256 )
            X = torch.cat(
                [self.subject_layer[i](x.unsqueeze(dim=0)) for i, x in zip(subject_idxs, X)]
            )  # ( B, 270, 256 )
        # elif self.use_attn:
        #     X, attn = self.brain_multi_attention(X)
        #     X = self.conv(X)  # ( B, 270, 256 )
        return X


# class SubjectBlock_proto(nn.Module):
#     def __init__(self, args):
#         super(SubjectBlock_proto, self).__init__()

#         self.num_subjects = args.num_subjects
#         self.D1 = args.D1
#         self.K = args.K
#         self.spatial_attention = SpatialAttention(args)
#         self.conv = nn.Conv1d(
#             in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1
#         )

#         # NOTE: The below implementations are equivalent to learning a matrix:
#         self.subject_matrix = nn.Parameter(
#             torch.rand(self.num_subjects, self.D1, self.D1)
#         )
#         # self.subject_layer = [
#         #     nn.Conv1d(in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1, device=device)
#         #     for _ in range(self.num_subjects)
#         # ]

#     def forward(self, X, subject_idxs):
#         X = self.spatial_attention(X)  # ( B, 270, 256 )
#         X = self.conv(X)  # ( B, 270, 256 )

#         # NOTE to Sensho: this has caused problems. I slighly changed it here. Hope it doesn't break anything for you
#         _subject_idxs = subject_idxs.tolist()
#         X = (
#             self.subject_matrix[_subject_idxs] @ X
#         )  # ( 270, 270 ) @ ( B , 270, 256 ) -> ( B, 270, 256 )
#         # _X = []
#         # for i, x in enumerate(X):  # x: ( 270, 256 )
#         #     x = self.subject_layer[subject_idxs[i]](x.unsqueeze(0))  # ( 1, 270, 256 )
#         #     _X.append(x.squeeze())
#         # X = torch.stack(_X)

#         return X  # ( B, 270, 256 )


class ConvBlock(nn.Module):
    def __init__(self, k, D1, D2):
        super(ConvBlock, self).__init__()

        self.k = k
        self.D2 = D2
        self.in_channels = D1 if k == 0 else D2

        self.conv0 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.D2,
            kernel_size=3,
            padding="same",
            dilation=2 ** ((2 * k) % 5),
        )
        self.batchnorm0 = nn.BatchNorm1d(num_features=self.D2)
        self.conv1 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=self.D2,
            kernel_size=3,
            padding="same",
            dilation=2 ** ((2 * k + 1) % 5),
        )
        self.batchnorm1 = nn.BatchNorm1d(num_features=self.D2)
        self.conv2 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=2 * self.D2,
            kernel_size=3,
            padding="same",
            dilation=2,  # FIXME: The text doesn't say this, but the picture shows dilation=2
        )

    def forward(self, X):
        if self.k == 0:
            X = self.conv0(X)
        else:
            X = self.conv0(X) + X  # skip connection

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X)
        X = F.glu(X, dim=-2)

        return X  # ( B, 320, 256 )


class BrainEncoder(nn.Module):
    def __init__(self, args, num_subjects=None, layout_fn=None):
        super(BrainEncoder, self).__init__()

        self.num_subjects = args.num_subjects if num_subjects is None else num_subjects
        self.D1 = args.D1
        self.D2 = args.D2
        self.F = args.F  # if not args.preprocs["last4layers"] else 1024 -> 128
        self.K = args.K  # 32
        self.dataset_name = args.dataset
        self.use_fc = args.use_fc
        # self.use_attn = args.use_attn

        if layout_fn is None:
            layout_fn = ch_locations_2d
        self.subject_block = SubjectBlock(args, self.num_subjects, layout_fn)
        # cprint(
        #     "USING THE OLD IMPLEMENTATION OF THE SUBJECT BLOCK",
        #     "red",
        #     "on_blue",
        #     attrs=["bold"],
        # )

        self.conv_blocks = nn.Sequential()
        for k in range(5):
            self.conv_blocks.add_module(f"conv{k}", ConvBlock(k, self.D1, self.D2))

        self.conv_final1 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=2 * self.D2,
            kernel_size=args.final_kernel_size,
            stride=args.final_stride,
        )

        self.conv_final2 = nn.Conv1d(
            in_channels=2 * self.D2,
            out_channels=self.F,
            kernel_size=args.final_kernel_size,
            stride=args.final_stride,
        )
        if self.use_fc:
            self.T = args.seq_len * args.fps  # 90
            self.fc1 = nn.Linear(in_features=self.T, out_features=self.T * 2)
            self.fc2 = nn.Linear(in_features=self.T * 2, out_features=self.T * 4)
            self.fc3 = nn.Linear(in_features=self.T * 4, out_features=self.T)
        
        # self.brain_multi_attention = BrainMultiAttention(args, embed_dim=self.F, num_heads=1)

    def forward(self, X, subject_idxs):
        X = self.subject_block(X, subject_idxs)
        X = self.conv_blocks(X)
        X = F.gelu(self.conv_final1(X))
        X = F.gelu(self.conv_final2(X))  # # X_f.shape: torch.Size([64, 128, 90])
        if self.use_fc:
            X = self.fc1(X)
            X = self.fc2(X)
            X = self.fc3(X)
        #     # cprint(f"X.shape: {X.shape}", "yellow")  # https://discuss.pytorch.org/t/how-to-pass-a-3d-tensor-to-linear-layer/908
        # elif self.use_fft_train_with_attn:
        #     X, attn = self.brain_multi_attention(X)
        return X


class Classifier(nn.Module):
    # NOTE: experimental

    def __init__(self, args, training=True):
        super(Classifier, self).__init__()

        # NOTE: Do we need to adjust the accuracies for the dataset size?
        self.factor = 1  # self.batch_size / 241

        self.training = training

    @torch.no_grad()
    def forward(self, Z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        batch_size_z = Z.size(0)
        batch_size_y = Y.size(0)
        diags = torch.arange(batch_size_y).to(device)
        x = Z.reshape(batch_size_z, -1)
        y = Y.reshape(batch_size_y, -1)

        # x_ = rearrange(x, 'b f -> 1 b f')
        # y_ = rearrange(y, 'b f -> b 1 f')
        # similarity = torch.nn.functional.cosine_similarity(x_, y_, dim=-1)  # ( B, B )

        # NOTE: avoid CUDA out of memory like this
        similarity = torch.empty(batch_size_z, batch_size_y).to(device)
        for i in range(batch_size_z):
            for j in range(batch_size_y):
                similarity[i, j] = (x[i] @ y[j]) / max(
                    (x[i].norm() * y[j].norm()), 1e-8
                )

        if self.training:

            similarity = similarity.T

            # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
            top1accuracy = (
                (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()
            )
            try:
                top10accuracy = np.mean(
                    [
                        label in row
                        for row, label in zip(
                            torch.topk(similarity, 10, dim=1, largest=True)[1], diags
                        )
                    ]
                )
            except:
                print(similarity.size())
                raise

            return top1accuracy, top10accuracy

        else:
            return similarity
