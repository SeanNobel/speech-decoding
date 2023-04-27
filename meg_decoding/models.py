import sys
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
from time import time
import torch.nn.functional as F
from termcolor import cprint
from einops import rearrange
from tqdm import tqdm

from constants import device
from meg_decoding.utils.layout import ch_locations_2d



def get_model(args):
    if args.model == 'brain_encoder':
        return BrainEncoder(args)
    elif args.model == 'brain_endcoder_seq2static':
        return  BrainEncoderSeq2Static(args)
    elif args.model == 'linear':
        return LinearEncoder(args)
    elif args.model == 'eegnet':
        return EEGNet(args)
    else:
        raise ValueError('no model named {} is prepared'.format(args.model))

class EEGNet(nn.Module):
    def __init__(self, args):
        super(EEGNet, self).__init__()
        T = int((args.window.end - args.window.start) * args.preprocs.brain_resample_rate)
        # DownSampling
        # self.down1 = nn.Sequential(nn.AvgPool2d((1, p0)))

        # Conv2d(in,out,kernel,stride,padding,bias) #k1 30
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, args.F1, (1, args.k1), padding="same", bias=False), nn.BatchNorm2d(args.F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                args.F1, args.D * args.F1, (args.num_channels, 1), groups=args.F1, bias=False
            ),
            nn.BatchNorm2d(args.D * args.F1),
            nn.ELU(),
            nn.AvgPool2d((1, args.p1)), # 2
            nn.Dropout(args.dr1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                args.D * args.F1,
                args.D * args.F1,
                (1, args.k2), # 4
                padding="same",
                groups=args.D * args.F1,
                bias=False,
            ),
            nn.Conv2d(args.D * args.F1, args.F2, (1, 1), bias=False),
            nn.BatchNorm2d(args.F2),
            nn.ELU(),
            nn.AvgPool2d((1, args.p2)), # 4
            nn.Dropout(args.dr2),
        )

        self.n_dim = self.compute_dim(args.num_channels, T)
        self.classifier = nn.Linear(self.n_dim, 512, bias=True)

    def forward(self, x):
        # 1, 1, 128, 300
        # x = self.down1(x)
        x = self.conv1(x) # 1, 16, 128, 300
        x = self.conv2(x) # 1, 32, 1, 150
        x = self.conv3(x) # 1, 32, 1, 37
        x = x.view(-1, self.n_dim)
        x = self.classifier(x)
        return x

    def compute_dim(self, num_channels, T):
        x = torch.zeros((1, 1, num_channels, T))

        # x = self.down1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x.size()[1] * x.size()[2] * x.size()[3]


class SpatialAttention(nn.Module):
    """Same as SpatialAttentionVer2, but a little more concise"""

    def __init__(self, args):
        super(SpatialAttention, self).__init__()

        # vectorize of k's and l's
        a = []
        for k in range(args.K):
            for l in range(args.K):
                a.append((k, l))
        a = torch.tensor(a)
        k, l = a[:, 0], a[:, 1]

        # vectorize x- and y-positions of the sensors
        loc = ch_locations_2d(args)
        x, y = loc[:, 0], loc[:, 1]

        # make a complex-valued parameter, reshape k,l into one dimension
        self.z = nn.Parameter(torch.rand(size=(args.D1, args.K ** 2), dtype=torch.cfloat)).to(
            device
        )

        # NOTE: pre-compute the values of cos and sin (they depend on k, l, x and y which repeat)
        phi = (
            2 * torch.pi * (torch.einsum("k,x->kx", k, x) + torch.einsum("l,y->ly", l, y))
        )  # torch.Size([1024, 60]))
        self.cos = torch.cos(phi).to(device)
        self.sin = torch.sin(phi).to(device)

        # self.spatial_dropout = SpatialDropoutX(args)
        self.spatial_dropout = SpatialDropout(loc, args.d_drop)

    def forward(self, X):
        """X: (batch_size, num_channels, T)"""

        # NOTE: do hadamard product and and sum over l and m (i.e. m, which is l X m)
        re = torch.einsum("jm, me -> je", self.z.real, self.cos)  # torch.Size([270, 60])
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
            mask = torch.where(distances < self.d_drop, 0.0, 1.0).to(device)  # ( num_channels, )
            return torch.einsum("c,bct->bct", mask, X)
        else:
            return X


class SubjectBlock(nn.Module):
    def __init__(self, args):
        super(SubjectBlock, self).__init__()

        self.num_subjects = args.num_subjects
        self.D1 = args.D1
        self.K = args.K
        self.spatial_attention = SpatialAttention(args)
        self.conv = nn.Conv1d(in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1)
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

    def forward(self, X, subject_idxs):
        X = self.spatial_attention(X)  # ( B, 270, 256 )
        X = self.conv(X)  # ( B, 270, 256 )
        X = torch.cat(
            [self.subject_layer[i](x.unsqueeze(dim=0)) for i, x in zip(subject_idxs, X)]
        )  # ( B, 270, 256 )
        return X


class ConvBlock(nn.Module):
    def __init__(self, k, D1, D2, ks=3):
        super(ConvBlock, self).__init__()

        self.k = k
        self.D2 = D2
        self.in_channels = D1 if k == 0 else D2

        self.conv0 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.D2,
            kernel_size=ks,
            padding="same",
            # dilation= 2 ** ((2 * k) % 5),
        )
        self.batchnorm0 = nn.BatchNorm1d(num_features=self.D2)
        self.conv1 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=self.D2,
            kernel_size=ks,
            padding="same",
            # dilation=2 ** ((2 * k + 1) % 5),
        )
        self.batchnorm1 = nn.BatchNorm1d(num_features=self.D2)
        self.conv2 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=2 * self.D2,
            kernel_size=ks,
            padding="same",
            # dilation=2,  # FIXME: The text doesn't say this, but the picture shows dilation=2
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


class LinearEncoder(nn.Module):
    def __init__(self, args):
        super(LinearEncoder, self).__init__()
        input_size = args.channel_size
        self.linear = nn.Linear(in_features=input_size, out_features=512, bias=True)
        self.scp = args.scp


    def forward(self, X, subject_idxs):
        if self.scp:
            X = X.mean(dim=-1) # X: batch x ch x time
        # import pdb; pdb.set_trace()
        return self.linear(X)



class BrainEncoder(nn.Module):
    def __init__(self, args):
        super(BrainEncoder, self).__init__()

        self.num_subjects = args.num_subjects
        self.D1 = args.D1
        self.D2 = args.D2
        self.F = args.F if not args.preprocs["last4layers"] else 1024
        self.K = args.K
        self.dataset_name = args.dataset

        self.subject_block = SubjectBlock(args)
        # self.subject_block = SubjectBlock_proto(args)
        # cprint("USING THE OLD IMPLEMENTATION OF THE SUBJECT BLOCK", 'red', 'on_blue', attrs=['bold'])

        self.conv_blocks = nn.Sequential()
        for k in range(5):
            self.conv_blocks.add_module(f"conv{k}", ConvBlock(k, self.D1, self.D2))

        self.conv_final1 = nn.Conv1d(in_channels=self.D2, out_channels=2 * self.D2, kernel_size=1,)
        self.conv_final2 = nn.Conv1d(in_channels=2 * self.D2, out_channels=self.F, kernel_size=1,)
        if args.seq2seq:
            self._forward = self._forward_seq_seq
        else:
            self._forward = self._forward_seq_static

    def forward(self, X, subject_idxs):
        return self._forward(X, subject_idxs)

    def _forward_seq_seq(self, X, subject_idxs):
        X = self.subject_block(X, subject_idxs)
        X = self.conv_blocks(X)
        X = F.gelu(self.conv_final1(X))
        X = F.gelu(self.conv_final2(X))
        return X

    def _forward_seq_static(self, X, subject_idxs):
        X = self.subject_block(X, subject_idxs)
        X = self.conv_blocks(X)
        X = F.gelu(self.conv_final1(X))
        X = F.gelu(self.conv_final2(X))
        X = torch.mean(X, axis=2)
        return X


class Classifier(nn.Module):
    # NOTE: experimental

    def __init__(self, args):
        super(Classifier, self).__init__()

        # NOTE: Do we need to adjust the accuracies for the dataset size?
        self.factor = 1  # self.batch_size / 241
        self.normalize_image_features = args.normalize_image_features

    def normalize_per_unit(self, tensor):
        print('normalize image_feature along unit dim')
        # array: n_samples x n_units(512)
        tensor = tensor - torch.mean(tensor, 0, keepdim=True)
        tensor = tensor / torch.std(tensor, 0,  keepdim=True)
        return tensor

    @torch.no_grad()
    def forward(self, Z: torch.Tensor, Y: torch.Tensor, test=False, top_k=None) -> torch.Tensor:

        batch_size = Z.size(0)
        diags = torch.arange(batch_size).to(device)
        x = Z.view(batch_size, -1)
        y = Y.view(batch_size, -1)


        if self.normalize_image_features:
            # y = self.normalize_per_unit(y)
            pass
        # x_ = rearrange(x, 'b f -> 1 b f')
        # y_ = rearrange(y, 'b f -> b 1 f')
        # similarity = torch.nn.functional.cosine_similarity(x_, y_, dim=-1)  # ( B, B )

        # NOTE: avoid CUDA out of memory like this
        similarity = torch.empty(batch_size, batch_size).to(device)

        if test:
            pbar = tqdm(total=batch_size, desc="[Similarities]")

        for i in range(batch_size):
            for j in range(batch_size):
                similarity[i, j] = (x[i] @ y[j]) / max((x[i].norm() * y[j].norm()), 1e-8)

            if test:
                pbar.update(1)

        similarity = similarity.T

        # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
        top1accuracy = (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()
        try:
            top10accuracy = np.mean(
                [
                    label in row
                    for row, label in zip(torch.topk(similarity, 10, dim=1, largest=True)[1], diags)
                ]
            )
        except:
            print(similarity.size())
            raise
        if top_k is None:

            return top1accuracy, top10accuracy
        else:
            try:
                topkaccuracy = np.mean(
                    [
                        label in row
                        for row, label in zip(torch.topk(similarity, top_k, dim=1, largest=True)[1], diags)
                    ]
                    )
            except:
                print(similarity.size())
                raise
            return top1accuracy, top10accuracy, topkaccuracy




class BrainEncoderSeq2Static(nn.Module):
    def __init__(self, args):
        super( BrainEncoderSeq2Static, self).__init__()

        self.num_subjects = args.num_subjects
        self.D1 = args.D1
        self.D2 = args.D2
        self.F = args.F if not args.preprocs["last4layers"] else 1024
        self.K = args.K
        self.dataset_name = args.dataset

        self.subject_block = SubjectBlock(args)
        # self.subject_block = SubjectBlock_proto(args)
        # cprint("USING THE OLD IMPLEMENTATION OF THE SUBJECT BLOCK", 'red', 'on_blue', attrs=['bold'])

        self.conv_blocks = nn.Sequential()
        ks_list = args.ConvBlocks.ks  # 200 ms = 200 samples  receptive field: ks * 3 * stride
        for k in range(5):
            ks = ks_list[k]
            self.conv_blocks.add_module(f"conv{k}", ConvBlock(k, self.D1, self.D2, ks=ks))
            if k < 4:
                self.conv_blocks.add_module(f"avgpool{k}", torch.nn.AvgPool1d(3, stride=2))
            else:
                self.conv_blocks.add_module(f"globalavgpool{k}", torch.nn.AdaptiveAvgPool1d(output_size=1))
        self.conv_final1 = nn.Conv1d(in_channels=self.D2, out_channels=2 * self.D2, kernel_size=1,)
        self.conv_final2 = nn.Conv1d(in_channels=2 * self.D2, out_channels=self.F, kernel_size=1,)
        if args.seq2seq:
            self._forward = self._forward_seq_seq
        else:
            self._forward = self._forward_seq_static

    def forward(self, X, subject_idxs):
        return self._forward(X, subject_idxs)

    def _forward_seq_seq(self, X, subject_idxs):
        X = self.subject_block(X, subject_idxs)
        X = self.conv_blocks(X)
        X = F.gelu(self.conv_final1(X))
        X = F.gelu(self.conv_final2(X))
        return X

    def _forward_seq_static(self, X, subject_idxs):
        X = self.subject_block(X, subject_idxs)
        X = self.conv_blocks(X)
        X = F.gelu(self.conv_final1(X))
        X = F.gelu(self.conv_final2(X))
        X = torch.mean(X, axis=2)
        return X
