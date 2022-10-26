import sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from utils.layout import ch_locations_2d
import torch.nn.functional as F
from constants import device
from termcolor import cprint


class SpatialAttentionVer1(nn.Module):
    """This is easier to understand but very slow. I reimplemented to SpatialAttentionVer2"""

    def __init__(self, args, z_re=None, z_im=None):
        super(SpatialAttentionVer1, self).__init__()

        self.D1 = args.D1
        self.K = args.K

        if z_re is None or z_im is None:
            self.z_re = nn.Parameter(torch.Tensor(self.D1, self.K, self.K))
            self.z_im = nn.Parameter(torch.Tensor(self.D1, self.K, self.K))
            nn.init.kaiming_uniform_(self.z_re, a=np.sqrt(5))
            nn.init.kaiming_uniform_(self.z_im, a=np.sqrt(5))
        else:
            self.z_re = z_re
            self.z_im = z_im

        self.ch_locations_2d = ch_locations_2d(args.dataset).cuda()

    def fourier_space(self, j, x: torch.Tensor, y: torch.Tensor):  # x: ( 60, ) y: ( 60, )
        a_j = 0
        for k in range(self.K):
            for l in range(self.K):
                a_j += self.z_re[j, k, l] * torch.cos(2 * torch.pi * (k * x + l * y))
                a_j += self.z_im[j, k, l] * torch.sin(2 * torch.pi * (k * x + l * y))

        return a_j  # ( 60, )

    def forward(self, X):  # ( B, C, T ) (=( 128, 60, 256 ))
        spat_attn = []
        loc = self.ch_locations_2d  # ( 60, 2 )
        for j in range(self.D1):
            a_j = self.fourier_space(j, loc[:, 0], loc[:, 1])  # ( 60, )

            # sa.append(torch.exp(a_j) @ X / torch.exp(a_j).sum()) # ( 128, 256 )
            spat_attn.append(torch.einsum('c,bct->bt', torch.exp(a_j), X) / torch.exp(a_j).sum())  # ( 128, 256 )

        spat_attn = torch.stack(spat_attn)  # ( 270, 128, 256 )

        return spat_attn.permute(1, 0, 2)  # ( 128, 270, 256 )


class SpatialAttentionVer2(nn.Module):
    """Faster version of SpatialAttentionVer1"""

    def __init__(self, args):
        super(SpatialAttentionVer2, self).__init__()

        self.D1 = args.D1
        self.K = args.K

        self.z_re = nn.Parameter(torch.Tensor(self.D1, self.K, self.K))
        self.z_im = nn.Parameter(torch.Tensor(self.D1, self.K, self.K))
        nn.init.kaiming_uniform_(self.z_re, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.z_im, a=np.sqrt(5))

        self.K_arange = torch.arange(self.K).to(device)

        self.ch_locations_2d = ch_locations_2d(args.dataset).cuda()

    def fourier_space(self, x: torch.Tensor, y: torch.Tensor):  # x: ( 60, ) y: ( 60, )

        rad1 = torch.einsum('k,c->kc', self.K_arange, x)
        rad2 = torch.einsum('l,c->lc', self.K_arange, y)
        # rad = torch.einsum('kc,lc->kcl', rad1, rad2)

        # ( 32, 1, 60 ) + ( 1, 32, 60 ) -> ( 32, 32, 60 )
        rad = rad1.unsqueeze(1) + rad2.unsqueeze(0)

        real = torch.einsum('dkl,klc->dc', self.z_re, torch.cos(2 * torch.pi * rad))  # ( 270, 60 )
        imag = torch.einsum('dkl,klc->dc', self.z_im, torch.sin(2 * torch.pi * rad))

        return real + imag  # ( 270, 60 )

    def fourier_space_orig(self, x: torch.Tensor, y: torch.Tensor):  # x: ( 60, ) y: ( 60, )
        """Slower version of fourier_space"""

        a = torch.zeros(self.D1, x.shape[0], device=device)  # ( 270, 60 )
        for k in range(self.K):
            for l in range(self.K):
                # This einsum is same as torch.stack([_d * c for _d in d])
                a += torch.einsum('d,c->dc', self.z_re[:, k, l],
                                  torch.cos(2 * torch.pi * (k * x + l * y)))  # ( 270, 60 )
                a += torch.einsum('d,c->dc', self.z_im[:, k, l], torch.sin(2 * torch.pi * (k * x + l * y)))

        return a  # ( 270, 60 )

    def forward(self, X):  # ( 128, 60, 256 )
        loc = self.ch_locations_2d  # ( 60, 2 )

        a = self.fourier_space(loc[:, 0], loc[:, 1])  # ( 270, 60 )
        # _a = self.fourier_space_orig(loc[:,0], loc[:,1]) # ( 270, 60 )
        # print(torch.equal(_a, a))

        # ( 270, 60 ) @ ( 128, 60, 256 ) -> ( 128, 256, 270 )
        spat_attn = torch.einsum('dc,bct->btd', torch.exp(a), X) / torch.exp(a).sum(dim=1)

        return spat_attn.permute(0, 2, 1)  # ( 128, 270, 256 )


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
        loc = ch_locations_2d(args.dataset)
        x, y = loc[:, 0], loc[:, 1]

        # make a complex-valued parameter, reshape k,l into one dimension
        self.z = nn.Parameter(torch.rand(size=(args.D1, args.K**2), dtype=torch.cfloat)).to(device)

        # NOTE: pre-compute the values of cos and sin (they depend on k, l, x and y which repeat)
        phi = 2 * torch.pi * (torch.einsum('k,x->kx', k, x) + torch.einsum('l,y->ly', l, y))  # torch.Size([1024, 60]))
        self.cos = torch.cos(phi).to(device)
        self.sin = torch.sin(phi).to(device)

        # self.spatial_dropout = SpatialDropoutX(args)
        self.spatial_dropout = SpatialDropout(loc, args.d_drop)

    def forward(self, X):

        # NOTE: do hadamard product and and sum over l and m (i.e. m, which is l X m)
        re = torch.einsum('jm, me -> je', self.z.real, self.cos)  # torch.Size([270, 60])
        im = torch.einsum('jm, me -> je', self.z.imag, self.sin)
        a = re + im  # essentially (unnormalized) weights with which to mix input channels into ouput channels
        # ( D1, num_channels )

        # NOTE: to get the softmax spatial attention weights over input electrodes,
        # we don't compute exp, etc (as in the eq. 5), we take softmax instead:
        SA_wts = F.softmax(a, dim=-1)  # each row sums to 1
        # ( D1, num_channels )

        # NOTE: drop some channels within a d_drop of the sampled channel
        dropped_X = self.spatial_dropout(X)

        # NOTE: each output is a diff weighted sum over each input channel
        return torch.einsum('oi,bit->bot', SA_wts, dropped_X)


class SpatialDropout(nn.Module):

    def __init__(self, loc, d_drop):
        super(SpatialDropout, self).__init__()
        self.loc = loc  # ( num_channels, 2 )
        self.d_drop = d_drop
        self.num_channels = loc.shape[0]

    def make_mask(self, batch_size):
        drop_center_idxs = np.random.randint(self.num_channels, size=batch_size)  # ( B, )
        drop_centers = self.loc[drop_center_idxs]  # ( B, 2 )
        distances = (self.loc.unsqueeze(0) - drop_centers.unsqueeze(1)).norm(dim=-1)
        # ( B, num_channels )
        mask = torch.where(distances < self.d_drop, 0., 1.)
        # cprint(mask.shape, color="yellow")
        # cprint(
        #     f"{self.num_channels - int(is_dropped.sum())} channels were dropped.",
        #     color="cyan")

        return mask.to(device)  # ( B, num_channels )

    def forward(self, X):  # ( B, num_channels, seq_len )
        assert X.shape[1] == self.num_channels

        if self.training:
            mask = self.make_mask(X.shape[0])  # ( B, num_channels )
            return torch.einsum('bc,bct->bct', mask, X)
        else:
            return X


class SpatialDropoutX(nn.Module):
    # NOTE: each item in a batch gets the same channels masked

    def __init__(self, args):
        super(SpatialDropoutX, self).__init__()
        self.d_drop = args.d_drop
        self.bsz = args.batch_size

        loc = ch_locations_2d(args.dataset)
        self.loc = [loc[i, :].flatten() for i in range(loc.shape[0])]

    def make_mask(self):
        # TODO: could just pre-compute all the possilbe drop locations
        mask = torch.ones(size=(self.bsz, len(self.loc)))
        for b in range(self.bsz):
            drop_id = np.random.choice(len(self.loc))
            drop_center = self.loc[drop_id]
            for i, coord in enumerate(self.loc):
                if (coord - drop_center).norm() < self.d_drop:
                    mask[b, i] *= 0.0
        return mask.to(device)

    def forward(self, X):
        print(self.training)
        if self.training:
            mask = self.make_mask()  # mask: B by num_chans. Each item in batch gets a different mask
            return torch.einsum('bc,bct->bct', mask, X)
        else:
            return X


class SubjectBlock(nn.Module):

    def __init__(self, args):
        super(SubjectBlock, self).__init__()

        self.num_subjects = args.num_subjects
        self.D1 = args.D1
        self.K = args.K
        self.spatial_attention = SpatialAttention(args)
        # self.spatial_attention = SpatialAttentionVer2(args)
        # self.spatial_attention = SpatialAttentionVer1()
        self.conv = nn.Conv1d(in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1)

        # NOTE: The below implementations are equivalent to learning a matrix:
        self.subject_matrix = nn.Parameter(torch.rand(self.num_subjects, self.D1, self.D1))
        # self.subject_layer = [
        #     nn.Conv1d(in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1, device=device)
        #     for _ in range(self.num_subjects)
        # ]

    def forward(self, X, subject_idxs):
        X = self.spatial_attention(X)  # ( B, 270, 256 )
        X = self.conv(X)  # ( B, 270, 256 )

        # TODO: make this more efficient
        X = self.subject_matrix[subject_idxs] @ X  # ( 270, 270 ) @ ( B , 270, 256 ) -> ( B, 270, 256 )
        # _X = []
        # for i, x in enumerate(X):  # x: ( 270, 256 )
        #     x = self.subject_layer[subject_idxs[i]](x.unsqueeze(0))  # ( 1, 270, 256 )
        #     _X.append(x.squeeze())
        # X = torch.stack(_X)

        return X  # ( B, 270, 256 )


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
            padding='same',
            dilation=2**((2 * k) % 5),
        )
        self.batchnorm0 = nn.BatchNorm1d(num_features=self.D2)
        self.conv1 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=self.D2,
            kernel_size=3,
            padding='same',
            dilation=2**((2 * k + 1) % 5),
        )
        self.batchnorm1 = nn.BatchNorm1d(num_features=self.D2)
        self.conv2 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=2 * self.D2,
            kernel_size=3,
            padding='same',
            dilation=2,  #FIXME: The text doesn't say this, but the picture shows dilation=2
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

    def __init__(self, args):
        super(BrainEncoder, self).__init__()

        self.num_subjects = args.num_subjects
        self.D1 = args.D1
        self.D2 = args.D2
        self.F = args.F if not args.preprocs["last4layers"] else 1024
        self.K = args.K
        self.dataset_name = args.dataset

        self.subject_block = SubjectBlock(args)

        self.conv_blocks = nn.Sequential()
        for k in range(5):
            self.conv_blocks.add_module(f"conv{k}", ConvBlock(k, self.D1, self.D2))

        self.conv_final1 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=2 * self.D2,
            kernel_size=1,
        )
        self.conv_final2 = nn.Conv1d(
            in_channels=2 * self.D2,
            out_channels=self.F,
            kernel_size=1,
        )

    def forward(self, X, subject_idxs):
        X = self.subject_block(X, subject_idxs)
        X = self.conv_blocks(X)
        # print(X.shape)
        X = nn.GELU()(self.conv_final1(X))
        # print(X.shape)
        X = nn.GELU()(self.conv_final2(X))

        return X


if __name__ == '__main__':
    from configs.args import args

    batch_size = 2

    # torch.autograd.set_detect_anomaly(True)

    brain_encoder = BrainEncoder(args).to(device)
    # brain_encoder = SpatialAttention().cuda()
    # brain_encoder = SubjectBlock().cuda()
    # brain_encoder_ = SpatialAttentionVer1(
    #     brain_encoder.z_re.clone(), brain_encoder.z_im.clone()
    # ).cuda()

    X = torch.rand(batch_size, 208, 256).to(device)
    X.requires_grad = False

    subject_idxs = torch.randint(args.num_subjects, size=(batch_size,))

    # spatial_attention = SpatialAttention(D1=args.D1,
    #                                      K=args.K,
    #                                      dataset_name=args.dataset).to(device)
    # spatial_attention_x = SpatialAttentionX(
    #     D1=args.D1, K=args.K, dataset_name=args.dataset).to(device)

    # output = spatial_attention(X)
    # output_x = spatial_attention_x(X)

    # print(torch.equal(output, output_x))

    Z = brain_encoder(X, subject_idxs)

    # print(torch.equal(Z, Z_))

    # print((Z - Z_).sum())

    # stime = time()
    # grad = torch.autograd.grad(outputs=Z,
    #                            inputs=X,
    #                            grad_outputs=torch.ones_like(Z),
    #                            create_graph=True,
    #                            retain_graph=True,
    #                            only_inputs=True)[0]
    # print(f"grad {time() - stime}")