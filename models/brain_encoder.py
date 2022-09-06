import sys
import numpy as np
import torch
import torch.nn as nn
import mne
from time import time

def ch_locations_2d(montage_name="easycap-M10"):
    montage = mne.channels.make_standard_montage(montage_name)
    info = mne.create_info(ch_names=montage.ch_names, sfreq=512., ch_types="eeg")
    info.set_montage(montage)

    layout = mne.channels.find_layout(info, ch_type="eeg")

    loc = layout.pos[:,:2] # ( 61, 2 )

    if montage_name == "easycap-M10":
        # Channel 29 was broken in Brennan 2018
        loc = np.delete(loc, 28, axis=0) # ( 60, 2 )

    # min-max normalization
    loc = (loc - loc.min(axis=0)) / (loc.max(axis=0) - loc.min(axis=0))

    return torch.from_numpy(loc.astype(np.float32))


class SpatialAttentionOrig(nn.Module):
    """This is easier to understand but very slow. I reimplemented to SpatialAttention"""

    def __init__(self, z_re=None, z_im=None, D1=270, K=32, montage_name="easycap-M10"):
        super(SpatialAttentionOrig, self).__init__()

        self.D1 = D1
        self.K = K

        if z_re is None or z_im is None:
            self.z_re = nn.Parameter(torch.Tensor(self.D1, self.K, self.K))
            self.z_im = nn.Parameter(torch.Tensor(self.D1, self.K, self.K))
            nn.init.kaiming_uniform_(self.z_re, a=np.sqrt(5))
            nn.init.kaiming_uniform_(self.z_im, a=np.sqrt(5))
        else:
            self.z_re = z_re
            self.z_im = z_im

        self.ch_locations_2d = ch_locations_2d(montage_name).cuda()

    def fourier_space(self, j, x:torch.Tensor, y:torch.Tensor): # x: ( 60, ) y: ( 60, )
        a_j = 0
        for k in range(self.K):
            for l in range(self.K):
                a_j += self.z_re[j, k, l] * torch.cos(2 * torch.pi * (k * x + l * y))
                a_j += self.z_im[j, k, l] * torch.sin(2 * torch.pi * (k * x + l * y))

        return a_j # ( 60, )

    def forward(self, X): # ( B, C, T ) (=( 128, 60, 256 ))
        spat_attn = []
        loc = self.ch_locations_2d # ( 60, 2 )
        for j in range(self.D1):
            a_j = self.fourier_space(j, loc[:,0], loc[:,1]) # ( 60, )

            # sa.append(torch.exp(a_j) @ X / torch.exp(a_j).sum()) # ( 128, 256 )
            spat_attn.append(torch.einsum('c,bct->bt', torch.exp(a_j), X) / torch.exp(a_j).sum()) # ( 128, 256 )

        spat_attn = torch.stack(spat_attn) # ( 270, 128, 256 )

        return spat_attn.permute(1,0,2) # ( 128, 270, 256 )


class SpatialAttention(nn.Module):
    """Faster version of SpatialAttentionOrig"""

    def __init__(self, D1=270, K=32, montage_name="easycap-M10"):
        super(SpatialAttention, self).__init__()

        self.D1 = D1
        self.K = K

        self.z_re = nn.Parameter(torch.Tensor(self.D1, self.K, self.K))
        self.z_im = nn.Parameter(torch.Tensor(self.D1, self.K, self.K))
        nn.init.kaiming_uniform_(self.z_re, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.z_im, a=np.sqrt(5))

        self.K_arange = torch.arange(self.K).cuda()

        self.ch_locations_2d = ch_locations_2d(montage_name).cuda()


    def fourier_space(self, x:torch.Tensor, y:torch.Tensor): # x: ( 60, ) y: ( 60, )

        rad1 = torch.einsum('k,c->kc', self.K_arange, x)
        rad2 = torch.einsum('l,c->lc', self.K_arange, y)
        # rad = torch.einsum('kc,lc->kcl', rad1, rad2)

        # ( 32, 1, 60 ) + ( 1, 32, 60 ) -> ( 32, 32, 60 )
        rad = rad1.unsqueeze(1) + rad2.unsqueeze(0)

        real = torch.einsum('dkl,klc->dc', self.z_re, torch.cos(2 * torch.pi * rad)) # ( 270, 60 )
        imag = torch.einsum('dkl,klc->dc', self.z_im, torch.sin(2 * torch.pi * rad))

        return real + imag # ( 270, 60 )


    def fourier_space_orig(self, x:torch.Tensor, y:torch.Tensor): # x: ( 60, ) y: ( 60, )
        """Slower version of fourier_space"""

        a = torch.zeros(self.D1, x.shape[0], device='cuda') # ( 270, 60 )
        for k in range(self.K):
            for l in range(self.K):
                # This einsum is same as torch.stack([_d * c for _d in d])
                a += torch.einsum('d,c->dc',
                    self.z_re[:, k, l], torch.cos(2 * torch.pi * (k * x + l * y))
                ) # ( 270, 60 )
                a += torch.einsum('d,c->dc',
                    self.z_im[:, k, l], torch.sin(2 * torch.pi * (k * x + l * y))
                )

        return a # ( 270, 60 )


    def forward(self, X): # ( 128, 60, 256 )
        loc = self.ch_locations_2d # ( 60, 2 )

        a = self.fourier_space(loc[:,0], loc[:,1]) # ( 270, 60 )
        # _a = self.fourier_space_orig(loc[:,0], loc[:,1]) # ( 270, 60 )
        # print(torch.equal(_a, a))

        # ( 270, 60 ) @ ( 128, 60, 256 ) -> ( 128, 256, 270 )
        spat_attn = torch.einsum('dc,bct->btd', torch.exp(a), X) / torch.exp(a).sum(dim=1)

        return spat_attn.permute(0,2,1) # ( 128, 270, 256 )


class SubjectBlock(nn.Module):
    def __init__(self, num_subjects=19, D1=270):
        super(SubjectBlock, self).__init__()

        self.num_subjects = num_subjects
        self.D1 = D1

        self.spatial_attention = SpatialAttention()
        # self.spatial_attention = SpatialAttentionOrig()
        self.conv = nn.Conv1d(
            in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1
        )
        self.subject_matrix = nn.Parameter(
            torch.rand(self.num_subjects, self.D1, self.D1)
        )

    def forward(self, X, s=0):
        X = self.spatial_attention(X) # ( B, 270, 256 )
        X = self.conv(X) # ( B, 270, 256 )
        X = self.subject_matrix[s] @ X # ( B, 270, 256 )

        return X


class ConvBlock(nn.Module):
    def __init__(self, k, D1=270, D2=320):
        super(ConvBlock, self).__init__()

        self.k = k
        self.D2 = D2
        self.in_channels = D1 if k==1 else D2

        self.conv1 = nn.Conv1d(
            in_channels=self.in_channels, out_channels=self.D2,
            kernel_size=3, padding='same', dilation=2**(2 * self.k % 5),
        )
        self.batchnorm1 = nn.BatchNorm1d(num_features=self.D2)
        self.conv2 = nn.Conv1d(
            in_channels=self.D2, out_channels=self.D2,
            kernel_size=3, padding='same', dilation=2**(2 * self.k + 1 % 5),
        )
        self.batchnorm2 = nn.BatchNorm1d(num_features=self.D2)
        self.conv3 = nn.Conv1d(
            in_channels=self.D2, out_channels=2*self.D2,
            kernel_size=3, padding='same', dilation=2,
        )

    def forward(self, X):
        if self.k == 1:
            X = self.conv1(X)
        else:
            X = self.conv1(X) + X # skip connection
        X = nn.GELU()(self.batchnorm1(X))

        X = self.conv2(X) + X # skip connection
        X = nn.GELU()(self.batchnorm2(X))

        X = self.conv3(X)
        X = nn.GLU(dim=-2)(X)

        return X # ( B, 320, 256 )


class BrainEncoder(nn.Module):
    def __init__(self, D2=320, F=512):
        super(BrainEncoder, self).__init__()

        self.D2 = D2
        self.F = F

        self.subject_block = SubjectBlock()

        self.conv_blocks = nn.Sequential()
        for k in range(1,6):
            self.conv_blocks.add_module(f"conv{k}", ConvBlock(k))

        self.conv_final1 = nn.Conv1d(
            in_channels=self.D2, out_channels=2*self.D2, kernel_size=1,
        )
        self.conv_final2 = nn.Conv1d(
            in_channels=2*self.D2, out_channels=self.F, kernel_size=1,
        )

    def forward(self, X):
        X = self.subject_block(X)
        X = self.conv_blocks(X)
        X = self.conv_final1(X)
        X = self.conv_final2(X)

        return X


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)

    brain_encoder = BrainEncoder().cuda()
    # brain_encoder = SpatialAttention().cuda()
    # brain_encoder = SubjectBlock().cuda()
    # brain_encoder_ = SpatialAttentionOrig(
    #     brain_encoder.z_re.clone(), brain_encoder.z_im.clone()
    # ).cuda()
    
    X = torch.rand(128, 60, 256).cuda()
    # X.requires_grad = True

    Z = brain_encoder(X) # ( 512, 270, 256 )

    # Z_ = brain_encoder_(X)

    # print(torch.equal(Z, Z_))

    # print((Z - Z_).sum())


    # stime = time()
    # grad = torch.autograd.grad(
    #     outputs=Z, inputs=X, grad_outputs=torch.ones_like(Z),
    #     create_graph=True, retain_graph=True, only_inputs=True
    # )[0]
    # print(f"grad {time() - stime}")