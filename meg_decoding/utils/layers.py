import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F

# BatchNorm2Dをsubject specificにするだけでいい気もする
class ReadoutNorm2D(nn.Module):
    def __init__(self, n_subs):
        super(ReadoutNorm2D, self).__init__()
        self.n_subs = n_subs

    def forward(self, x:torch.Tensor, sub:torch.Tensor)->torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): inputs for this layer. batch_size x n_ch x height(originally for electrodes) x width(originally for time)
            sub (torch.Tensor): subject ids. batch_size

        Returns:
            torch.Tensor: normalized inputs. batch_size x n_ch x height x width
        """
        for s in range(self.n_subs):
            s_inds = (sub==s).to(torch.long)
            if s_inds.sum() == 0:
                continue
            # across trial normalization
            with torch.no_grad():
                s_mean = x[s_inds].mean(dim=0, keepdims=True)
                s_std = x[s_inds].std(dim=0, keepdims=True)
            x[s_inds] = (x[s_inds] - s_mean) / s_std
            # across temporal normalization
            with torch.no_grad():
                s_mean = x[s_inds].mean(dim=-1, keepdims=True)
                s_std = x[s_inds].std(dim=-1, keepdims=True)
            x[s_inds] = (x[s_inds] - s_mean) / s_std
        return x


class SubBatchNorm2D(nn.Module):
    def __init__(self, n_dims, n_subs):
        super(SubBatchNorm2D, self).__init__()
        self.n_dims = n_dims
        self.n_subs = n_subs
        self.bns = nn.ModuleList([nn.BatchNorm2d(n_dims) for _ in range(n_subs)])

    def forward(self, x:torch.Tensor, sub:torch.Tensor)->torch.Tensor:
        # indices = []
        # hs = []
        ret_tensor = torch.zeros_like(x).to(x.device)
        for s in range(self.n_subs):
            s_inds = torch.where(sub==s)[0]
            if len(s_inds) == 0:
                continue
            # indices.append(s_inds)
            h =  self.bns[s](torch.index_select(x, 0, s_inds))
            # hs.append(h)
            # ret_tensor[s_inds[i]]に　h[i]の値を入れる(hのiは本来s_inds[i]に存在するべき値)
            # ret_tensor.scatter_(0, h, s_inds)
            ret_tensor[s_inds] = h

        # hs = torch.cat(hs, dim=0)
        # indices = torch.cat(indices, dim=0)
        # # hs（n_sub_trials x ch x h x w） をindicesに従って
        # x = torch.gather(hs, 0, indices)
        return ret_tensor


class DeepSetConv2D(nn.Module):
    def __init__(self, input_ch:int, middle_ch:int, output_ch:int,
                 ks1:int, ks2:int, n_subs:int):
        super(DeepSetConv2D, self).__init__()
        self.n_subs = n_subs
        Gamma = Parameter(torch.empty(
                (input_ch, middle_ch, 1, 1)))
        Lambda = Parameter(torch.empty(
                (middle_ch, output_ch, 1, 1)))

        self.weight1 = Gamma.repeat(1, 1, ks1, ks1)
        self.weight2 = Lambda.repeat(1, 1, ks2, ks2)

        self.stride1 = 1
        self.stride2 = 1
        # self.padding =


    def forward(self, x:torch.Tensor, sub:torch.Tensor)->torch.Tensor:
        h = F.conv2d(input, self.weight1, None, self.stride,
                        self.padding, self.dilation, self.groups)



class CogitatDeepSetNorm(nn.Module):
    def __init__(self, input_dims:int, middle_dims:int, output_dims:int,
                 n_subs:int):
        super(CogitatDeepSetNorm, self).__init__()
        self.n_subs = n_subs
        self.fc1 = nn.Linear(input_dims, middle_dims, bias=False)
        self.fc2 = nn.Linear(middle_dims, output_dims, bias=False)
        self.act = nn.ReLU()


    def forward(self, x:torch.Tensor, sub:torch.Tensor)->torch.Tensor:
        # x: batch_size x input_dims
        mean_list = []
        for s in range(self.n_subs):
            indices =torch.where(sub == s)[0]
            if indices.sum() == 0:
                continue
            # mean across subject-trials
            m = torch.index_select(x, 0, indices)
            mean_list.append(m.mean(dim=0, keepdims=True))
        stats = torch.cat(mean_list, dim=0) # 1 x input_dims
        stats = self.fc1(stats) # 1 x middle_dims
        stats = self.act(stats)

        # aligned_stats = torch.zeros_like(stats)
        # aligned_stats.index_put_((sub,) , stats)
        # n_subs個の統計量をsubに従って分配
        stats = torch.gather(stats, 0, sub) # batch x middle_dims



        x = torch.cat([x, stats], dim=-1) # batch x (input_dims + middle_dims)
        x = self.fc2(x) # batch x output_dims
        x = self.act(x)
        return x






