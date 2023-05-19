import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F

# BatchNorm2Dをsubject specificにするだけでいい気もする
class ReadoutNorm2D(nn.Module):
    def __init__(self, n_subs):
        super(ReadoutNorm2D, self).__init__()
        self.n_subs = n_subs
        self.epsilon = 1e-5

    def forward(self, x:torch.Tensor, sub:torch.Tensor)->torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): inputs for this layer. batch_size x n_ch x height(originally for electrodes) x width(originally for time)
            sub (torch.Tensor): subject ids. batch_size

        Returns:
            torch.Tensor: normalized inputs. batch_size x n_ch x height x width
        """
        for s in range(self.n_subs):
            s_inds = torch.where(sub==s)[0]
            if s_inds.sum() == 0:
                continue
            # across trial normalization
            with torch.no_grad():
                s_mean = x[s_inds].mean(dim=0, keepdims=True)
                s_std = x[s_inds].std(dim=0, keepdims=True) + self.epsilon
            x[s_inds] = (x[s_inds] - s_mean) / s_std
            # across temporal normalization
            with torch.no_grad():
                s_mean = x[s_inds].mean(dim=-1, keepdims=True)
                s_std = x[s_inds].std(dim=-1, keepdims=True) + self.epsilon
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
            s_inds = torch.where(sub==s)[0].to(x.device)
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
        # self.fc1 = nn.Linear(input_dims, middle_dims, bias=False)
        # self.fc2 = nn.Linear(input_dims+middle_dims, output_dims, bias=False)
        # これが被験者ごとなのか共通なのか不明
        self.Gamma = Parameter(torch.ones(1) / (middle_dims * input_dims), requires_grad=True)
        self.Lambda = Parameter(torch.ones(1)/ (output_dims * (input_dims+middle_dims)), requires_grad=True)
        # self.weight1 = 
        # self.weight2 = 
        self.act = nn.ReLU()
        self.input_dims = input_dims
        self.middle_dims = middle_dims
        self.output_dims = output_dims


    def forward(self, x:torch.Tensor, sub:torch.Tensor)->torch.Tensor:
        # x: batch_size x input_dims
        mean_list = []
        for s in range(self.n_subs):
            indices =torch.where(sub == s)[0].to(x.device)
            if len(indices) == 0:
                # dummy indices
                indices = torch.tensor([0]).to(torch.long).to(x.device)
            # mean across subject-trials
            m = torch.index_select(x, 0, indices)
            mean_list.append(m.mean(dim=0, keepdims=True))
        stats = torch.cat(mean_list, dim=0) # 1 x input_dims
        stats = F.linear(stats, self.Gamma.repeat(self.middle_dims,  self.input_dims) , None)
        stats = self.act(stats)

        # aligned_stats = torch.zeros_like(stats)
        # aligned_stats.index_put_((sub,) , stats)
        # n_subs個の統計量をsubに従って分配 これだと3sub必ずbatch内に含まれることを仮定している
        batch_stats = stats[sub,:] # torch.gather(stats, 0, sub) # batch x middle_dims



        x = torch.cat([x, batch_stats], dim=1) # batch x (input_dims + middle_dims)
        # x = self.fc2(x) # batch x output_dims
        x = F.linear(x, self.Lambda.repeat(self.output_dims, self.input_dims+self.middle_dims) , None)
        x = self.act(x)
        return x



if __name__ == '__main__':
    subs = torch.Tensor([0,1,2,0,1,2,0,1,2]).to(torch.long)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # CogitatDeepSetNorm test
    print('==================CogitatDeepSetNorm===================')
    ## build module
    cogitat_deepset_norm = CogitatDeepSetNorm(input_dims=96, middle_dims=8, output_dims=96, n_subs=3)
    cogitat_deepset_norm.to(device)
    ## prepare dummy input
    dummy_input = torch.ones([9,96], requires_grad=True).to(device)
    dummy_input[torch.where(subs==0)] *= 1
    dummy_input[torch.where(subs==1)] *= 2
    dummy_input[torch.where(subs==2)] *= 3
    print('dummy_input: ', dummy_input.shape)
    print(dummy_input.mean(dim=1))

    output = cogitat_deepset_norm(dummy_input, subs)
    print('output', output.shape)
    print(output.mean(dim=1))
    loss = output.mean()
    loss.backward()
    print('loss', loss)
    print('grad: ', dummy_input.grad)
    print('grad: ', cogitat_deepset_norm.Gamma.grad)

    # SubBatchNorm2D
    print('================SubBatchNorm2D===============')
    ## build module
    sub_bn2d = SubBatchNorm2D(16, 3).to(device)
    ## prepare dummy input
    dummy_input = torch.ones([9,16, 2, 2], requires_grad=True).to(device)
    dummy_input[torch.where(subs==0)] *= 1
    dummy_input[torch.where(subs==1)] *= 2
    dummy_input[torch.where(subs==2)] *= 3
    print('dummy_input: ', dummy_input.shape)
    print(dummy_input.mean(dim=1).mean(dim=1).mean(dim=1))

    output = sub_bn2d(dummy_input, subs)
    print('output', output.shape)
    print(output.mean(dim=1).mean(dim=1).mean(dim=1))
    loss = output.mean()
    loss.backward()
    print('loss', loss)
    print('grad: ', dummy_input.grad)



    # ReadoutNorm2D
    print('==================ReadoutNotm2D==================')
    ## build module
    readoutnorm = ReadoutNorm2D(3).to(device)
    ## prepare dummy input
    dummy_input = torch.ones([9,16,2,2], requires_grad=True).to(device)
    dummy_input[torch.where(subs==0)] *= 1
    dummy_input[torch.where(subs==1)] *= 2
    dummy_input[torch.where(subs==2)] *= 3
    print('dummy_input: ', dummy_input.shape)
    print(dummy_input.mean(dim=1).mean(dim=1).mean(dim=1))

    output = readoutnorm(dummy_input, subs)
    print('output', output.shape)
    print(output.mean(dim=1).mean(dim=1).mean(dim=1))
    loss = output.mean()
    loss.backward()
    print('loss', loss)
    print('grad: ', dummy_input.grad)
