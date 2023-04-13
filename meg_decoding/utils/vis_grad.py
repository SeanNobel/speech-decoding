import torch


def get_grad(model:torch.nn.Module):
    for n, m in model.named_parameters():
        print(n, m.grad.sum())


