import torch
# assert torch.cuda.is_available(), "Training without GPU is not supported."
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
