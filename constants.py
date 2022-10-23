import torch
# assert torch.cuda.is_available(), "Training without GPU is not supported."
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# NOTE: I'm using GPU on my machine so heavily on other project that I can't use
# device = "cpu"