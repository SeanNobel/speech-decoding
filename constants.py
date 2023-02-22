import torch, random
import numpy as np
# assert torch.cuda.is_available(), "Training without GPU is not supported."
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
bar_format = '{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'

# NOTE: I'm using GPU on my machine so heavily on other project that I can't use
# device = "cpu"