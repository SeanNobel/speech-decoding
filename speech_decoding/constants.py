import torch, random
import numpy as np

# assert torch.cuda.is_available(), "Training without GPU is not supported."
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BAR_FORMAT = "{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"

# NOTE: I'm using GPU on my machine so heavily on other project that I can't use
# device = "cpu"

BRAIN_RESAMPLE_RATE = 120

AUDIO_RESAMPLE_RATE = 16000  # before wave2vec2.0
