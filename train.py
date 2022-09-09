import os, sys
import sched
import numpy as np
import torch
import torch.nn as nn
import argparse
from time import time
from tqdm import tqdm
from data.datasets import Brennan2018Dataset
from models.brain_encoder import BrainEncoder
from utils.loss import CLIPLoss

assert torch.cuda.is_available(), "Training without GPU is not supported."

parser = argparse.ArgumentParser(description="Speech decoding by MetaAI reimplementation")
parser.add_argument("--batch-size", default=128)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--epochs", default=20)
parser.add_argument("--seq-len", type=int, default=256, help="T in the paper") # seq-len 256 is approximately 1.8 seconds in real world
parser.add_argument("--num-subjects", default=33)
parser.add_argument("--D1", type=int, default=270, help="D_1 in the paper")
parser.add_argument("--D2", type=int, default=320)
parser.add_argument("--F", type=int, default=512, help="Embedding dimension for both speech and M/EEG")
parser.add_argument("--K", type=int, default=32, help="Number of harmonics in fourier space for spatial attention")
parser.add_argument("--montage", type=str, default="easycap-M10", help="easycap-M10 is the one used in Brennan2018")
parser.add_argument("--wav2vec-model", type=str, default="xlsr_53_56k", help="Type of wav2vec2.0 model to use")
args = parser.parse_args()

# ---------------------
#        Model
# ---------------------
brain_encoder = BrainEncoder(args).cuda()
optimizer = torch.optim.Adam(brain_encoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# -----------------------
#       Dataloader
# -----------------------
train_loader = torch.utils.data.DataLoader(
    dataset=Brennan2018Dataset(args.seq_len, args.wav2vec_model),
    batch_size=args.batch_size,
    shuffle=True,
)
# TODO: test loader


# loss_func = CLIPLoss("sum").cuda()
loss_func = nn.MSELoss(reduction="sum").cuda()


brain_encoder.train()
for epoch in range(args.epochs):
    losses = []

    # weight_prev = brain_encoder.subject_block.spatial_attention.z_re.clone()

    for i, (X, Y, subject_idxs) in enumerate(tqdm(train_loader)):
        brain_encoder.zero_grad()

        X, Y = X.cuda(), Y.cuda()

        Z = brain_encoder(X, subject_idxs)

        loss = loss_func(Y, Z)
        losses.append(loss.item())
        print(loss.item())

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} avg loss: {np.mean(losses)} lr: {optimizer.param_groups[0]['lr']}")

    # weight_after = brain_encoder.subject_block.spatial_attention.z_re.clone()
    # print(f"Learning: {not torch.equal(weight_prev, weight_after)}")

    scheduler.step()