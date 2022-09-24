import os, sys
import sched
import numpy as np
import torch
import torch.nn as nn
import argparse
from time import time
from tqdm import tqdm
from data.datasets import Gwilliams2022Dataset, Brennan2018Dataset, ToyDataset
from models.brain_encoder import BrainEncoder
from utils.loss import CLIPLoss, MSELoss, CLIPLossOrig, CLIPLossX
from utils.wav2vec_util import load_wav2vec_model
from tqdm import trange
from termcolor import cprint
import wandb

# assert torch.cuda.is_available(), "Training without GPU is not supported."
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="Speech decoding by MetaAI reimplementation")
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--batch-size", default=64)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--epochs", type=int, default=100)
# parser.add_argument("--seq-len", type=int, default=256, help="T in the paper") # seq-len 256 is approximately 1.8 seconds in real world
parser.add_argument("--num-subjects", default=27)
parser.add_argument("--D1", type=int, default=270, help="D_1 in the paper")
parser.add_argument("--D2", type=int, default=320)
parser.add_argument("--F", type=int, default=512, help="Embedding dimension for both speech and M/EEG")
parser.add_argument("--K", type=int, default=32, help="Number of harmonics in fourier space for spatial attention")
parser.add_argument("--dataset", type=str, default="Brennan2018", choices=["Gwilliams2022", "Brennan2018", "Toy"])
parser.add_argument("--wav2vec-model", type=str, default="xlsr_53_56k", help="Type of wav2vec2.0 model to use")
parser.add_argument("--wandb", type=bool, default=False, help="If you want to log progress to W&B")

parser.add_argument(
    "--seq_len",
    type=int,
    default=140,
    help="length (i.e. bptt, or number of embeddings in a sequence). 140 corresponds to ~1 s",
)
args = parser.parse_args()

# NOTE: we'll remove this, once the repo is ready
if args.wandb:
    wandb.config = {k: v for k, v in args.__dict__.items() if not k.startswith('__')}
    wandb.init(
        project="speech_decoding",
        entity="nightdude",
        config=wandb.config,
        save_code=True,
    )

# ---------------------
#        Models
# ---------------------
brain_encoder = BrainEncoder(args).to(device)
optimizer = torch.optim.Adam(brain_encoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

wav2vec = load_wav2vec_model(args.wav2vec_model).to(device)
wav2vec.eval()

# -----------------------
#       Dataloader
# -----------------------
# dataset = Gwilliams2022Dataset(args.wav2vec_model)
dataset = Brennan2018Dataset(args.seq_len, args.wav2vec_model)

train_size = int(dataset.X.shape[0] * 0.8)
test_size = dataset.X.shape[0] - train_size
train_set, test_set = torch.utils.data.random_split(dataset, lengths=[train_size, test_size])

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

# loss_func = CLIPLoss("sum").to(device)
loss_func = CLIPLossX(device)
# loss_func = CLIPLossOrig("sum").to(device)
# loss_func = MSELoss().to(device)

for epoch in range(args.epochs):
    train_losses = []
    test_losses = []

    # weight_prev = brain_encoder.subject_block.spatial_attention.z_re.clone()

    brain_encoder.train()
    for i, (X, Y, subject_idxs) in enumerate(tqdm(train_loader)):

        X, Y = X.to(device), Y.to(device)

        # NOTE: we don't need this for Brennan, we have precomputed W2V2 embeddings
        # with torch.no_grad():
        #     Y = wav2vec.feature_extractor(Y)

        Z = brain_encoder(X, subject_idxs)

        loss = loss_func(Y, Z)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # weight_after = brain_encoder.subject_block.spatial_attention.z_re.clone()
    # print(f"Learning: {not torch.equal(weight_prev, weight_after)}")

    brain_encoder.eval()
    for X, Y, subject_idxs in test_loader:

        X, Y = X.to(device), Y.to(device)

        with torch.no_grad():
            # NOTE: we don't need it. We already have precomputed W2V2 embeddings.
            # Y = wav2vec.feature_extractor(Y)
            Z = brain_encoder(X, subject_idxs)

        loss = loss_func(Y, Z)
        test_losses.append(loss.item())

    print(
        f"Epoch {epoch} | avg train loss: {np.mean(train_losses):.3f} | avg test loss: {np.mean(test_losses):.3f} | lr: {optimizer.param_groups[0]['lr']:.3f}"
    )

    if args.wandb:
        performance_now = {
            'epoch': epoch,
            'train_loss': np.mean(train_losses),
            'test_loss': np.mean(test_losses),
            'lrate': optimizer.param_groups[0]['lr'],
        }
        wandb.log(performance_now)

    scheduler.step()

    torch.save(brain_encoder.state_dict(), f"weights/brain_encoder/{args.name}.pt")