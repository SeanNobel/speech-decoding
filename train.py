import os, sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from configs.args import args
from data.datasets import Gwilliams2022Dataset, Brennan2018Dataset, ToyDataset
from models.brain_encoder import BrainEncoder
from utils.loss import CLIPLoss, MSELoss, CLIPLossOrig, CLIPLossX
from utils.wav2vec_util import load_wav2vec_model
from tqdm import trange
from termcolor import cprint
import wandb
from constants import device

run_dir = f"runs/{args.name}/"
if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# NOTE: we'll remove this, once the repo is ready
if args.wandb:
    wandb.config = {
        k: v
        for k, v in args.__dict__.items() if not k.startswith('__')
    }
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
if args.dataset == 'Gwilliams2022':
    dataset = Gwilliams2022Dataset(args)
elif args.dataset == 'Brennan2018':
    # NOTE: now the DS take not the number of samples, but the seconds to make windows
    # NOTE: takes an optional debug param force_recompute to pre-process the EEG even if it exists
    dataset = Brennan2018Dataset(args)
else:
    raise ValueError('Unknown dataset')

train_size = int(dataset.X.shape[0] * 0.8)
test_size = dataset.X.shape[0] - train_size
train_set, test_set = torch.utils.data.random_split(
    dataset,
    lengths=[train_size, test_size],
    generator=torch.Generator().manual_seed(1234),
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True,
)

# ---------------
#      Loss
# ---------------
# loss_func = CLIPLoss("sum").cuda()
loss_func = CLIPLossX(device, args.batch_size, reduction="sum")
# loss_func = CLIPLossOrig("sum").cuda()
# loss_func = MSELoss().cuda()

for epoch in range(args.epochs):
    train_losses = []
    test_losses = []

    # weight_prev = brain_encoder.subject_block.spatial_attention.z_re.clone()

    brain_encoder.train()
    for i, (X, Y, subject_idxs) in enumerate(tqdm(train_loader)):

        X, Y = X.to(device), Y.to(device)

        Z = brain_encoder(X, subject_idxs)

        loss = loss_func(Y, Z)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # weight_after = brain_encoder.subject_block.spatial_attention.z_re.clone()
    # print(f"Learning: {not torch.equal(weight_prev, weight_after)}")

    # NOTE: maybe testing in this way is meaningless for contrastive loss
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
        f"Epoch {epoch}/{args.epochs} | avg train loss: {np.mean(train_losses):.3f} | avg test loss: {np.mean(test_losses):.3f} | lr: {optimizer.param_groups[0]['lr']:.5f}"
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

    torch.save(brain_encoder.state_dict(), run_dir + "model_last.pt")
