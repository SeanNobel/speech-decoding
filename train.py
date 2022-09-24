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

assert torch.cuda.is_available(), "Training without GPU is not supported."

run_dir = f"runs/{args.name}/"
if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# ---------------------
#        Models
# ---------------------
brain_encoder = BrainEncoder(args).cuda()
optimizer = torch.optim.Adam(brain_encoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

wav2vec = load_wav2vec_model(args.wav2vec_model).cuda()
wav2vec.eval()

# -----------------------
#       Dataloader
# -----------------------
dataset = Gwilliams2022Dataset(args.wav2vec_model, shift_brain=True)
# dataset = Brennan2018Dataset(args.seq_len, args.wav2vec_model)

train_size = int(dataset.X.shape[0] * 0.8)
test_size = dataset.X.shape[0] - train_size
train_set, test_set = torch.utils.data.random_split(
    dataset,
    lengths=[train_size, test_size],
    generator=torch.Generator().manual_seed(1234)
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=args.batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=args.batch_size,
    shuffle=False,
)

# ---------------
#      Loss
# ---------------
# loss_func = CLIPLoss("sum").cuda()
loss_func = CLIPLossX(args.batch_size, reduction="sum")
# loss_func = CLIPLossOrig("sum").cuda()
# loss_func = MSELoss().cuda()


for epoch in range(args.epochs):
    train_losses = []
    test_losses = []

    # weight_prev = brain_encoder.subject_block.spatial_attention.z_re.clone()

    brain_encoder.train()
    for i, (X, Y, subject_idxs) in enumerate(tqdm(train_loader)):

        X, Y = X.cuda(), Y.cuda()

        with torch.no_grad():
            Y = wav2vec.feature_extractor(Y)

        Z = brain_encoder(X, subject_idxs)

        loss = loss_func(Y, Z)
        train_losses.append(loss.item())

        brain_encoder.zero_grad()
        loss.backward()
        optimizer.step()

    # weight_after = brain_encoder.subject_block.spatial_attention.z_re.clone()
    # print(f"Learning: {not torch.equal(weight_prev, weight_after)}")


    # NOTE: maybe testing in this way is meaningless for contrastive loss
    brain_encoder.eval()
    for i, (X, Y, subject_idxs) in enumerate(tqdm(test_loader)):

        X, Y = X.cuda(), Y.cuda()

        with torch.no_grad():
            Y = wav2vec.feature_extractor(Y)

            Z = brain_encoder(X, subject_idxs)

        loss = loss_func(Y, Z)
        test_losses.append(loss.item())

    print(
        f"Epoch {epoch}/{args.epochs} | avg train loss: {np.mean(train_losses):.3f} | avg test loss: {np.mean(test_losses):.3f} | lr: {optimizer.param_groups[0]['lr']}"
    )

    scheduler.step()

    torch.save(brain_encoder.state_dict(), run_dir+"model_last.pt")