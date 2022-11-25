from constants import bar_format, device
from configs.args import args
if args.reproducible:
    from utils.reproducibility import g, seed_worker
else:
    g = None
    seed_worker = None
import os, sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from data.brennan2018 import Brennan2018Dataset
# from data.gwilliams2022 import Gwilliams2022Dataset
from data.gwilliams2022_multicore import Gwilliams2022Dataset

from models.brain_encoder import BrainEncoder
from models.classifier import Classifier
from utils.get_dataloaders import get_dataloaders, get_samplers
from utils.loss import *
from tqdm import trange
from termcolor import cprint
import wandb

run_dir = f"runs/{args.name}/"
if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# NOTE: we'll remove this, once the repo is ready
if args.wandb:
    wandb.config = {k: v for k, v in args.__dict__.items() if not k.startswith('__')}
    wandb.init(
        project="speech_decoding",
        # entity="nightdude",
        config=wandb.config,
        save_code=True,
    )

# -----------------------
#       Dataloader
# -----------------------
# NOTE: For Gwilliams dataset, dataset size is the number of speech segments
# so that no overlapping segments are included in a single batch
if args.dataset == 'Gwilliams2022':
    dataset = Gwilliams2022Dataset(args)

    args.num_subjects = dataset.num_subjects

    train_size = int(dataset.Y.shape[0] * 0.8)
    test_size = dataset.Y.shape[0] - train_size
    train_set, test_set = torch.utils.data.random_split(
        dataset,
        lengths=[train_size, test_size],
        generator=g if args.reproducible else None,
    )

    if args.use_sampler:
        # NOTE: currently not supporting reproducibility
        train_loader, test_loader = get_samplers(
            train_set,
            test_set,
            args,
        )
    else:
        if args.reproducible:
            train_loader, test_loader = get_dataloaders(
                train_set,
                test_set,
                args,
                seed_worker,
                g,
            )
        else:
            train_loader, test_loader = get_dataloaders(
                train_set,
                test_set,
                args,
            )

elif args.dataset == 'Brennan2018':
    # NOTE: now the DS take not the number of samples, but the seconds to make windows
    # NOTE: takes an optional debug param force_recompute to pre-process the EEG even if it exists
    # dataset = Brennan2018Dataset(args)
    train_set = Brennan2018Dataset(args, train=True)
    test_set = Brennan2018Dataset(args, train=False)

    # train_size = int(dataset.X.shape[0] * 0.8)
    # test_size = dataset.X.shape[0] - train_size
    # train_set, test_set = torch.utils.data.random_split(
    #     dataset,
    #     lengths=[train_size, test_size],
    #     generator=g if args.reproducible else None,
    # )

    train_loader, test_loader = get_dataloaders(train_set, test_set, args, g, seed_worker)

else:
    raise ValueError('Unknown dataset')

# ---------------------
#        Models
# ---------------------
brain_encoder = BrainEncoder(args).to(device)

# classifier
classifier = Classifier(args)

# ---------------
#      Loss
# ---------------
loss_func = CLIPLoss(args).to(device)
loss_func.train()

# --------------------
#      Optimizer
# --------------------
optimizer = torch.optim.Adam(
    list(brain_encoder.parameters()) + list(loss_func.parameters()),
    lr=float(args.lr),
)

if args.lr_scheduler == "exponential":
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_exp_gamma)
elif args.lr_scheduler == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=args.epochs // args.lr_step_numsteps,
                                                gamma=args.lr_step_gamma)
elif args.lr_scheduler == "multistep":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(m * args.epochs) for m in args.lr_multistep_mlstns], gamma=args.lr_step_gamma)
else:
    raise ValueError()

# ======================================
for epoch in range(args.epochs):
    train_losses = []
    test_losses = []
    trainTop1accs = []
    trainTop10accs = []
    testTop1accs = []
    testTop10accs = []

    # weight_prev = brain_encoder.subject_block.spatial_attention.z_re.clone()

    brain_encoder.train()
    for i, batch in enumerate(tqdm(train_loader)):
        X = batch[0]
        Y = batch[1]
        subject_idxs = batch[2]
        if not isinstance(train_loader.dataset.dataset, Gwilliams2022Dataset):
            chunkIDs = batch[3]

        X, Y = X.to(device), Y.to(device)
        # print([(s.item(), chid.item()) for s, chid in zip(subject_idxs, chunkIDs)])
        Z = brain_encoder(X, subject_idxs)

        loss = loss_func(Y, Z)

        with torch.no_grad():
            trainTop1acc, trainTop10acc = classifier(Z, Y)

        train_losses.append(loss.item())
        trainTop1accs.append(trainTop1acc)
        trainTop10accs.append(trainTop10acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # weight_after = brain_encoder.subject_block.spatial_attention.z_re.clone()
    # print(f"Learning: {not torch.equal(weight_prev, weight_after)}")

    # NOTE: maybe testing in this way is meaningless for contrastive loss
    brain_encoder.eval()
    for batch in test_loader:
        X = batch[0]
        Y = batch[1]
        subject_idxs = batch[2]
        if not isinstance(test_loader.dataset.dataset, Gwilliams2022Dataset):
            chunkIDs = batch[3]

        X, Y = X.to(device), Y.to(device)

        with torch.no_grad():
            Z = brain_encoder(X, subject_idxs)

        loss = loss_func(Y, Z)

        with torch.no_grad():
            testTop1acc, testTop10acc = classifier(Z, Y)

        test_losses.append(loss.item())
        testTop1accs.append(testTop1acc)
        testTop10accs.append(testTop10acc)

    print(
        f"Ep {epoch}/{args.epochs} | ",
        f"train l: {np.mean(train_losses):.3f} | ",
        f"test l: {np.mean(test_losses):.3f} | ",
        f"trainTop10acc: {np.mean(trainTop10accs):.3f} | ",
        f"testTop10acc: {np.mean(testTop10accs):.3f} | ",
        f"lr: {optimizer.param_groups[0]['lr']:.5f}",
        f"temp: {loss_func.temp.item():.3f}",
    )

    if args.wandb:
        performance_now = {
            'epoch': epoch,
            'train_loss': np.mean(train_losses),
            'test_loss': np.mean(test_losses),
            'trainTop1acc': np.mean(trainTop1accs),
            'trainTop10acc': np.mean(trainTop10accs),
            'testTop1acc': np.mean(testTop1accs),
            'testTop10acc': np.mean(testTop10accs),
            'lrate': optimizer.param_groups[0]['lr'],
            'temp': loss_func.temp.item()
        }
        wandb.log(performance_now)

    scheduler.step()

    torch.save(brain_encoder.state_dict(), run_dir + "model_last.pt")
