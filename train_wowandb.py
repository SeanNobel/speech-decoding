import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm, trange
from termcolor import cprint
# import wandb

from omegaconf import DictConfig, open_dict
import hydra
from hydra.utils import get_original_cwd

from constants import device
# from speech_decoding.dataclass.brennan2018 import Brennan2018Dataset
# from speech_decoding.dataclass.gwilliams2022 import (
#     Gwilliams2022SentenceSplit,
#     Gwilliams2022ShallowSplit,
#     Gwilliams2022DeepSplit,
#     Gwilliams2022Collator,
# )
from speech_decoding.models import BrainEncoder, Classifier
from speech_decoding.utils.get_dataloaders import get_dataloaders, get_samplers
from speech_decoding.utils.loss import *
from speech_decoding.utils.reproducibility import seed_worker


@hydra.main(version_base=None, config_path="configs", config_name="config_GOD")
def run(args: DictConfig) -> None:

    # NOTE: We do need it (IMHO).
    if args.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        g = torch.Generator()
        g.manual_seed(0)
    else:
        g = None
        seed_worker = None

    with open_dict(args):
        args.root_dir = get_original_cwd()
    cprint(f"Current working directory : {os.getcwd()}")
    cprint(args, color="white")

    # -----------------------
    #       Dataloader
    # -----------------------
    # NOTE: Segmentation should always be by word onsets, not just every 3 seconds
    if args.dataset == "Gwilliams2022":

        if args.split_mode == "sentence":

            train_set = Gwilliams2022SentenceSplit(args)
            test_set = Gwilliams2022SentenceSplit(args, train_set.test_word_idxs_dict)

            assert train_set.num_subjects == test_set.num_subjects
            with open_dict(args):
                args.num_subjects = train_set.num_subjects

            test_size = test_set.Y.shape[0]

        elif args.split_mode == "shallow":

            dataset = Gwilliams2022ShallowSplit(args)

            with open_dict(args):
                args.num_subjects = dataset.num_subjects

            train_size = int(dataset.Y.shape[0] * args.split_ratio)
            test_size = dataset.Y.shape[0] - train_size
            train_set, test_set = torch.utils.data.random_split(
                dataset, lengths=[train_size, test_size], generator=g,
            )

        elif args.split_mode == "deep":

            train_set = Gwilliams2022DeepSplit(args, train=True)
            test_set = Gwilliams2022DeepSplit(args, train=False)
            assert train_set.num_subjects == test_set.num_subjects

            with open_dict(args):
                args.num_subjects = train_set.num_subjects

            test_size = test_set.Y.shape[0]

        cprint(f"Test segments: {test_size}", "cyan")

        if args.use_sampler:
            # NOTE: currently not supporting reproducibility
            train_loader, test_loader = get_samplers(
                train_set,
                test_set,
                args,
                test_bsz=test_size,
                collate_fn=Gwilliams2022Collator(args),
            )
        else:
            # FIXME: maybe either get rid of reproducibility, or remove this?
            if args.reproducible:
                train_loader, test_loader = get_dataloaders(
                    train_set, test_set, args, seed_worker, g, test_bsz=test_size
                )
            else:
                train_loader, test_loader = get_dataloaders(
                    train_set, test_set, args, test_bsz=test_size
                )

    elif args.dataset == "Brennan2018":
        # NOTE: takes an optional debug param force_recompute to pre-process the EEG even if it exists
        dataset = Brennan2018Dataset(args)
        with open_dict(args):
            args.num_subjects = dataset.num_subjects

        train_size = int(len(dataset) * args.split_ratio)
        test_size = len(dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(
            dataset, lengths=[train_size, test_size], generator=g,
        )
        cprint(
            f"Number of samples: {len(train_set)} (train), {len(test_set)} (test)", color="blue",
        )
        train_loader, test_loader = get_dataloaders(
            train_set, test_set, args, g, seed_worker, test_bsz=test_size
        )

    elif args.dataset == "GOD":
        train_loader, test_loader ='', ''
    else:

        raise ValueError("Unknown dataset")

    if args.use_wandb:
        wandb.config = {k: v for k, v in dict(args).items() if k not in ["root_dir", "wandb"]}
        wandb.init(
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=wandb.config,
            save_code=True,
        )
        wandb.run.name = args.wandb.run_name + "_" + args.split_mode
        wandb.run.save()

    # ---------------------
    #        Models
    # ---------------------
    brain_encoder = BrainEncoder(args).to(device)

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
        list(brain_encoder.parameters()) + list(loss_func.parameters()), lr=float(args.lr),
    )

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
        )
    elif args.lr_scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(m * args.epochs) for m in args.lr_multistep_mlstns],
            gamma=args.lr_step_gamma,
        )
    else:
        scheduler = None

    # ======================================
    for epoch in range(args.epochs):
        train_losses = []
        test_losses = []
        trainTop1accs = []
        trainTop10accs = []
        testTop1accs = []
        testTop10accs = []

        brain_encoder.train()
        for i, batch in enumerate(tqdm(train_loader)):

            if len(batch) == 3:
                X, Y, subject_idxs = batch
            elif len(batch) == 4:
                X, Y, subject_idxs, chunkIDs = batch
                assert (
                    len(chunkIDs.unique()) == X.shape[0]
                ), "Duplicate segments in batch are not allowed. Aborting."
            else:
                raise ValueError("Unexpected number of items from dataloader.")

            X, Y = X.to(device), Y.to(device)
            # print([(s.item(), chid.item()) for s, chid in zip(subject_idxs, chunkIDs)])
            Z = brain_encoder(X, subject_idxs)

            loss = loss_func(Y, Z)

            with torch.no_grad():
                trainTop1acc, trainTop10acc = classifier(Z, Y)

            train_losses.append(loss.item())
            trainTop1accs.append(trainTop1acc)
            trainTop10accs.append(trainTop10acc)

            if args.dataset == "Gwilliams2022":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Accumulate gradients for Gwilliams for the whole epoch
        if args.dataset == "Brennan2018":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        brain_encoder.eval()
        for batch in test_loader:

            with torch.no_grad():

                if len(batch) == 3:
                    X, Y, subject_idxs = batch
                elif len(batch) == 4:
                    X, Y, subject_idxs, chunkIDs = batch
                else:
                    raise ValueError("Unexpected number of items from dataloader.")

                X, Y = X.to(device), Y.to(device)

                Z = brain_encoder(X, subject_idxs)  # 0.96 GB

                loss = loss_func(Y, Z)

                testTop1acc, testTop10acc = classifier(Z, Y, test=True)  # ( 250, 1024, 360 )

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

        if args.use_wandb:
            performance_now = {
                "epoch": epoch,
                "train_loss": np.mean(train_losses),
                "test_loss": np.mean(test_losses),
                "trainTop1acc": np.mean(trainTop1accs),
                "trainTop10acc": np.mean(trainTop10accs),
                "testTop1acc": np.mean(testTop1accs),
                "testTop10acc": np.mean(testTop10accs),
                "lrate": optimizer.param_groups[0]["lr"],
                "temp": loss_func.temp.item(),
            }
            wandb.log(performance_now)

        if scheduler is not None:
            scheduler.step()

        torch.save(brain_encoder.state_dict(), "model_last.pt")


if __name__ == "__main__":
    run()
