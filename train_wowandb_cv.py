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
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

from meg_decoding.models import get_model, Classifier
from meg_decoding.utils.get_dataloaders import get_dataloaders, get_samplers
from meg_decoding.utils.loss import *
from meg_decoding.dataclass.god import GODDatasetBase, GODCollator
from meg_decoding.utils.loggers import Pickleogger
from meg_decoding.utils.vis_grad import get_grad
from torch.utils.data.dataset import Subset


def run(args: DictConfig) -> None:

    from meg_decoding.utils.reproducibility import seed_worker
    # NOTE: We do need it (IMHO).
    if args.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        g = torch.Generator()
        g.manual_seed(0)
        seed_worker = seed_worker
    else:
        g = None
        seed_worker = None

    pkl_logger = Pickleogger(os.path.join(args.save_root, 'runs'))

    # with open_dict(args):
    #     args.root_dir = get_original_cwd()
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
        source_dataset = GODDatasetBase(args, 'train')
        # val_dataset = GODDatasetBase(args, 'val')
        # train_size = int(np.round(len(source_dataset)*0.8))
        # val_size = len(source_dataset) - train_size

        # train_dataset, val_dataset = torch.utils.data.random_split(source_dataset, [train_size, val_size])
        ind_tr = list(range(0, 3000)) + list(range(3600, 6600)) #+ list(range(7200, 21600)) # + list(range(7200, 13200)) + list(range(14400, 20400))
        ind_te = list(range(3000,3600)) + list(range(6600, 7200)) # + list(range(13200, 14400)) + list(range(20400, 21600))
        train_dataset = Subset(source_dataset, ind_tr)
        val_dataset   = Subset(source_dataset, ind_te)

        with open_dict(args):
            args.num_subjects = source_dataset.num_subjects
            print('num subject is {}'.format(args.num_subjects))


        if args.use_sampler:
            test_size = 50# 重複サンプルが存在するのでval_dataset.Y.shape[0]
            train_loader, test_loader = get_samplers(
                train_dataset,
                val_dataset,
                args,
                test_bsz=test_size,
                collate_fn=GODCollator(args),)

        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size= args.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            test_loader = DataLoader(
                val_dataset,
                batch_size=50, # args.batch_size,
                drop_last=True,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )

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
    brain_encoder = get_model(args).to(device) #BrainEncoder(args).to(device)

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
    best_acc = 0
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        pbar.set_description("training {}/{} epoch".format(epoch, args.epochs))
        train_losses = []
        test_losses = []
        trainTop1accs = []
        trainTop10accs = []
        testTop1accs = []
        testTop10accs = []

        brain_encoder.train()
        pbar2 = tqdm(train_loader)
        for i, batch in enumerate(pbar2):

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
            # import pdb; pdb.set_trace()
            Z = brain_encoder(X, subject_idxs)
            loss = loss_func(Y, Z)
            with torch.no_grad():
                trainTop1acc, trainTop10acc = classifier(Z, Y)

            train_losses.append(loss.item())
            trainTop1accs.append(trainTop1acc)
            trainTop10accs.append(trainTop10acc)

            pbar.set_description("training {}/{} iters Train/Loss: {}, Train/Top1Acc: {}, Train/Top10Acc: {}".format(i, len(train_loader), loss.item(), trainTop1acc, trainTop10acc))
            if args.dataset == "Gwilliams2022":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if args.dataset == "GOD":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # get_grad(brain_encoder)
            # break

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
        pkl_logger.log({
                "epoch": epoch,
                "train_loss": np.mean(train_losses),
                "test_loss": np.mean(test_losses),
                "trainTop1acc": np.mean(trainTop1accs),
                "trainTop10acc": np.mean(trainTop10accs),
                "testTop1acc": np.mean(testTop1accs),
                "testTop10acc": np.mean(testTop10accs),
                "lrate": optimizer.param_groups[0]["lr"],
                "temp": loss_func.temp.item(),
            }, 'logs')

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

        savedir = os.path.join(args.save_root, 'weights')
        last_weight_file = os.path.join(savedir, "model_last.pt")
        torch.save(brain_encoder.state_dict(), last_weight_file)
        print('model is saved as ', last_weight_file)
        if best_acc < np.mean(testTop10accs):
            best_weight_file = os.path.join(savedir, "model_best.pt")
            torch.save(brain_encoder.state_dict(), best_weight_file)
            best_acc =  np.mean(testTop10accs)
            print('best model is updated !!, {}'.format(best_acc), best_weight_file)

if __name__ == "__main__":
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="../configs/"):
        args = compose(config_name='20230607_sbj03_eegnet')
    if not os.path.exists(os.path.join(args.save_root, 'weights')):
        os.makedirs(os.path.join(args.save_root, 'weights'))
    run(args)
