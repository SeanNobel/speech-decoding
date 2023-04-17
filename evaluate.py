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


def run(args: DictConfig) -> None:
    from meg_decoding.utils.reproducibility import seed_worker
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

    if args.dataset == "GOD":
        train_dataset = GODDatasetBase(args, 'train')
        val_dataset = GODDatasetBase(args, 'val')
        with open_dict(args):
            args.num_subjects = train_dataset.num_subjects
            print('num subject is {}'.format(args.num_subjects))


        test_size = val_dataset.Y.shape[0]
        if args.use_sampler:
            train_loader, test_loader = get_samplers(
                train_dataset,
                val_dataset,
                args,
                test_bsz=test_size,
                collate_fn=GODCollator(args),)

        else:
            test_loader = DataLoader(
                val_dataset,
                batch_size=test_size,
                drop_last=True,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )

    else:
        raise ValueError("Unknown dataset")
    assert len(test_loader) == 1
    brain_encoder = get_model(args).to(device) #BrainEncoder(args).to(device)

    weight_dir = os.path.join(args.save_root, 'weights')
    last_weight_file = os.path.join(weight_dir, "model_last.pt")
    brain_encoder.load_state_dict(torch.load(last_weight_file))
    print('weight is loaded from ', last_weight_file)

    testTop1accs = []
    testTop10accs = []
    testTopKaccs = []
    classifier = Classifier(args)
    classifier.eval()
    brain_encoder.eval()
    half_k = int(test_size/2)
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

            testTop1acc, testTop10acc, testTopKacc = classifier(Z, Y, test=True, topk=half_k)  # ( 250, 1024, 360 )

        testTop1accs.append(testTop1acc)
        testTop10accs.append(testTop10acc)
        testTopKaccs.append(testTopKacc)

    print(
            f"testTop1acc: {np.mean(testTop1accs):.3f} | "
            f"testTop10acc: {np.mean(testTop10accs):.3f} | "
            f"testTop{half_k}acc: {np.mean(testTopKaccs):.3f} | "
        )



if __name__ == "__main__":
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="../configs/"):
        args = compose(config_name='20230414_sbj01_seq2stat')
    if not os.path.exists(os.path.join(args.save_root, 'weights')):
        os.makedirs(os.path.join(args.save_root, 'weights'))
    run(args)
