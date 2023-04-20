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


def zero_shot_classification(Z: torch.Tensor, Y: torch.Tensor, label: torch.Tensor, test=False, top_k=None)-> torch.Tensor:
    batch_size = Z.size(0)
    sample_size = Y.size(0)
    label = label -1 # labelは1始まり
    x = Z # .view(batch_size, -1) # 300 x 512
    y = Y # .view(batch_size, -1) # 50 x 512
    # import pdb; pdb.set_trace()
    # NOTE: avoid CUDA out of memory like this
    similarity = torch.empty(batch_size, sample_size).to(device)

    if test:
        pbar = tqdm(total=batch_size, desc="[Similarities]")

    for i in range(batch_size):
        for j in range(sample_size):
            similarity[i, j] = (x[i] @ y[j]) / max((x[i].norm() * y[j].norm()), 1e-8)

        if test:
            pbar.update(1)

    # similarity = similarity.T # brain x image -> image x brain

    # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
    # import pdb; pdb.set_trace()
    top1accuracy = (similarity.argmax(axis=1) == label).to(torch.float).cpu().numpy()

    try:
        top10accuracy = np.array(
            [
                label in row
                for row, label in zip(torch.topk(similarity, 10, dim=1, largest=True)[1], label)
            ]
        )
    except:
        print(similarity.size())
        raise
    if top_k is None:

        return top1accuracy, top10accuracy
    else:
        try:
            topkaccuracy = np.array(
                [
                    label in row
                    for row, label in zip(torch.topk(similarity, top_k, dim=1, largest=True)[1], label)
                ]
                )
        except:
            print(similarity.size())
            raise
        return top1accuracy, top10accuracy, topkaccuracy


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
        train_dataset = GODDatasetBase(args, 'train', return_label=True)
        val_dataset = GODDatasetBase(args, 'val', return_label=True)
        with open_dict(args):
            args.num_subjects = train_dataset.num_subjects
            print('num subject is {}'.format(args.num_subjects))


        test_size = val_dataset.Y.shape[0]
        if args.use_sampler:
            train_loader, test_loader = get_samplers(
                train_dataset,
                val_dataset,
                args,
                test_bsz=args.batch_size,
                collate_fn=GODCollator(args,  return_label=True),)

        else:
            test_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                drop_last=True,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )

    else:
        raise ValueError("Unknown dataset")
    # assert len(test_loader) == 1
    brain_encoder = get_model(args).to(device) #BrainEncoder(args).to(device)

    weight_dir = os.path.join(args.save_root, 'weights')
    last_weight_file = os.path.join(weight_dir, "model_last.pt")
    best_weight_file = os.path.join(weight_dir, "model_best.pt")
    if os.path.exists(best_weight_file):
        brain_encoder.load_state_dict(torch.load(best_weight_file))
        print('weight is loaded from ', best_weight_file)
    else:
        brain_encoder.load_state_dict(torch.load(last_weight_file))
        print('weight is loaded from ', last_weight_file)

    testTop1accs = []
    testTop10accs = []
    testTopKaccs = []
    classifier = Classifier(args)
    classifier.eval()
    brain_encoder.eval()

    sorted_image_features = np.load('./data/GOD/image_features.npy')
    Y = torch.Tensor(sorted_image_features).to(device)
    half_k = int(len(sorted_image_features)/2)
    for batch in tqdm(test_loader):

        with torch.no_grad():
            if len(batch) == 4:
                X, _, subject_idxs, label = batch
            else:
                raise ValueError("Unexpected number of items from dataloader.")

            X, label = X.to(device), label.to(device)

            Z = brain_encoder(X, subject_idxs)  # 0.96 GB

            testTop1acc, testTop10acc, testTopKacc = zero_shot_classification(Z, Y, label,test=True, top_k=half_k)  # ( 250, 1024, 360 )

        testTop1accs.append(testTop1acc)
        testTop10accs.append(testTop10acc)
        testTopKaccs.append(testTopKacc)

    testTop1accs = np.concatenate(testTop1accs)
    testTop10accs = np.concatenate(testTop10accs)
    testTopKaccs = np.concatenate(testTopKaccs)
    print(
            f"testTop1acc: {np.mean(testTop1accs):.3f} | "
            f"testTop10acc: {np.mean(testTop10accs):.3f} | "
            f"testTop{half_k}acc: {np.mean(testTopKaccs):.3f} | "
        )


def get_average_features(predicted_y, val_index):
    test_labels_unique = np.unique(val_index)
    test_pred_features_avg = []
    for i in range(len(test_labels_unique)):
        target_ids = val_index== i
        test_pred_features_avg.append(predicted_y[target_ids].mean(axis=0, keepdims=True))
    test_pred_features_avg = np.concatenate(test_pred_features_avg, axis=0)
    return test_pred_features_avg, np.arange(len(test_labels_unique))

def acc_category_identification(predicted_y, val_index, use_average=False):
    # predicted_y: num_trials x 512
    # val_index: num_trials
    val_index = val_index - 1 # ラベルは1始まり
    image_features = np.load('./data/GOD/image_features.npy') # 50 x 512
    if use_average:
        print('use average')
        predicted_y, val_index = get_average_features(predicted_y, val_index)
    num_images = len(image_features)
    num_trials = len(predicted_y)
    acc_tmp = np.zeros((num_trials, 1))

    cat_wise_acc = {i:[] for i in range(len(image_features))}
    # import pdb; pdb.set_trace()
    for i_pred in range(num_trials):
        space_corr = np.zeros((num_images, 1))
        # iterating over all images
        # calculating the correlation between the predicted and the image features
        for i_img in range(num_images):
            R = np.corrcoef(predicted_y[i_pred], image_features[i_img])
            space_corr[i_img] = R[0,1]

        # assigning the index of the current predicred vector to image_id
        image_id = val_index[i_pred]
        # calculating the accuracy of the cirrent predicted vector by counting the number of images with correlation coefficiens less than that of corresponding image
        # and dividing by the total number of images minus one

        acc_tmp[i_pred] = np.sum(space_corr < space_corr[image_id]) / (num_images - 1)
        cat_wise_acc[image_id].append(acc_tmp[i_pred])
    cat_wise_acc = {i: np.mean(cat_wise_acc[i]) for i in range(len(image_features))}
    return np.mean(acc_tmp), cat_wise_acc

def run_acc_from_corr(args, use_average=False):
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
        train_dataset = GODDatasetBase(args, 'train', return_label=True)
        val_dataset = GODDatasetBase(args, 'val', return_label=True)
        with open_dict(args):
            args.num_subjects = train_dataset.num_subjects
            print('num subject is {}'.format(args.num_subjects))


        test_size = val_dataset.Y.shape[0]
        if args.use_sampler:
            train_loader, test_loader = get_samplers(
                train_dataset,
                val_dataset,
                args,
                test_bsz=test_size, #args.batch_size,
                collate_fn=GODCollator(args,  return_label=True),)

        else:
            test_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                drop_last=True,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )

    else:
        raise ValueError("Unknown dataset")
    # assert len(test_loader) == 1
    brain_encoder = get_model(args).to(device) #BrainEncoder(args).to(device)

    weight_dir = os.path.join(args.save_root, 'weights')
    last_weight_file = os.path.join(weight_dir, "model_last.pt")
    best_weight_file = os.path.join(weight_dir, "model_best.pt")
    if os.path.exists(best_weight_file):
        brain_encoder.load_state_dict(torch.load(best_weight_file))
        print('weight is loaded from ', best_weight_file)
    else:
        brain_encoder.load_state_dict(torch.load(last_weight_file))
        print('weight is loaded from ', last_weight_file)

    pred_features = []
    labels = []
    for batch in tqdm(test_loader):

        with torch.no_grad():
            if len(batch) == 4:
                X, _, subject_idxs, label = batch
            else:
                raise ValueError("Unexpected number of items from dataloader.")

            X, label = X.to(device), label.to(device)

            Z = brain_encoder(X, subject_idxs)  # 0.96 GB

        pred_features.append(Z.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
    pred_features = np.concatenate(pred_features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print('total predictions: {}'.format(len(labels)))
    acc_from_corr, cat_wise_acc = acc_category_identification(pred_features, labels, use_average=use_average)
    print('acc_from_corr: {}'.format(acc_from_corr))
    print('category wise accuracy: ', cat_wise_acc)

if __name__ == "__main__":
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="../configs/"):
        args = compose(config_name='20230419_sbj01_seq2stat')
    if not os.path.exists(os.path.join(args.save_root, 'weights')):
        os.makedirs(os.path.join(args.save_root, 'weights'))
    # run(args)
    run_acc_from_corr(args, use_average=True)
