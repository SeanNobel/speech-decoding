import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm, trange
from termcolor import cprint
# import wandb
import pandas as pd
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
import matplotlib.pyplot as plt
import seaborn as sns


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
        source_dataset = GODDatasetBase(args, 'train', return_label=True)
        outlier_dataset = GODDatasetBase(args, 'val', return_label=True,
                                         mean_X= source_dataset.mean_X,
                                         mean_Y=source_dataset.mean_Y,
                                         std_X=source_dataset.std_X,
                                         std_Y=source_dataset.std_Y
                                        )
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
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            test_loader = DataLoader(
                # val_dataset, # 
                outlier_dataset,  # val_dataset
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

    weight_dir = os.path.join(args.save_root, 'weights')
    last_weight_file = os.path.join(weight_dir, "model_last.pt")
    best_weight_file = os.path.join(weight_dir, "model_best.pt")
    if os.path.exists(best_weight_file):
        brain_encoder.load_state_dict(torch.load(best_weight_file))
        print('weight is loaded from ', best_weight_file)
    else:
        brain_encoder.load_state_dict(torch.load(last_weight_file))
        print('weight is loaded from ', last_weight_file)


    classifier = Classifier(args)

    # ---------------
    #      Loss
    # ---------------
    loss_func = CLIPLoss(args).to(device)
    loss_func.eval()
    # ======================================
    train_losses = []
    test_losses = []
    trainTop1accs = []
    trainTop10accs = []
    testTop1accs = []
    testTop10accs = []
    brain_encoder.eval()
    pbar2 = tqdm(train_loader)
    for i, batch in enumerate(pbar2):
        with torch.no_grad():
            if len(batch) == 3:
                X, Y, subject_idxs = batch
            elif len(batch) == 4:
                X, Y, subject_idxs, chunkIDs = batch
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

    Zs = []
    Ys = []
    Ls = []
    brain_encoder.eval()
    for batch in test_loader:
        with torch.no_grad():

            if len(batch) == 3:
                X, Y, subject_idxs = batch
            elif len(batch) == 4:
                X, Y, subject_idxs, Labels = batch
            else:
                raise ValueError("Unexpected number of items from dataloader.")

            X, Y = X.to(device), Y.to(device)

            Z = brain_encoder(X, subject_idxs)  # 0.96 GB
            Zs.append(Z)
            Ys.append(Y)
            Ls.append(Labels)

            loss = loss_func(Y, Z)

            testTop1acc, testTop10acc = classifier(Z, Y, test=True)  # ( 250, 1024, 360 )

        test_losses.append(loss.item())
        testTop1accs.append(testTop1acc)
        testTop10accs.append(testTop10acc)
    Zs = torch.cat(Zs, dim=0)
    Ys = torch.cat(Ys, dim=0)
    Ls = torch.cat(Ls, dim=0).detach().cpu().numpy()

    print(
        f"train l: {np.mean(train_losses):.3f} | ",
        f"test l: {np.mean(test_losses):.3f} | ",
        f"trainTop10acc: {np.mean(trainTop10accs):.3f} | ",
        f"testTop10acc: {np.mean(testTop10accs):.3f} | ",
        f"temp: {loss_func.temp.item():.3f}",
    )


    # 仮説1:判定に偏りがある。-> あるサンプルのimageの特徴量がMEGの潜在空間ににているかどうかを判定するだけの基準になっているのではないか？
    Zs = Zs - Zs.mean(dim=0, keepdims=True)
    Zs = Zs / Zs.std(dim=0, keepdims=True)
    Zs = Zs - Zs.mean(dim=1, keepdims=True)
    Zs = Zs / Zs.std(dim=1, keepdims=True)

    acc, mat = evaluate(Zs, Ys)
    vis_confusion_mat(mat, acc, os.path.join(args.save_root, 'confusion_mat.png'))
    n_database_hits = mat.sum(axis=0)
    print('Num of hits of dataset \n', n_database_hits)

    miss_detection = np.sum(mat < 0, axis=0)/(len(mat)-1)  # FP
    print('Num miss detection: \n', miss_detection)

    true_detection = np.sum(mat > 0, axis=1) / (len(mat)-1) # TP
    print('Num query detection: \n', true_detection)


    N=1
    plot_array_label =  np.argsort(miss_detection)[::-1][:N]
    plot_array_value = np.sort(miss_detection)[::-1][:N]
    print('miss detection id', plot_array_label)
    print('miss detection value', plot_array_value)
    fig, axes = plt.subplots(nrows=2, figsize=(32,8))
    boxplot_and_plot(Zs.detach().cpu().numpy(), plot_array_label, axes[0])
    boxplot_and_plot(Ys.detach().cpu().numpy(), plot_array_label, axes[1])
    plt.savefig(os.path.join(args.save_root, 'boxplot_and_plot.png'),bbox_inches='tight')
    plt.close()

    N=1
    plot_array_label =  np.argsort(true_detection)[:N]
    plot_array_value = np.sort(true_detection)[:N]
    print(plot_array_label)
    print(plot_array_value)
    fig, axes = plt.subplots(nrows=2, figsize=(32,8))
    boxplot_and_plot(Zs.detach().cpu().numpy(), plot_array_label, axes[0])
    boxplot_and_plot(Ys.detach().cpu().numpy(), plot_array_label, axes[1])
    plt.savefig(os.path.join(args.save_root, 'boxplot_and_plot_weak.png'),bbox_inches='tight')
    plt.close()
    
    mask = np.tril(np.ones_like(mat), k=-1) > 0
    bias_detection = np.abs(mat - mat.T)
    biased_judge = np.sum((bias_detection==2) * mask)
    fair_judge = np.sum((bias_detection==0) * mask)
    print('num biased {} vs num fair judged {}'.format(biased_judge, fair_judge))
    
    Zs_std = Zs.std(dim=1)
    plt.scatter(Zs_std.detach().cpu().numpy(), true_detection)
    plt.xlabel('std of Z')
    plt.ylabel('TP ratio')
    plt.savefig(os.path.join(args.save_root, 'std_vs_tp.png'),bbox_inches='tight')

    similarity = calc_similarity(Zs, Ys)
    # output top5 similarity
    top5_similarity = {'query_image_id':[], 'acc(scene_id)':[], 
                       'top1_image_id':[], 'top2_image_id':[], 'top3_image_id':[], 'top4_image_id':[], 'top5_image_id':[]}
    acc_per_sample = np.round((mat>0).sum(axis=1)/49, decimals=3)
    for i, l in enumerate(Ls):
        sim_vec = similarity[i,:]
        top5_similarity['query_image_id'].append(l)
        top5_similarity['acc(scene_id)'].append(acc_per_sample[i])
        ranking = np.argsort(sim_vec)[::-1][:5] + 1 # 1始まりにする
        for k in range(1,6):
            key = f'top{k}_image_id'
            top5_similarity[key].append(ranking[k-1])
    top5_similarity = pd.DataFrame(top5_similarity)
    top5_similarity.to_csv(os.path.join(args.save_root, 'top5.csv'))
    # import pdb; pdb.set_trace()



def boxplot_and_plot(bp_array, plot_array_label, ax):
    # array: n_image x dims(512)
    plot_array = bp_array[plot_array_label]
    ax.boxplot(bp_array)
    for l, ar in zip(plot_array_label, plot_array):
        ax.plot(np.arange(len(ar)), ar, label=str(l))
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('unit id')
    ax.set_ylabel('logits')
    
def calc_similarity(x, y):
    batch_size = len(x)
    gt_size = len(y)

    similarity = torch.empty(batch_size, gt_size).to('cuda')
    for i in range(batch_size):
        for j in range(gt_size):
            similarity[i, j] = (x[i] @ y[j]) / max((x[i].norm() * y[j].norm()), 1e-8)
    return similarity.cpu().numpy()

def evaluate(Z, Y):
    # Z: (batch_size, 512)
    # Y: (gt_size, 512)
    binary_confusion_matrix = np.zeros([len(Z), len(Y)])
    similarity = calc_similarity(Z, Y)
    acc_tmp = np.zeros(len(similarity))
    for i in range(len(similarity)):
        acc_tmp[i] = np.sum(similarity[i,:] < similarity[i,i]) / (len(similarity)-1)
        binary_confusion_matrix[i,similarity[i,:] < similarity[i,i]] = 1 
        binary_confusion_matrix[i,similarity[i,:] > similarity[i,i]] = -1 
    similarity_acc = np.mean(acc_tmp)
    

    print('Similarity Acc', similarity_acc)
    
    return similarity_acc, binary_confusion_matrix

def vis_confusion_mat(mat, acc, savefile=None):
    sns.heatmap(mat, square=True, annot=False)
    plt.xlabel('database data')
    plt.ylabel('query data')
    plt.title('similarity acc: {}'.format(acc))
    plt.savefig(savefile)
    plt.close()
    print('saved to ', savefile)


if __name__ == "__main__":
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="../configs/"):
        # args = compose(config_name='20230429_sbj01_eegnet_regression')
        # args = compose(config_name='20230501_all_eegnet_regression')
        # args = compose(config_name='20230425_sbj01_seq2stat')
        args = compose(config_name='20230516_sbj03_eegnet_regression')
    if not os.path.exists(os.path.join(args.save_root, 'weights')):
        os.makedirs(os.path.join(args.save_root, 'weights'))
    run(args)
