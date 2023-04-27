'''Generic Object Decoding: Feature prediction
Analysis summary
----------------
- Learning method:   Sparse linear regression
- Preprocessing:     Normalization and voxel selection
- Data:              GenericDecoding_demo
- Results format:    Pandas dataframe
'''


from __future__ import print_function

import os
import sys
sys.path.append('.')
import pickle
from itertools import product
from time import time
import itertools
import numpy as np
from scipy import stats

from meg_decoding.kamitani_lab.slir import SparseLinearRegression
from sklearn.linear_model import LinearRegression  # For quick demo

from meg_decoding.kamitani_lab.ml import add_bias
from meg_decoding.kamitani_lab.preproc import select_top
from meg_decoding.kamitani_lab.stats import corrcoef, corrmat

from meg_decoding.matlab_utils.load_meg import get_meg_data, roi, time_window, get_baseline
import tqdm
import random
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def prepare_dataset(args, split, manual_ch=None, onsets:dict=None):
    DATAROOT = args.data_root
    processed_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')
    label_path_pattern = os.path.join(DATAROOT, '{sub}/labels/{name}')
    trigger_meg_path_pattern = os.path.join(DATAROOT, '{sub}/trigger/{name}')
    processed_rest_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')

    sub_list = list(args.subjects.keys())
    sub_id_map = {s:i for i, s in enumerate(sub_list)}

    roi_channels = roi(args) if manual_ch is None else manual_ch # if onsets is specified. this setting is ignored.

    def epoching(meg, window, preprocess_pipeline=[]):
        assert len(labels) == len(window)
        # n_epochs x n_chs x time_smaples
        meg_epochs = np.zeros([len(window), len(meg), window[0][1]-window[0][0]])
        for i, w in enumerate(window):
            window_meg = meg[:, w[0]:w[1]] # ch x time_samples
            for func_ in preprocess_pipeline:
                window_meg = func_(window_meg)
            meg_epochs[i] = window_meg
        return meg_epochs

    meg_epochs = []
    sub_epochs = []
    label_epochs = []
    image_feature_epochs = []
    rest_mean, rest_std = None, None
    pbar = tqdm.tqdm(sub_list)
    for sub in pbar:
        pbar.set_description("load subject data -- current: {}".format(sub))
        fs = args.subjects[sub]['fs']
        for meg_name, label_name, trigger_name, rest_name in zip(args.subjects[sub][split]['mat'], args.subjects[sub][split]['labels'], args.subjects[sub][split]['trigger'], args.subjects[sub][split]['rest']):
            processed_meg_path = processed_meg_path_pattern.format(sub=sub, name=meg_name)
            label_path = label_path_pattern.format(sub=sub, name=label_name)
            trigger_path = trigger_meg_path_pattern.format(sub=sub, name=trigger_name)
            processed_rest_meg_path = processed_rest_meg_path_pattern.format(sub=sub, name=rest_name)
            if args.z_scoring:
                print('z_scoring start')
                rest_mean, rest_std = get_baseline(processed_rest_meg_path, fs, args.rest_duration)
            MEG_Data, image_features, labels, triggers = get_meg_data(processed_meg_path, label_path, trigger_path, rest_mean=rest_mean, rest_std=rest_std, split=split)
            if onsets is None:
                ROI_MEG_Data = MEG_Data[roi_channels, :] #  num_roi_channels x time_samples
                window = time_window(args, triggers, fs)
                ROI_MEG_epochs = epoching(ROI_MEG_Data, window)
            else:
                ROI_MEG_epochs = []
                for r, o in onsets.items():
                    args.region = r if isinstance(r, list) else [r]
                    roi_channels = roi(args)
                    ROI_MEG_Data = MEG_Data[roi_channels, :]
                    duration = args.window.end - args.window.start
                    args.window.start = o
                    args.window.end = o + duration
                    window = time_window(args, triggers, fs)
                    single_ROI_MEG_epoch = epoching(ROI_MEG_Data, window) # n_epochs x n_chs x time_smaples
                    # print('DEBUG: ', window[-1], ROI_MEG_Data.shape)
                    ROI_MEG_epochs.append(single_ROI_MEG_epoch)
                ROI_MEG_epochs = np.concatenate(ROI_MEG_epochs, axis=1)

            meg_epochs.append(ROI_MEG_epochs) # array [epoch x ch x time_stamp]
            sub_epochs+=[sub_id_map[sub]] * len(ROI_MEG_epochs) # list [epoch]
            label_epochs.append(labels) # array [epoch]
            image_feature_epochs.append(image_features) # array [epoch x dim]
    meg_epochs = np.concatenate(meg_epochs, axis=0)
    label_epochs = np.concatenate(label_epochs, axis=0)
    image_feature_epochs = np.concatenate(image_feature_epochs, axis=0)
    print('dataset is created. size is ', meg_epochs.shape)
    return meg_epochs, sub_epochs, label_epochs, image_feature_epochs


def get_average_features(predicted_y, val_index):
    test_labels_unique = np.unique(val_index)
    test_pred_features_avg = []
    for i in range(len(test_labels_unique)):
        target_ids = val_index== i
        test_pred_features_avg.append(predicted_y[target_ids].mean(axis=0, keepdims=True))
    test_pred_features_avg = np.concatenate(test_pred_features_avg, axis=0)
    return test_pred_features_avg, np.arange(len(test_labels_unique))

def metric_kamitani(results:dict, label:list):
    pred_percept = [results['predicted_feature_averaged_percept']] # n_preds x n_units
    cat_feature_percept = [results['category_feature_averaged_percept']] # n_sample x n_units

    pwident_cr_pt = []  # Prop correct in pair-wise identification (perception)
    # cnt = 0
    for fpt, pred_pt in zip(cat_feature_percept, pred_percept):
        # ind_cat_other = list(range(len(cat_feature_percept))).remove(cnt)
        # feat_other = cat_feature_percept[ind_cat_other,:]

        # n_unit = fpt.shape[1]
        # feat_other = feat_other[:, :n_unit]

        feat_candidate_pt = fpt # np.vstack([fpt, feat_other])

        simmat_pt = corrmat(pred_pt, feat_candidate_pt)

        cr_pt = get_pwident_correctrate(simmat_pt)

        pwident_cr_pt.append(np.mean(cr_pt))
        # cnt += 1
    assert len(pwident_cr_pt) == 1
    return np.mean(cr_pt), {k:v for k, v in zip(label, cr_pt)}


# Functions ############################################################

def get_pwident_correctrate(simmat):
    '''
    Returns correct rate in pairwise identification
    Parameters
    ----------
    simmat : numpy array [num_prediction * num_category]
        Similarity matrix
    Returns
    -------
    correct_rate : correct rate of pair-wise identification
    '''

    num_pred = simmat.shape[0]
    labels = range(num_pred)

    correct_rate = []
    for i in range(num_pred):
        pred_feat = simmat[i, :]
        correct_feat = pred_feat[labels[i]]
        pred_num = len(pred_feat) - 1
        correct_rate.append((pred_num - np.sum(pred_feat > correct_feat)) / float(pred_num))

    return correct_rate

def corr_of_corr(corrmat1, corrmat2):
    '''
    Returns correlation of correlation matrices
    Parameters
    ----------
    corrmat1 : numpy array [num_category * num_category]
        Correlation matrix 1
    corrmat2 : numpy array [num_category * num_category]
        Correlation matrix 2
    Returns
    -------
    corr_of_corr : correlation of correlation matrices
    '''

    corr_of_corr = np.corrcoef(corrmat1.flatten(), corrmat2.flatten())[0, 1]

    return corr_of_corr

def calc_corr_of_corr(meg_data:np.ndarray, image_data:np.ndarray, savedir:str)->np.ndarray:
    """_summary_

    Args:
        meg_data (np.ndarray): n_epochs x n_ch x time_samples
        image_data (np.ndarray): n_epochs x n_dims

    Returns:
        np.ndarray: _description_
    """
    meg_corr = np.corrcoef(meg_data.reshape(meg_data.shape[0], -1))
    vis_corr(meg_corr, os.path.join(savedir, 'meg_corr.png'))
    image_corr = np.corrcoef(image_data)
    vis_corr(image_corr, os.path.join(savedir, 'image_corr.png'))
    corr_of_corr = corr_of_corr(meg_corr, image_corr)
    vis_corr(corr_of_corr, os.path.join(savedir, 'corr_of_corr.png'))

def vis_corr(corr, savefile=None):
    sns.heatmap(corr, square=True, annot=False)
    plt.savefig(savefile)
    plt.close()
    print('saved to ', savefile)

def run_meg_fit_and_evaluate(args, ch_ratios=1, manual_ch:list=None, onsets:dict=None):
    prefix='sbj01'
    random.seed(0)
    ## prepare dataset
    train_X, train_subs, train_label, train_Y = prepare_dataset(args, split='train', manual_ch=manual_ch, onsets=onsets)
    test_X, test_subs, test_label, test_Y = prepare_dataset(args, split='val', manual_ch=manual_ch, onsets=onsets)
    # import pdb; pdb.set_trace()

    s1_d1_train_X, s1_d1_train_label, s1_d1_train_Y = train_X[:3600], train_label[:3600], train_Y[:3600]
    s1_d2_train_X, s1_d2_train_label, s1_d2_train_Y = train_X[3600:7200], train_label[3600:7200], train_Y[3600:7200]


    ## preprocess
    train_X = np.mean(train_X, axis=-1) # get SCP
    test_X = np.mean(test_X, axis=-1)




if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    # main()
    # main_meg_repetiton_roi()