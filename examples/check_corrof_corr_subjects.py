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
import mne
import pandas as pd
import scipy.io



def prepare_dataset(args, split, manual_ch=None, onsets:dict=None):
    DATAROOT = args.data_root
    processed_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')
    label_path_pattern = os.path.join(DATAROOT, '{sub}/labels/{name}')
    trigger_meg_path_pattern = os.path.join(DATAROOT, '{sub}/trigger/{name}')
    processed_rest_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')
    processed_kernel_path_pattern = os.path.join(DATAROOT, '{sub}/kernel/{name}')

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
    target_roi_indices = get_kernel_block_ids(args)
    for sub in pbar:
        pbar.set_description("load subject data -- current: {}".format(sub))
        fs = args.subjects[sub]['fs']
        common_kernel_path = os.path.join(DATAROOT, f'{sub}/kernel/tess_cortex_pial_low.mat')
        for meg_name, label_name, trigger_name, rest_name, kernel_name in zip(args.subjects[sub][split]['mat'], args.subjects[sub][split]['labels'], args.subjects[sub][split]['trigger'], args.subjects[sub][split]['rest'], args.subjects[sub][split]['kernel']):
            processed_meg_path = processed_meg_path_pattern.format(sub=sub, name=meg_name)
            label_path = label_path_pattern.format(sub=sub, name=label_name)
            trigger_path = trigger_meg_path_pattern.format(sub=sub, name=trigger_name)
            processed_rest_meg_path = processed_rest_meg_path_pattern.format(sub=sub, name=rest_name)
            subject_kernel_path = processed_kernel_path_pattern.format(sub=sub, name=kernel_name)
            target_region_kernel = get_common_kernel(target_roi_indices, subject_kernel_path, common_kernel_path)
            if args.z_scoring:
                print('z_scoring start')
                rest_mean, rest_std = get_baseline(processed_rest_meg_path, fs, args.rest_duration)
            MEG_Data, image_features, labels, triggers = get_meg_data(processed_meg_path, label_path, trigger_path, rest_mean=rest_mean, rest_std=rest_std, split=split)
            if onsets is None:
                ROI_MEG_Data = MEG_Data[roi_channels, :] #  num_roi_channels x time_samples
                assert len(ROI_MEG_Data) == 160, 'get {}'.format(len(ROI_MEG_Data)) # len(target_region_kernel)
                # ROI_MEG_Data = np.matmul(target_region_kernel, ROI_MEG_Data)
                if args.preprocs.brain_filter is not None:
                    brain_filter_low = args.preprocs.brain_filter[0]
                    brain_filter_high = args.preprocs.brain_filter[1]
                    ROI_MEG_Data = mne.filter.filter_data(ROI_MEG_Data, sfreq=fs, l_freq=brain_filter_low, h_freq=brain_filter_high,)
                    print(f'band path filter: {brain_filter_low}-{brain_filter_high}')
                if args.preprocs.brain_resample_rate is not None:
                    ROI_MEG_Data = mne.filter.resample(ROI_MEG_Data, down=fs / args.preprocs.brain_resample_rate)
                    print('resample {} to {} Hz'.format(fs,args.preprocs.brain_resample_rate))
                    window = time_window(args, triggers, args.preprocs.brain_resample_rate)
                else:
                    window = time_window(args, triggers, fs)
                # import pdb; pdb.set_trace()
                ROI_MEG_Data = np.matmul(target_region_kernel, ROI_MEG_Data)
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

def get_kernel_block_ids(args):
    target_roi_indices = []
    for i in args.roi_block_ids:
        filepath = args.bdata_path.format(block_id=str(i).zfill(2))
        print('roi block path: ', filepath)
        roi = scipy.io.loadmat(filepath)
        target_roi_indices.append(roi['vertex_ds'][:,0])
        # target_roi_indices.append(roi['vertex'][:,0])
    target_roi_indices = np.concatenate(target_roi_indices, axis=0) - 1 # 0始まりなので
    print('target_roi_indices', target_roi_indices.shape)
    return target_roi_indices

def get_common_kernel(target_roi_indices, sub_kernel_path, common_kernel_path):

    sub_kernel = scipy.io.loadmat(sub_kernel_path)
    common_kernel = scipy.io.loadmat(common_kernel_path)
    sub_mat = sub_kernel['ImagingKernel']
    common_mat = common_kernel['tess2tess_interp']['Wmat'][0,0].toarray()

    # target_region_kernel = common_mat[target_roi_indices] @ sub_mat
    # target_region_kernel = sub_mat[target_roi_indices]
    target_region_kernel = (common_mat.T)[target_roi_indices] @ sub_mat
    return target_region_kernel


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

    corr_of_corr = np.corrcoef(corrmat1.flatten(), corrmat2.flatten())

    return corr_of_corr

def calc_corr_of_corr(meg_data:np.ndarray, meg_data2:np.ndarray, savedir:str)->np.ndarray:
    """_summary_

    Args:
        meg_data (np.ndarray): n_epochs x n_ch x time_samples
        meg_data2 (np.ndarray): n_epochs x n_ch x time_samples

    Returns:
        np.ndarray: _description_
    """
    meg_corr = np.corrcoef(meg_data.reshape(meg_data.shape[0], -1))
    vis_corr(meg_corr, os.path.join(savedir, 'meg_corr.png'))
    vis_corr(meg_corr[:100,:100], os.path.join(savedir, 'meg_corr_zoom.png'))
    meg2_corr = np.corrcoef(meg_data2.reshape(meg_data2.shape[0], -1))
    # import pdb; pdb.set_trace()
    # vis_corr(image_corr, os.path.join(savedir, 'image_corr.png'))
    # vis_corr(image_corr[:100,:100], os.path.join(savedir, 'image_corr_zoom.png'))
    meg_corr = meg_corr[np.triu(meg_corr, k=1)!=0]
    meg2_corr = meg2_corr[np.triu(meg2_corr, k=1)!=0]
    meg_meg2_corr_corr = corr_of_corr(meg_corr,meg2_corr)
    # vis_corr(meg_image_corr_corr, os.path.join(savedir, 'corr_of_corr.png'))
    print('correlation: ', meg_meg2_corr_corr[0,1])

    plt.scatter(meg_corr, meg2_corr, s=2)
    plt.xlabel('meg1_corr')
    plt.ylabel('meg2_corr')
    plt.savefig(os.path.join(savedir, 'meg_cross_sbj_scatter.png'))
    print('save to ', os.path.join(savedir, 'meg_cross_sbj_scatter.png'))
    plt.close()
    return meg_meg2_corr_corr

def vis_corr(corr, savefile=None):
    sns.heatmap(corr, square=True, annot=False)
    plt.savefig(savefile)
    plt.close()
    print('saved to ', savefile)

def same_image2neighbor(X, Y, label):
    same_image_indices = []
    for i in range(1,1201):
        same_image_indices += list(np.where(label==i)[0])
        assert len(list(np.where(label==i)[0])) > 0
    return X[same_image_indices], Y[same_image_indices]


def run_original(X, Y, saveroot, subdir):
    # そのまま、corrを計算
    savedir = os.path.join(saveroot, subdir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    return calc_corr_of_corr(X, Y, savedir)

def run_normalize_trial(X, Y, saveroot, subdir):
    # epoch方向にnormalizeしてcorrを計算
    savedir = os.path.join(saveroot, subdir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # import pdb; pdb.set_trace()
    X = X - X.mean(axis=0, keepdims=True)
    X = X / X.std(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Y = Y / Y.std(axis=0, keepdims=True)
    return calc_corr_of_corr(X, Y, savedir)

def run_SCP(X, Y, saveroot, subdir):
    # SCPを計算してcorrを計算
    savedir = os.path.join(saveroot, subdir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    X = X - X.mean(axis=0, keepdims=True)
    X = X / X.std(axis=0, keepdims=True)
    X = np.mean(X, axis=-1)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Y = Y / Y.std(axis=0, keepdims=True)
    return calc_corr_of_corr(X, Y, savedir)


def parse_dataset(sub_list, train_X, train_subs, train_label):
    random.seed(0)
    train_subs = np.array(train_subs)
    data_dict = {s:[] for s in sub_list}
    num_labels = len(np.unique(train_label))
    for i, sub in enumerate(sub_list):
        sub_flags = train_subs==i
        tmp_data = []
        for label in range(1, num_labels+1):
            label_flags = train_label==label
            tmp_data.append(train_X[sub_flags & label_flags,:,:])
        data_dict[sub] = np.concatenate(tmp_data, axis=0)
    return data_dict



def run(args, ch_ratios=1, manual_ch:list=None, onsets:dict=None, saveroot:str='/home/yainoue/meg2image/results/20230531_corr_corr_reconst'):
    s1 = 0
    s2 = 2
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    random.seed(0)
    sub_list = list(args.subjects.keys())
    ## prepare dataset
    train_X, train_subs, train_label, train_Y = prepare_dataset(args, split='train', manual_ch=manual_ch, onsets=onsets)

    sbj_data_dict = parse_dataset(sub_list, train_X, train_subs, train_label)

    corr_of_corr1 = run_normalize_trial(sbj_data_dict[sub_list[s1]], sbj_data_dict[sub_list[s2]], saveroot, 's1d1_norm_unit') # 0.01078 # 0.01068 # 0.03266 (2-13 Hz) #  0.0160 # 0.032 # 0.021 # 0.21(30-40) # 0.023 # 0.024(30-60) # 0.0265(30-70) # 0.028(60-70) # 0.043(60-100) # 0.046

    print('{} vs{}: '.format(sub_list[s1], sub_list[s2]), corr_of_corr1)
    return corr_of_corr1


@hydra.main(version_base=None, config_path="../../configs", config_name="20230531_all_eegnet_regression_src_reconst.yaml")
def main(args):
    # bandpass_list = [(2,5), (5,12), (2,12), (12, 30), (30, 40), (60, 80), (2, 120)]
    # ret_dict = []
    # for bandpass in bandpass_list:
    #     args.preprocs.brain_filter = bandpass
    #     coc1, coc2 = run(args)
    #     ret_dict.append({'bandpass':bandpass, 'coc1': coc1, 'coc2':coc2})
    # print(ret_dict)
    args.preprocs.brain_filter = (2,5)
    run(args)



if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()