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
                # if args.preprocs.brain_filter is not None:
                #     brain_filter_low = args.preprocs.brain_filter[0]
                #     brain_filter_high = args.preprocs.brain_filter[1]
                #     ROI_MEG_Data = mne.filter.filter_data(ROI_MEG_Data, sfreq=fs, l_freq=brain_filter_low, h_freq=brain_filter_high,)
                #     print(f'band path filter: {brain_filter_low}-{brain_filter_high}')
                # if args.preprocs.brain_resample_rate is not None:
                #     ROI_MEG_Data = mne.filter.resample(ROI_MEG_Data, down=fs / args.preprocs.brain_resample_rate)
                #     print('resample {} to {} Hz'.format(fs,args.preprocs.brain_resample_rate))
                #     window = time_window(args, triggers, args.preprocs.brain_resample_rate)
                # else:
                #     window = time_window(args, triggers, fs)
                window = time_window(args, triggers, fs)
                # import pdb; pdb.set_trace()
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
                # import pdb; pdb.set_trace()

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

# def metric(predicted_y, val_index, use_average=False):
#     # predicted_y: num_trials x 512
#     # val_index: num_trials
#     val_index = val_index - 1 # ラベルは1始まり
#     image_features = np.load('./data/GOD/image_features.npy') # 50 x 512
#     if use_average:
#         print('use average')
#         predicted_y, val_index = get_average_features(predicted_y, val_index)
#     num_images = len(image_features)
#     num_trials = len(predicted_y)
#     acc_tmp = np.zeros((num_trials, 1))

#     cat_wise_acc = {i:[] for i in range(len(image_features))}
#     # import pdb; pdb.set_trace()
#     for i_pred in range(num_trials):
#         space_corr = np.zeros((num_images, 1))
#         # iterating over all images
#         # calculating the correlation between the predicted and the image features
#         for i_img in range(num_images):
#             R = np.corrcoef(predicted_y[i_pred], image_features[i_img])
#             space_corr[i_img] = R[0,1]

#         # assigning the index of the current predicred vector to image_id
#         image_id = val_index[i_pred]
#         # calculating the accuracy of the cirrent predicted vector by counting the number of images with correlation coefficiens less than that of corresponding image
#         # and dividing by the total number of images minus one

#         acc_tmp[i_pred] = np.sum(space_corr < space_corr[image_id]) / (num_images - 1)
#         cat_wise_acc[image_id].append(acc_tmp[i_pred])
#     cat_wise_acc = {i: np.mean(cat_wise_acc[i]) for i in range(len(image_features))}
#     return np.mean(acc_tmp), cat_wise_acc

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


def run_meg_fit_and_evaluate(args, ch_ratios=1, manual_ch:list=None, onsets:dict=None):
    prefix='sbj01'
    random.seed(0)
    ## prepare dataset
    train_X, train_subs, train_label, train_Y = prepare_dataset(args, split='train', manual_ch=manual_ch, onsets=onsets)
    test_X, test_subs, test_label, test_Y = prepare_dataset(args, split='val', manual_ch=manual_ch, onsets=onsets)
    # import pdb; pdb.set_trace()

    ## preprocess
    train_X = np.mean(train_X, axis=-1) # get SCP
    test_X = np.mean(test_X, axis=-1) 

    ## feature prediction 
    n_voxel = int(ch_ratios*(train_X.shape[1]))
    print('n_voxel: ', n_voxel)
    pred_y, true_y = feature_prediction(train_X, train_Y,
                                            test_X, test_Y,
                                            n_voxel=n_voxel,
                                            n_iter=200)
    
    ## evaluate
    pred_y_avg, true_y_avg, test_label_set = get_averaged_feature(pred_y, true_y, test_label)
    results = {}
    results['predicted_feature_averaged_percept'] = pred_y_avg
    results['category_feature_averaged_percept'] = true_y_avg
    acc, cat_wise_acc = metric_kamitani(results, test_label_set)# metric(pred_y_avg, test_label_set, use_average=False)
    print('ACC from binary corr', acc)
    print({k: '{:.3f}'.format(v) for k, v in cat_wise_acc.items()})

    ## save
    savefile = os.path.join(args.save_root, '{}-{}_{}-{:.3f}'.format(prefix, args.window.start, args.window.end, acc), 'ridge_regression.pkl')
    if not os.path.exists(os.path.dirname(savefile)):
        os.makedirs(os.path.dirname(savefile))
    # with open(savefile, 'wb') as f:
    #     pickle.dump({'pred_y':pred_y, 'true_y': true_y, 'test_label': test_label}, f)
    print('results is saved as ', savefile)
    # import pdb; pdb.set_trace()
    return acc

    

@hydra.main(version_base=None, config_path="../../configs", config_name="20230421_sbj01_kamitani_regression.yaml")
def main_meg_repetiton_roi(args):
    onsets =  [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    duration = 0.2
    
    roi_names = ['occipital', 'parietal', 'frontal', 'temporal', 'central']
    for roi_name in roi_names:
        acc_list = []
        args.region = [roi_name+'/right', roi_name + '/left']
        for start_ in onsets:
            args.window.start = start_
            args.window.end = start_ + duration
            acc = run_meg_fit_and_evaluate(args)  
            acc_list.append(acc)
        plt.plot(onsets, acc_list, label=roi_name)

    plt.xlabel('onset [s]')
    plt.ylabel('Acc')
    plt.legend()
    plt.title('sbj01 - 200 ms window')
    savefile = os.path.join(args.save_root, f'ridge_regression_{duration}s.png')
    plt.savefig(savefile)
    print('figure is saved as ', savefile)

@hydra.main(version_base=None, config_path="../../configs", config_name="20230421_sbj01_kamitani_regression.yaml")
def main_meg_repetiton_N(args):
    ch_ratios =  [0.2,  0.3,  0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plt.figure(figsize=(12,6))
    roi_names = ['occipital', 'parietal', 'frontal', 'temporal', 'central']
    roi_name_pairs = []
    results = {}
    for n in range(1, len(roi_names)+1):
        roi_name_pairs += list(itertools.combinations(roi_names,n))
    for roi_name_pair in roi_name_pairs:
        print('============================\n{}\n=================================='.format(roi_name_pair))
        acc_list = []
        region = []
        for r in roi_name_pair:
            region += [r+'/right', r+'/left']

        args.region = region
        for ch_ratio in ch_ratios:
            acc = run_meg_fit_and_evaluate(args, ch_ratio)  
            acc_list.append(acc)
        label = '-'.join(roi_name_pair)
        plt.plot(ch_ratios, acc_list, label=label)
        results[label] = acc_list
    pkl_file =  os.path.join(args.save_root, f'ridge_regression_ch_ratio.pkl')
    with open(pkl_file, 'wb' ) as f:
        pickle.dump(results, f)

    plt.xlabel('ch_ratio')
    plt.ylabel('Acc')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,)
    plt.title('sbj01 - 200 ms window')
    savefile = os.path.join(args.save_root, f'ridge_regression_ch_ratio.png')
    plt.savefig(savefile, bbox_inches="tight")
    print('figure is saved as ', savefile)
    print(results)

@hydra.main(version_base=None, config_path="../../configs", config_name="20230421_sbj01_kamitani_regression.yaml")
def main_meg_repetiton_onsets_per_ch(args):
    savefile = os.path.join(args.save_root, f'ridge_regression_var_onsets_per_ch.csv')
    onsets =  [0.2,  0.25,  0.3]# , 0.35]
    plt.figure(figsize=(12,6))
    roi_names = ['occipital', 'parietal', 'frontal', 'temporal', 'central']

    ## initialize
    results = {'acc':[]}
    for r in roi_names:
        results[r] = []

    cands_onsets = list(itertools.product(onsets, repeat=len(roi_names)))
    print(cands_onsets)
    for onset_list in cands_onsets:
        # onset_dict = {k:v for k, v in zip(roi_names, onsets)}
        onset_dict = {}
        for r, o in zip(roi_names, onset_list):
            onset_dict[r+'/right'] = o
            onset_dict[r+'/left'] = o
        print('===============================================================')
        print(onset_dict)
        print('===============================================================')

        acc = run_meg_fit_and_evaluate(args, ch_ratios=1, onsets=onset_dict)
        results['acc'].append(acc)
        for k, v in zip(roi_names, onset_list):
            results[k].append(v)
    
        df = pd.DataFrame(results)
    
        df.to_csv(savefile)
    print('results is saved as ', savefile)




@hydra.main(version_base=None, config_path="../../configs", config_name="20230421_sbj01_kamitani_regression.yaml")
def main_meg_run_manual_ch(args):
    acc_list = []
    manual_ch_list = [ [136, 137, 139, 151, 152, 154],
                       [136, 137, 139, 151, 152, 154, 135, 153],
                        [136, 137, 139, 151, 152, 154, 135, 153, 134, 149],
                        [136, 137, 139, 151, 152, 154, 135, 153, 134, 149, 133, 138, 150, 155],
                        None]
    for manual_ch in manual_ch_list:
        if manual_ch is not None:
            manual_ch = [c - 1 for c in manual_ch] # matlab -> python
        acc = run_meg_fit_and_evaluate(args, manual_ch=manual_ch)  
        acc_list.append(acc)
    print(acc_list)

# # Main #################################################################

# def main():
#     # Settings ---------------------------------------------------------

#     # Data settings
#     subjects = config.subjects
#     rois = config.rois
#     num_voxel = config.num_voxel

#     image_feature = config.image_feature_file
#     features = config.features

#     n_iter = 200

#     results_dir = config.results_dir

#     # Misc settings
#     analysis_basename = os.path.basename(__file__)

#     # Load data --------------------------------------------------------
#     print('----------------------------------------')
#     print('Loading data')

#     data_all = {}
#     for sbj in subjects:
#         if len(subjects[sbj]) == 1:
#             data_all[sbj] = bdpy.BData(subjects[sbj][0])
#         else:
#             # Concatenate data
#             suc_cols = ['Run', 'Block']
#             data_all[sbj] = concat_dataset([bdpy.BData(f) for f in subjects[sbj]],
#                                            successive=suc_cols)

#     data_feature = bdpy.BData(image_feature)

#     # Add any additional processing to data here

#     # Initialize directories -------------------------------------------
#     makedir_ifnot(results_dir)
#     makedir_ifnot('tmp')

#     # Analysis loop ----------------------------------------------------
#     print('----------------------------------------')
#     print('Analysis loop')

#     for sbj, roi, feat in product(subjects, rois, features):
#         print('--------------------')
#         print('Subject:    %s' % sbj)
#         print('ROI:        %s' % roi)
#         print('Num voxels: %d' % num_voxel[roi])
#         print('Feature:    %s' % feat)

#         # Distributed computation
#         analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
#         results_file = os.path.join(results_dir, analysis_id + '.pkl')

#         if os.path.exists(results_file):
#             print('%s is already done. Skipped.' % analysis_id)
#             continue

#         dist = DistComp(lockdir='tmp', comp_id=analysis_id)
#         if dist.islocked():
#             print('%s is already running. Skipped.' % analysis_id)
#             continue

#         dist.lock()

#         # Prepare data
#         print('Preparing data')
#         dat = data_all[sbj]

#         x = dat.select(rois[roi])           # Brain data
#         datatype = dat.select('DataType')   # Data type
#         labels = dat.select('stimulus_id')  # Image labels in brain data

#         y = data_feature.select(feat)             # Image features
#         y_label = data_feature.select('ImageID')  # Image labels

#         # For quick demo, reduce the number of units from 1000 to 100
#         y = y[:, :100]

#         y_sorted = get_refdata(y, y_label, labels)  # Image features corresponding to brain data

#         # Get training and test dataset
#         i_train = (datatype == 1).flatten()    # Index for training
#         i_test_pt = (datatype == 2).flatten()  # Index for perception test
#         i_test_im = (datatype == 3).flatten()  # Index for imagery test
#         i_test = i_test_pt + i_test_im

#         x_train = x[i_train, :]
#         x_test = x[i_test, :]

#         y_train = y_sorted[i_train, :]
#         y_test = y_sorted[i_test, :]

#         # Feature prediction
#         pred_y, true_y = feature_prediction(x_train, y_train,
#                                             x_test, y_test,
#                                             n_voxel=num_voxel[roi],
#                                             n_iter=n_iter)

#         # Separate results for perception and imagery tests
#         i_pt = i_test_pt[i_test]  # Index for perception test within test
#         i_im = i_test_im[i_test]  # Index for imagery test within test

#         pred_y_pt = pred_y[i_pt, :]
#         pred_y_im = pred_y[i_im, :]

#         true_y_pt = true_y[i_pt, :]
#         true_y_im = true_y[i_im, :]

#         # Get averaged predicted feature
#         test_label_pt = labels[i_test_pt, :].flatten()
#         test_label_im = labels[i_test_im, :].flatten()

#         pred_y_pt_av, true_y_pt_av, test_label_set_pt \
#             = get_averaged_feature(pred_y_pt, true_y_pt, test_label_pt)
#         pred_y_im_av, true_y_im_av, test_label_set_im \
#             = get_averaged_feature(pred_y_im, true_y_im, test_label_im)

#         # Get category averaged features
#         catlabels_pt = np.vstack([int(n) for n in test_label_pt])  # Category labels (perception test)
#         catlabels_im = np.vstack([int(n) for n in test_label_im])  # Category labels (imagery test)
#         catlabels_set_pt = np.unique(catlabels_pt)                 # Category label set (perception test)
#         catlabels_set_im = np.unique(catlabels_im)                 # Category label set (imagery test)

#         y_catlabels = data_feature.select('CatID')   # Category labels in image features
#         ind_catave = (data_feature.select('FeatureType') == 3).flatten()

#         y_catave_pt = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_pt)
#         y_catave_im = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_im)

#         # Prepare result dataframe
#         results = pd.DataFrame({'subject' : [sbj, sbj],
#                                 'roi' : [roi, roi],
#                                 'feature' : [feat, feat],
#                                 'test_type' : ['perception', 'imagery'],
#                                 'true_feature': [true_y_pt, true_y_im],
#                                 'predicted_feature': [pred_y_pt, pred_y_im],
#                                 'test_label' : [test_label_pt, test_label_im],
#                                 'test_label_set' : [test_label_set_pt, test_label_set_im],
#                                 'true_feature_averaged' : [true_y_pt_av, true_y_im_av],
#                                 'predicted_feature_averaged' : [pred_y_pt_av, pred_y_im_av],
#                                 'category_label_set' : [catlabels_set_pt, catlabels_set_im],
#                                 'category_feature_averaged' : [y_catave_pt, y_catave_im]})

#         # Save results
#         makedir_ifnot(os.path.dirname(results_file))
#         with open(results_file, 'wb') as f:
#             pickle.dump(results, f)

#         print('Saved %s' % results_file)

#         dist.unlock()


# Functions ############################################################

def feature_prediction(x_train, y_train, x_test, y_test, n_voxel=500, n_iter=200):
    '''Run feature prediction
    Parameters
    ----------
    x_train, y_train : array_like [shape = (n_sample, n_voxel)]
        Brain data and image features for training
    x_test, y_test : array_like [shape = (n_sample, n_unit)]
        Brain data and image features for test
    n_voxel : int
        The number of voxels
    n_iter : int
        The number of iterations
    Returns
    -------
    predicted_label : array_like [shape = (n_sample, n_unit)]
        Predicted features
    ture_label : array_like [shape = (n_sample, n_unit)]
        True features in test data
    '''

    n_unit = y_train.shape[1]

    # Normalize brian data (x)
    norm_mean_x = np.mean(x_train, axis=0)
    norm_scale_x = np.std(x_train, axis=0, ddof=1)

    x_train = (x_train - norm_mean_x) / norm_scale_x
    x_test = (x_test - norm_mean_x) / norm_scale_x

    # Feature prediction for each unit
    print('Running feature prediction')

    y_true_list = []
    y_pred_list = []

    pbar = tqdm.tqdm(range(n_unit))

    for i in pbar:

        start_time = time()

        # Get unit features
        y_train_unit = y_train[:, i]
        y_test_unit =  y_test[:, i]

        # Normalize image features for training (y_train_unit)
        norm_mean_y = np.mean(y_train_unit, axis=0)
        std_y = np.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y

        y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y
        # import pdb; pdb.set_trace()
        # Voxel selection
        corr = corrcoef(y_train_unit, x_train, var='col')

        x_train_unit, voxel_index = select_top(x_train, np.abs(corr), n_voxel, axis=1, verbose=False)
        x_test_unit = x_test[:, voxel_index]

        # Add bias terms
        x_train_unit = add_bias(x_train_unit, axis=1)
        x_test_unit = add_bias(x_test_unit, axis=1)

        # Setup regression
        # For quick demo, use linaer regression
        # model = LinearRegression()
        model = SparseLinearRegression(n_iter=n_iter, prune_mode=1)

        # Training and test
        try:
            model.fit(x_train_unit, y_train_unit)  # Training
            y_pred = model.predict(x_test_unit)    # Test
        except:
            # When SLiR failed, returns zero-filled array as predicted features
            y_pred = np.zeros(y_test_unit.shape)

        # Denormalize predicted features
        y_pred = y_pred * norm_scale_y + norm_mean_y

        y_true_list.append(y_test_unit)
        y_pred_list.append(y_pred)

        pbar.set_description('Unit %03d' % (i + 1) + 'Time: %.3f sec' % (time() - start_time))

    # Create numpy arrays for return values
    y_predicted = np.vstack(y_pred_list).T
    y_true = np.vstack(y_true_list).T

    return y_predicted, y_true


def get_averaged_feature(pred_y, true_y, labels):
    '''Return category-averaged features'''

    labels_set = np.unique(labels)

    pred_y_av = np.array([np.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = np.array([np.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


# Run as a scirpt ######################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    # main()
    # main_meg_repetiton_roi()
    # main_meg_run_manual_ch()
    # main_meg_repetiton_N()
    main_meg_repetiton_onsets_per_ch()
    # main_meg()