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
import pickle
from itertools import product
from time import time

import numpy as np
import pandas as pd
from scipy import stats

from meg_decoding.kamitani_lab.slir import SparseLinearRegression
from sklearn.linear_model import LinearRegression  # For quick demo

from meg_decoding.kamitani_lab.ml import add_bias
from meg_decoding.kamitani_lab.preproc import select_top
from meg_decoding.kamitani_lab.stats import corrcoef

from meg_decoding.matlab_utils.load_meg import get_meg_data, roi, time_window, get_baseline
import tqdm
import hydra


def prepare_dataset(args, split):
    DATAROOT = args.data_root
    processed_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')
    label_path_pattern = os.path.join(DATAROOT, '{sub}/labels/{name}')
    trigger_meg_path_pattern = os.path.join(DATAROOT, '{sub}/trigger/{name}')
    processed_rest_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')

    sub_list = list(args.subjects.keys())
    sub_id_map = {s:i for i, s in enumerate(sub_list)}

    roi_channels = roi(args)

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
            ROI_MEG_epochs = epoching(ROI_MEG_Data, window)

            meg_epochs.append(ROI_MEG_epochs) # array [epoch x ch x time_stamp]
            sub_epochs+=[sub_id_map[sub]] * len(ROI_MEG_epochs) # list [epoch]
            label_epochs.append(labels) # array [epoch]
            image_feature_epochs.append(image_features) # array [epoch x dim]
    meg_epochs = np.concatenate(meg_epochs, axis=0)
    label_epochs = np.concatenate(label_epochs, axis=0)
    image_feature_epochs = np.concatenate(image_feature_epochs, axis=0)
    print('dataset is created. size is ', meg_epochs.shape)
    return meg_epochs, sub_epochs, label_epochs, image_feature_epochs

@hydra.main(version_base=None, config_path="../../configs", config_name="20230420_sbj01_linear.yaml")
def main_meg(args):
    ## prepare dataset
    train_X, train_subs, train_label, train_Y = prepare_dataset(args, split='train')
    test_X, test_subs, test_label, test_Y = prepare_dataset(args, split='test')

    ## preprocess
    pass
    ## feature prediction 
    pred_y, true_y = feature_prediction(train_X, train_Y,
                                            test_X, test_Y,
                                            n_voxel=20,
                                            n_iter=200)


# Main #################################################################

def main():
    # Settings ---------------------------------------------------------

    # Data settings
    subjects = config.subjects
    rois = config.rois
    num_voxel = config.num_voxel

    image_feature = config.image_feature_file
    features = config.features

    n_iter = 200

    results_dir = config.results_dir

    # Misc settings
    analysis_basename = os.path.basename(__file__)

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_all = {}
    for sbj in subjects:
        if len(subjects[sbj]) == 1:
            data_all[sbj] = bdpy.BData(subjects[sbj][0])
        else:
            # Concatenate data
            suc_cols = ['Run', 'Block']
            data_all[sbj] = concat_dataset([bdpy.BData(f) for f in subjects[sbj]],
                                           successive=suc_cols)

    data_feature = bdpy.BData(image_feature)

    # Add any additional processing to data here

    # Initialize directories -------------------------------------------
    makedir_ifnot(results_dir)
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for sbj, roi, feat in product(subjects, rois, features):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % num_voxel[roi])
        print('Feature:    %s' % feat)

        # Distributed computation
        analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
        results_file = os.path.join(results_dir, analysis_id + '.pkl')

        if os.path.exists(results_file):
            print('%s is already done. Skipped.' % analysis_id)
            continue

        dist = DistComp(lockdir='tmp', comp_id=analysis_id)
        if dist.islocked():
            print('%s is already running. Skipped.' % analysis_id)
            continue

        dist.lock()

        # Prepare data
        print('Preparing data')
        dat = data_all[sbj]

        x = dat.select(rois[roi])           # Brain data
        datatype = dat.select('DataType')   # Data type
        labels = dat.select('stimulus_id')  # Image labels in brain data

        y = data_feature.select(feat)             # Image features
        y_label = data_feature.select('ImageID')  # Image labels

        # For quick demo, reduce the number of units from 1000 to 100
        y = y[:, :100]

        y_sorted = get_refdata(y, y_label, labels)  # Image features corresponding to brain data

        # Get training and test dataset
        i_train = (datatype == 1).flatten()    # Index for training
        i_test_pt = (datatype == 2).flatten()  # Index for perception test
        i_test_im = (datatype == 3).flatten()  # Index for imagery test
        i_test = i_test_pt + i_test_im

        x_train = x[i_train, :]
        x_test = x[i_test, :]

        y_train = y_sorted[i_train, :]
        y_test = y_sorted[i_test, :]

        # Feature prediction
        pred_y, true_y = feature_prediction(x_train, y_train,
                                            x_test, y_test,
                                            n_voxel=num_voxel[roi],
                                            n_iter=n_iter)

        # Separate results for perception and imagery tests
        i_pt = i_test_pt[i_test]  # Index for perception test within test
        i_im = i_test_im[i_test]  # Index for imagery test within test

        pred_y_pt = pred_y[i_pt, :]
        pred_y_im = pred_y[i_im, :]

        true_y_pt = true_y[i_pt, :]
        true_y_im = true_y[i_im, :]

        # Get averaged predicted feature
        test_label_pt = labels[i_test_pt, :].flatten()
        test_label_im = labels[i_test_im, :].flatten()

        pred_y_pt_av, true_y_pt_av, test_label_set_pt \
            = get_averaged_feature(pred_y_pt, true_y_pt, test_label_pt)
        pred_y_im_av, true_y_im_av, test_label_set_im \
            = get_averaged_feature(pred_y_im, true_y_im, test_label_im)

        # Get category averaged features
        catlabels_pt = np.vstack([int(n) for n in test_label_pt])  # Category labels (perception test)
        catlabels_im = np.vstack([int(n) for n in test_label_im])  # Category labels (imagery test)
        catlabels_set_pt = np.unique(catlabels_pt)                 # Category label set (perception test)
        catlabels_set_im = np.unique(catlabels_im)                 # Category label set (imagery test)

        y_catlabels = data_feature.select('CatID')   # Category labels in image features
        ind_catave = (data_feature.select('FeatureType') == 3).flatten()

        y_catave_pt = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_pt)
        y_catave_im = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_im)

        # Prepare result dataframe
        results = pd.DataFrame({'subject' : [sbj, sbj],
                                'roi' : [roi, roi],
                                'feature' : [feat, feat],
                                'test_type' : ['perception', 'imagery'],
                                'true_feature': [true_y_pt, true_y_im],
                                'predicted_feature': [pred_y_pt, pred_y_im],
                                'test_label' : [test_label_pt, test_label_im],
                                'test_label_set' : [test_label_set_pt, test_label_set_im],
                                'true_feature_averaged' : [true_y_pt_av, true_y_im_av],
                                'predicted_feature_averaged' : [pred_y_pt_av, pred_y_im_av],
                                'category_label_set' : [catlabels_set_pt, catlabels_set_im],
                                'category_feature_averaged' : [y_catave_pt, y_catave_im]})

        # Save results
        makedir_ifnot(os.path.dirname(results_file))
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        print('Saved %s' % results_file)

        dist.unlock()


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

    for i in range(n_unit):

        print('Unit %03d' % (i + 1))
        start_time = time()

        # Get unit features
        y_train_unit = y_train[:, i]
        y_test_unit =  y_test[:, i]

        # Normalize image features for training (y_train_unit)
        norm_mean_y = np.mean(y_train_unit, axis=0)
        std_y = np.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y

        y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        # Voxel selection
        corr = corrcoef(y_train_unit, x_train, var='col')

        x_train_unit, voxel_index = select_top(x_train, np.abs(corr), n_voxel, axis=1, verbose=False)
        x_test_unit = x_test[:, voxel_index]

        # Add bias terms
        x_train_unit = add_bias(x_train_unit, axis=1)
        x_test_unit = add_bias(x_test_unit, axis=1)

        # Setup regression
        # For quick demo, use linaer regression
        model = LinearRegression()
        #model = SparseLinearRegression(n_iter=n_iter, prune_mode=1)

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

        print('Time: %.3f sec' % (time() - start_time))

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
    main()