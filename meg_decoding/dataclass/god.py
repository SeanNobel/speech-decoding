import os
from torch.utils.data import Sampler, Dataset
import numpy as np
import mne
from tqdm import tqdm
from meg_decoding.matlab_utils.load_meg import get_meg_data, roi, time_window, get_baseline
import tqdm
import torch
from typing import List
from meg_decoding.utils.preproc_utils import (
    check_preprocs,
    scaleAndClamp,
    scaleAndClamp_single,
    baseline_correction_single,
)
import scipy.io
mne.set_log_level(verbose="WARNING")


def normalize_per_unit(tensor, subs, return_stats=False):
    print('normalize image_feature along unit dim')
    # array: n_samples x n_units(512)
    subs = np.array(subs)
    subject_list = np.unique(subs)
    means = []
    stds = []
    for sub in subject_list:
        indices = np.where(subs==sub)[0]
        mean = np.mean(tensor[indices,:], axis=0, keepdims=True)
        std = np.std(tensor[indices,:], axis=0,  keepdims=True)
        tensor[indices,:] = tensor[indices,:] - mean
        tensor[indices,:] = tensor[indices,:] / std
        means.append(mean)
        stds.append(std)
        # import pdb; pdb.set_trace()
    if return_stats:
        if len(subject_list)==1:
            return tensor, mean, std
        else:
            return tensor, means, stds
    else:
        return tensor

class GODDatasetBase(Dataset):
    def __init__(self, args, split, preprocess_pipleine:list=[], return_label:bool=False,
                 mean_X=None, mean_Y=None, std_X=None, std_Y=None):
        self.args = args
        self.sub_id_map = {s:i for i, s in enumerate(list(self.args.subjects.keys()))}
        self.preprocess_pipeline = preprocess_pipleine
        # prepare dataset
        meg_epochs, sub_epochs, label_epochs, \
            image_feature_epochs = self.prepare_data(args, split=split)

        self.X = meg_epochs.astype(np.float32) # epochs x ch x time_samples
        self.Y = image_feature_epochs.astype(np.float32) # epochs x dims
        if mean_X is not None:
            print('MEG is normalized by given stats')
            self.mean_X, self.std_X= mean_X, std_X
            if isinstance(mean_X, list):
                for i in range(len(mean_X)):
                    indices = np.where(sub_epochs==i)[0]
                    self.X[indices,:] = self.X[indices,:] - self.mean_X[i]
                    self.X[indices,:] = self.X[indices,:] / self.std_X[i]
            else:
                self.X = self.X - self.mean_X
                self.X = self.X / self.std_X
        else:
            if args.normalize_meg:
                print('MEG is normalized by self stats')
                self.X, self.mean_X, self.std_X = normalize_per_unit(self.X, sub_epochs, return_stats=True)
            else:
                self.mean_X, self.std_X = None, None
        if mean_Y is not None:
            print('image features is normalized by self stats')
            self.mean_Y, self.std_Y = mean_Y, std_Y
            self.Y = self.Y - self.mean_Y
            self.Y = self.Y / self.std_Y
        else:
            if args.normalize_image_features:
                print('Image features is normalized by self stats')
                self.Y, self.mean_Y, self.std_Y = normalize_per_unit(self.Y, np.zeros_like(sub_epochs), return_stats=True)
            else:
                self.mean_Y, self.std_Y = None, None

        self.subs = sub_epochs # epochs (x 1)
        self.labels = label_epochs # epochs (x 1)

        if split == 'val':
            self.X, self.Y, self.subs, self.labels = self.avg_same_image_sub_epochs(self.X, self.Y, self.subs, self.labels)

        self.labels = self.labels.astype(np.int16)
        self.num_subjects = len(np.unique(self.subs))
        self.return_label = return_label

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):  # NOTE: i is id of a speech segment
        x, y, s = self.X[i], self.Y[i], self.subs[i]
        if self.return_label:
            l = self.labels[i]
            return x, y, s, l
        else:
            return x, y, s

    def prepare_data(self, args, split:str):
        DATAROOT = args.data_root
        processed_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')
        label_path_pattern = os.path.join(DATAROOT, '{sub}/labels/{name}')
        trigger_meg_path_pattern = os.path.join(DATAROOT, '{sub}/trigger/{name}')
        processed_rest_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')
        processed_kernel_path_pattern = os.path.join(DATAROOT, '{sub}/kernel/{name}')

        sub_list = list(args.subjects.keys())

        roi_channels = roi(args)

        def epoching(meg, window):
            assert len(labels) == len(window)
            # n_epochs x n_chs x time_smaples
            meg_epochs = np.zeros([len(window), len(meg), window[0][1]-window[0][0]])
            for i, w in enumerate(window):
                window_meg = meg[:, w[0]:w[1]] # ch x time_samples
                for func_ in self.preprocess_pipeline:
                    window_meg = func_(window_meg)
                meg_epochs[i] = window_meg
            return meg_epochs

        meg_epochs = []
        sub_epochs = []
        label_epochs = []
        image_feature_epochs = []
        rest_mean, rest_std = None, None
        pbar = tqdm.tqdm(sub_list)
        if args.src_reconstruction:
            print('src reconstruction')
            target_roi_indices = get_kernel_block_ids(args)
        for sub in pbar:
            pbar.set_description("load subject data -- current: {}".format(sub))
            fs = args.subjects[sub]['fs']

            if args.src_reconstruction:
                common_kernel_path = os.path.join(DATAROOT, f'{sub}/kernel/tess_cortex_pial_low.mat')
                for meg_name, label_name, trigger_name, rest_name, kernel_name in zip(args.subjects[sub][split]['mat'], args.subjects[sub][split]['labels'], args.subjects[sub][split]['trigger'], args.subjects[sub][split]['rest'], args.subjects[sub][split]['kernel']):
                    processed_meg_path = processed_meg_path_pattern.format(sub=sub, name=meg_name)
                    label_path = label_path_pattern.format(sub=sub, name=label_name)
                    trigger_path = trigger_meg_path_pattern.format(sub=sub, name=trigger_name)
                    processed_rest_meg_path = processed_rest_meg_path_pattern.format(sub=sub, name=rest_name)
                    subject_kernel_path = processed_kernel_path_pattern.format(sub=sub, name=kernel_name)
                    target_region_kernel = get_common_kernel(target_roi_indices, subject_kernel_path, common_kernel_path)
                    if args.z_scoring:
                        rest_mean, rest_std = get_baseline(processed_rest_meg_path, fs, args.rest_duration)
                    MEG_Data, image_features, labels, triggers = get_meg_data(processed_meg_path, label_path, trigger_path, rest_mean=rest_mean, rest_std=rest_std, split=split)
                    ROI_MEG_Data = MEG_Data[roi_channels, :] #  num_roi_channels x time_samples
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
                    assert len(ROI_MEG_Data) == 160 # len(target_region_kernel)
                    ROI_MEG_Data = np.matmul(target_region_kernel, ROI_MEG_Data) # source reconstruction of target region
                    ROI_MEG_epochs = epoching(ROI_MEG_Data, window)

                    meg_epochs.append(ROI_MEG_epochs) # array [epoch x ch x time_stamp]
                    sub_epochs+=[self.sub_id_map[sub]] * len(ROI_MEG_epochs) # list [epoch]
                    label_epochs.append(labels) # array [epoch]
                    image_feature_epochs.append(image_features) # array [epoch x dim]
            else:
                for meg_name, label_name, trigger_name, rest_name in zip(args.subjects[sub][split]['mat'], args.subjects[sub][split]['labels'], args.subjects[sub][split]['trigger'], args.subjects[sub][split]['rest']):
                    processed_meg_path = processed_meg_path_pattern.format(sub=sub, name=meg_name)
                    label_path = label_path_pattern.format(sub=sub, name=label_name)
                    trigger_path = trigger_meg_path_pattern.format(sub=sub, name=trigger_name)
                    processed_rest_meg_path = processed_rest_meg_path_pattern.format(sub=sub, name=rest_name)
                    if args.z_scoring:
                        rest_mean, rest_std = get_baseline(processed_rest_meg_path, fs, args.rest_duration)
                    MEG_Data, image_features, labels, triggers = get_meg_data(processed_meg_path, label_path, trigger_path, rest_mean=rest_mean, rest_std=rest_std, split=split)
                    ROI_MEG_Data = MEG_Data[roi_channels, :] #  num_roi_channels x time_samples
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
                    ROI_MEG_epochs = epoching(ROI_MEG_Data, window)

                    meg_epochs.append(ROI_MEG_epochs) # array [epoch x ch x time_stamp]
                    sub_epochs+=[self.sub_id_map[sub]] * len(ROI_MEG_epochs) # list [epoch]
                    label_epochs.append(labels) # array [epoch]
                    image_feature_epochs.append(image_features) # array [epoch x dim]
        meg_epochs = np.concatenate(meg_epochs, axis=0)
        label_epochs = np.concatenate(label_epochs, axis=0)
        image_feature_epochs = np.concatenate(image_feature_epochs, axis=0)
        print('dataset is created. size is ', meg_epochs.shape)
        # if split == 'val':
        #     meg_epochs, image_feature_epochs, sub_epochs, label_epochs = self.avg_same_image_sub_epochs(meg_epochs, image_feature_epochs, sub_epochs, label_epochs)

        return meg_epochs, sub_epochs, label_epochs, image_feature_epochs

    def avg_same_image_sub_epochs(self, Xs, Ys, subs, labels):
        subs = np.array(subs)
        avg_Xs = []
        avg_Ys = []
        new_subs = []
        new_labels = []
        for i in np.unique(labels):
            for s in np.unique(subs):
                sub_and_label_equivalent_flag = (labels==i) * (subs==s)
                avg_Xs.append(np.mean(Xs[sub_and_label_equivalent_flag], axis=0, keepdims=True))
                avg_Ys.append(np.mean(Ys[sub_and_label_equivalent_flag], axis=0, keepdims=True))
                new_subs.append(s)
                new_labels.append(i)
        return np.concatenate(avg_Xs), np.concatenate(avg_Ys), new_subs, np.asarray(new_labels)

def same_image2neighbor(X, Y, label):
    same_image_indices = []
    for i in range(1,1201):
        same_image_indices += list(np.where(label==i)[0])
        assert len(list(np.where(label==i)[0])) > 0
    return X[same_image_indices], Y[same_image_indices]

def get_kernel_block_ids(args):
    target_roi_indices = []
    for i in args.roi_block_ids:
        filepath = args.bdata_path.format(block_id=i)
        # print('roi block path: ', filepath)
        roi = scipy.io.loadmat(filepath)
        target_roi_indices.append(roi['vertex_ds'][:,0])
    target_roi_indices = np.concatenate(target_roi_indices, axis=0) - 1 # 0始まりなので
    print('target_roi_indices', target_roi_indices.shape)
    return target_roi_indices

def get_common_kernel(target_roi_indices, sub_kernel_path, common_kernel_path):

    sub_kernel = scipy.io.loadmat(sub_kernel_path)
    common_kernel = scipy.io.loadmat(common_kernel_path)
    sub_mat = sub_kernel['ImagingKernel']
    common_mat = common_kernel['tess2tess_interp']['Wmat'][0,0].toarray()

    target_region_kernel = common_mat[target_roi_indices] @ sub_mat
    return target_region_kernel

class GODCollator(torch.nn.Module):
    """
    Runs baseline correction and robust scaling and clamping for batch
    """

    def __init__(self, args, return_label=False):
        super(GODCollator, self).__init__()

        self.brain_resample_rate = args.preprocs["brain_resample_rate"]
        self.baseline_len_samp = int(self.brain_resample_rate * args.preprocs["baseline_len_sec"])
        self.clamp = args.preprocs["clamp"]
        self.clamp_lim = args.preprocs["clamp_lim"]
        self.return_label = return_label

    def forward(self, batch: List[tuple]):
        X = torch.stack([torch.Tensor(item[0]) for item in batch])  # ( 64, 208, 360 )
        Y = torch.stack([torch.Tensor(item[1]) for item in batch])
        # print('debug', [item[2] for item in batch])
        subject_idx = torch.IntTensor([item[2] for item in batch])
        if self.baseline_len_samp > 0:
            X = baseline_correction_single(X, self.baseline_len_samp)
        X = scaleAndClamp(X, self.clamp_lim, self.clamp)
        if self.return_label:
            label = torch.IntTensor([item[3] for item in batch])
            return X, Y, subject_idx, label
        else:
            return X, Y, subject_idx


    @torch.no_grad()
    def baseline_correction_single(self, X: torch.Tensor, baseline_len_samp):
        """args:
            X: ( chunks, ch, time )
        returns:
            X ( chunks, ch, time ) baseline-corrected channel-wise
        """
        X = X.permute(1, 0, 2).clone()  # ( ch, chunks, time )

        for chunk_id in range(X.shape[1]):
            baseline = X[:, chunk_id, -baseline_len_samp:].mean(axis=1)

            X[:, chunk_id, :] -= baseline.view(-1, 1)

        return X.permute(1, 0, 2)
