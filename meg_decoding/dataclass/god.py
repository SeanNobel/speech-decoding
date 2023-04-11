import os
from torch.utils.data import Sampler, Dataset
import numpy as np
import mne
from tqdm import tqdm
from meg_decoding.matlab_utils.load_meg import get_meg_data, roi, time_window, get_baseline
import tqdm


mne.set_log_level(verbose="WARNING")


class GODDatasetBase(Dataset):
    def __init__(self, args, split, preprocess_pipleine:list=[]):
        self.args = args
        self.sub_id_map = {s:i for i, s in enumerate(list(self.args.subjects.keys()))}
        self.preprocess_pipeline = preprocess_pipleine
        # prepare dataset
        meg_epochs, sub_epochs, label_epochs, \
            image_feature_epochs = self.prepare_data(args, split=split)

        self.X = meg_epochs.astype(np.float32) # epochs x ch x time_samples
        self.Y = image_feature_epochs.astype(np.float32) # epochs x dims
        self.subs = sub_epochs # epochs (x 1)
        self.labels = label_epochs # epochs (x 1)

        self.num_subjects = len(np.unique(self.subs))

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):  # NOTE: i is id of a speech segment
        x, y, s = self.X[i], self.Y[i], self.subs[i]
        return x, y, s

    def prepare_data(self, args, split:str):
        DATAROOT = args.data_root
        processed_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')
        label_path_pattern = os.path.join(DATAROOT, '{sub}/labels/{name}')
        trigger_meg_path_pattern = os.path.join(DATAROOT, '{sub}/trigger/{name}')
        processed_rest_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')

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
        for sub in pbar:
            pbar.set_description("load subject data -- current: {}".format(sub))
            fs = args.subjects[sub]['fs']
            for meg_name, label_name, trigger_name, rest_name in zip(args.subjects[sub][split]['mat'], args.subjects[sub][split]['labels'], args.subjects[sub][split]['trigger'], args.subjects[sub][split]['rest']):
                processed_meg_path = processed_meg_path_pattern.format(sub=sub, name=meg_name)
                label_path = label_path_pattern.format(sub=sub, name=label_name)
                trigger_path = trigger_meg_path_pattern.format(sub=sub, name=trigger_name)
                processed_rest_meg_path = processed_rest_meg_path_pattern.format(sub=sub, name=rest_name)
                if args.z_scoring:
                    rest_mean, rest_std = get_baseline(processed_rest_meg_path, fs, args.rest_duration)
                MEG_Data, image_features, labels, triggers = get_meg_data(processed_meg_path, label_path, trigger_path, rest_mean=rest_mean, rest_std=rest_std, split=split)
                ROI_MEG_Data = MEG_Data[roi_channels, :] #  num_roi_channels x time_samples
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
        return meg_epochs, sub_epochs, label_epochs, image_feature_epochs
