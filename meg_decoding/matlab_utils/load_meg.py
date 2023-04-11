import scipy.io
import scipy
import numpy as np
import os
import csv
import numpy as np
from hydra import initialize, compose
import json
from typing import List


def get_baseline(meg_filepath, fs, duration):
    print('baseline: ', meg_filepath)
    data = scipy.io.loadmat(meg_filepath)
    MEG_Data = data['F'] # 203 x 391000 = ch x time sample
    assert len(MEG_Data) == 203

    Event_Data = data['Events'][0]
    for i in range(len(Event_Data)):
        if Event_Data[i][0][0] == 'visual':
            visual_id = i
            break
    assert Event_Data[visual_id][0][0] == 'visual', '{}'.format(Event_Data[0][1][0][0])
    onset_timing = Event_Data[visual_id][3][0] # trial
    assert len(onset_timing) == 60


    start_point = int(onset_timing[-1]*fs)
    end_point = start_point + int(duration*fs)
    rest_data = MEG_Data[:, start_point:end_point]
    return rest_data.mean(axis=1), rest_data.std(axis=1)


def get_meg_data(meg_filepath:str, label_filepath:str, trigger_filepath:str,
                 rest_mean:np.ndarray=None, rest_std:np.ndarray=None, split='train'):
    """
    Args:
        meg_filepath(str): preprocessed MEG data path
        label_filepath(str): label (image_features, image_idx) path
        trigger_filepath(str): trigger onset path
    Return:
        MEG_Data(np.ndarray): processed MEG Data. size = n_ch(203) x n_time_samples
        image_features(np.ndarray): image feature. size = n_images(600) x latent_dims(512)
        labels(np.ndarray): image index. size = n_images
        triggers(np.ndarray): stimulus onset timing [s] size = n_images

    """
    print('load ', meg_filepath)
    data = scipy.io.loadmat(meg_filepath)
    """
    dict_keys(['__header__', '__version__', '__globals__', 'F', 'Std',
    'Comment', 'ChannelFlag', 'Time', 'DataType', 'Device', 'nAvg',
    'Leff', 'Events', 'ColormapType', 'DisplayUnits', 'History'])
    """
    MEG_Data = data['F'] # 203 x 391000 = ch x time sample
    assert len(MEG_Data) == 203
    # REMOVE BASELINE
    if rest_mean is not None:
        MEG_Data -= rest_mean[:, np.newaxis]
    if rest_std is not None:
        MEG_Data /= rest_std[:, np.newaxis]
    # if args.fs is None:
    #     sampling_rate = MEG_Data.shape[1] / data['Time'] # 1000 Hz
    # else:
    #     sampling_rate = args.fs

    # Event_Data = data['Events'][0]
    # assert Event_Data[1][0][0] == 'visual', '{}'.format(Event_Data[0][1][0][0])
    # onset_timing = Event_Data[1][3][0] # trial
    # assert len(onset_timing) == 600 # GOD is 600

    Label_Data = scipy.io.loadmat(label_filepath)
    """
    dict_keys(['__header__', '__version__', '__globals__', 'vec_image', 'vec_index'])
    """
    image_features = Label_Data['vec_image']
    if split == 'train':
        assert image_features.shape == (600, 512), '{}, {}'.format(image_features.shape[0], image_features.shape[1]) # n_samples x image_feature_dims
    elif split == 'test':
        assert image_features.shape == (50, 512), '{}, {}'.format(image_features.shape[0], image_features.shape[1]) # n_samples x image_feature_dims
    elif split == 'rest':
        assert image_features.shape == (60, 512), '{}, {}'.format(image_features.shape[0], image_features.shape[1]) # n_samples x image_feature_dims
    labels = Label_Data['vec_index'][0] # index of image
    if split == 'train':
        assert len(labels) == 600 # n_samples
    elif split == 'test':
        assert len(labels) == 50 # n_samples
    elif split == 'rest':
        assert len(labels) == 60

    Trigger_Data = scipy.io.loadmat(trigger_filepath)
    """
    dict_keys(['__header__', '__version__', '__globals__', 'trigger'])
    """
    triggers = Trigger_Data['trigger'][0]
    if split == 'train':
        assert len(triggers) == 600 # stimulus onset timing
    elif split == 'test':
        assert len(triggers) == 50
    elif split == 'rest':
        assert len(triggers) == 60

    return MEG_Data, image_features, labels, triggers

def roi(args)->list:
    region:list = args.region
    with open(args.ch_region_path, 'r') as f:
        ch_region_info = json.load(f)
    roi_channels = []
    for reg in region:
        reg_subreg = reg.split('/')
        assert len(reg_subreg) == 2
        r = reg_subreg[0]
        s = reg_subreg[1]
        roi_channels += ch_region_info[r][s]
    print('ROI: ', [reg for reg in region])
    print('channel: ', roi_channels)
    print('num channels: ', len(roi_channels))
    return roi_channels


def time_window(args, trigger:np.ndarray, fs:float, mode='')->List[tuple]:
    trigger_point =  np.round(trigger * fs)
    start_point = np.round(args.window.start * fs)
    end_point = np.round(args.window.end * fs)

    point_set = [(int(t + start_point), int(t + end_point)) for t in trigger_point]

    return point_set


def read_montage(args):
    # montage
    montage = []
    print('read montage file ', args.montage_path)
    with open(args.montage_path) as f:
        reader = csv.reader(f)
        for row in reader:
            montage.append([float(r) for r in row])
    montage = np.array(montage)
    roi_channels = roi(args)
    return montage[roi_channels,:] # ch x 3





if __name__ == '__main__':
    DATAROOT = '/work/project/MEG_GOD/GOD_dataset/'

    with initialize(version_base=None, config_path="../../../configs/"):
        args = compose(config_name='test')
    processed_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')
    label_path_pattern = os.path.join(DATAROOT, '{sub}/labels/{name}')
    trigger_meg_path_pattern = os.path.join(DATAROOT, '{sub}/trigger/{name}')
    processed_rest_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')

    sub_list = list(args.subjects.keys())
    split = 'train'

    roi_channels = roi(args)

    for sub in sub_list:
        fs = args.subjects[sub]['fs']
        for meg_name, label_name, trigger_name, rest_name in zip(args.subjects[sub][split]['mat'], args.subjects[sub][split]['labels'], args.subjects[sub][split]['trigger'], args.subjects[sub][split]['rest']):
            processed_meg_path = processed_meg_path_pattern.format(sub=sub, name=meg_name)
            label_path = label_path_pattern.format(sub=sub, name=label_name)
            trigger_path = trigger_meg_path_pattern.format(sub=sub, name=trigger_name)
            processed_rest_meg_path = processed_rest_meg_path_pattern.format(sub=sub, name=rest_name)
            rest_mean, rest_std = get_baseline(processed_rest_meg_path, fs, args.rest_duration)
            MEG_Data, image_features, labels, triggers = get_meg_data(processed_meg_path, label_path, trigger_path, rest_mean=rest_mean, rest_std=rest_std)
            ROI_MEG_Data = MEG_Data[roi_channels, :] #  num_roi_channels x time_samples
            window = time_window(args, triggers, fs)

