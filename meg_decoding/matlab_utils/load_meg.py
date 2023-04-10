import scipy.io
import scipy
import numpy as np
import os
import csv
import numpy as np
from hydra import initialize, compose


def get_meg_data(args, meg_filepath:str, label_filepath:str, trigger_filepath:str):
    data = scipy.io.loadmat(meg_filepath)
    """
    dict_keys(['__header__', '__version__', '__globals__', 'F', 'Std',
    'Comment', 'ChannelFlag', 'Time', 'DataType', 'Device', 'nAvg',
    'Leff', 'Events', 'ColormapType', 'DisplayUnits', 'History'])
    """
    MEG_Data = data['F'] # 203 x 391000 = ch x time sample
    assert len(MEG_Data) == 203
    if args.fs is None:
        sampling_rate = MEG_Data.shape[1] / data['Time'] # 1000 Hz
    else:
        sampling_rate = args.fs

    # Event_Data = data['Events'][0]
    # assert Event_Data[1][0][0] == 'visual', '{}'.format(Event_Data[0][1][0][0])
    # onset_timing = Event_Data[1][3][0] # trial
    # assert len(onset_timing) == 600 # GOD is 600

    Label_Data = scipy.io.loadmat(label_filepath)
    """
    dict_keys(['__header__', '__version__', '__globals__', 'vec_image', 'vec_index'])
    """
    image_features = Label_Data['vec_image']
    assert image_features.shape == (600, 512) # n_samples x image_feature_dims
    labels = Label_Data['vec_index'][0] # index of image
    assert len(labels) == 600 # n_samples

    Trigger_Data = scipy.io.loadmat(trigger_filepath)
    """
    dict_keys(['__header__', '__version__', '__globals__', 'trigger'])
    """
    triggers = Trigger_Data['trigger'][0]
    assert len(triggers) == 600 # stimulus onset timing



    import pdb; pdb.set_trace()

if __name__ == '__main__':
    DATAROOT = '/work/project/MEG_GOD/GOD_dataset/'

    with initialize(version_base=None, config_path="../../../configs/"):
        args = compose(config_name='analysis')
    processed_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{name}')
    label_path_pattern = os.path.join(DATAROOT, '{sub}/labels/{name}')
    trigger_meg_path_pattern = os.path.join(DATAROOT, '{sub}/trigger/{name}')

    sub_list = list(args.subjects.keys())
    split = 'train'

    for sub in sub_list:
        for meg_name, label_name, trigger_name in zip(args.subjects[sub][split]['mat'], args.subjects[sub][split]['labels'], args.subjects[sub][split]['trigger']):
            processed_meg_path = processed_meg_path_pattern.format(sub=sub, name=meg_name)
            label_path = label_path_pattern.format(sub=sub, name=label_name)
            trigger_path = trigger_meg_path_pattern.format(sub=sub, name=trigger_name)
            get_meg_data(args, processed_meg_path, label_path, trigger_path)