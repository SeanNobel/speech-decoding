import sys
sys.path.append('./')
from meg_decoding.matlab_utils.load_meg import get_meg_data
import scipy.io

def rest_data(meg_filepath, fs, duration):
    data = scipy.io.loadmat(meg_filepath)
    """
    dict_keys(['__header__', '__version__', '__globals__', 'F', 'Std',
    'Comment', 'ChannelFlag', 'Time', 'DataType', 'Device', 'nAvg',
    'Leff', 'Events', 'ColormapType', 'DisplayUnits', 'History'])
    """
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

def run(filepath_list, fs, duration):
    for filepath in filepath_list:
        mean, std = rest_data(filepath, fs, duration)
        print(filepath)
        print('mean:{}, std:{}'.format(mean.shape, std.shape))


if __name__ == '__main__':
    filepath_list = [
        '/work/project/MEG_GOD/GOD_dataset/sbj01/mat/data_rest001.mat',
        '/work/project/MEG_GOD/GOD_dataset/sbj01/mat/data_rest002.mat',
        '/work/project/MEG_GOD/GOD_dataset/sbj02/mat/data_rest001.mat',
        '/work/project/MEG_GOD/GOD_dataset/sbj02/mat/data_rest002.mat',
        '/work/project/MEG_GOD/GOD_dataset/sbj03/mat/data_rest001.mat',
        '/work/project/MEG_GOD/GOD_dataset/sbj03/mat/data_rest002.mat',
    ]
    fs = 1000
    duration = 60 #[s]

    run(filepath_list, fs, duration)