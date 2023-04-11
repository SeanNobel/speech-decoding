import scipy.io
import scipy
import numpy as np
import os
import csv
import json

def create_montage(filename:str, savefile:str):
    data = scipy.io.loadmat(filename)
    channel_infos = data['Channel'][0]
    n_ch = len(channel_infos)
    assert n_ch == 203, 'tell Yang-san'
    montage_list = []
    for i in range(n_ch):
        loc_array = channel_infos[i][4]
        if len(loc_array) > 0:
            assert loc_array.ndim == 2, 'MEG montage file requires 3x8 corrdinates for cube.'
            montage_list.append(list(np.mean(loc_array, axis=1)))

    with open(savefile, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(montage_list)
    print('montage file is saved as ', savefile)

def create_ch_region_pair(filename:str, savefile):
    data = scipy.io.loadmat(filename)
    test_list = []
    region_ch = {}
    for k in data.keys():
        if '__' in k:
            continue
        region_ch[k] = {}
        sub_region = data[k][0].dtype.names
        for i, sr in enumerate(sub_region):
            # import pdb; pdb.set_trace()
            region_ch[k][sr] = [int(i) for i in data[k][0][0][i][0].astype(np.int64)]
            test_list += region_ch[k][sr]
        print(k, sr, data[k][0][0][i][0])
    assert len(test_list) == 160
    assert len(np.unique(test_list)) == 160

    with open(savefile, 'w') as f:
        json.dump(region_ch, f, indent=4)
    print('ch_region pair is saved as ', savefile)

if __name__ == '__main__':
    DATAROOT = '/work/project/MEG_GOD/GOD_dataset/'

    # filename = os.path.join(DATAROOT, 'channel_ricoh_acc1.mat')
    # savefile = './data/GOD/montage.csv'
    # create_montage(filename, savefile)

    filename = '/work/project/MEG_GOD/GOD_dataset/channel_index.mat'
    savefile = './data/GOD/ch_region.json'
    create_ch_region_pair(filename, savefile)

