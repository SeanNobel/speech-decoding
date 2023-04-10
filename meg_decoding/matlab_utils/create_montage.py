import scipy.io
import scipy
import numpy as np
import os
import csv

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





if __name__ == '__main__':
    DATAROOT = '/work/project/MEG_GOD/GOD_dataset/'

    filename = os.path.join(DATAROOT, 'channel_ricoh_acc1.mat')
    savefile = './data/GOD/montage.csv'

    create_montage(filename, savefile)


