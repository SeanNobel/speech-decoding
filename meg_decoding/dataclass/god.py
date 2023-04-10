import os, sys
from re import sub
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Sampler, Dataset
import numpy as np
import pandas as pd
import glob
import json
from natsort import natsorted
import scipy.io
import mne, mne_bids
from tqdm import tqdm
import ast
from typing import Union
from termcolor import cprint
from pprint import pprint
from einops import rearrange
from sklearn.preprocessing import RobustScaler, StandardScaler

from speech_decoding.utils.wav2vec_util import load_wav2vec_model, getW2VLastFourLayersAvg

mne.set_log_level(verbose="WARNING")


class GODDatasetBase(Dataset):
    def __init__(self, args, preprocess_pipleine:list=[]):
        pass

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):  # NOTE: i is id of a speech segment

        i_in_task, task = self.segment_to_task(i)

        key_no_task = np.random.choice(list(self.X.keys()))
        X = self.X[key_no_task][task]  # ( 208, ~100000 )
        onset = self.meg_onsets[key_no_task][task][i_in_task]  # scalar

        # NOTE: Extract MEG segment. Doing this to save memory from overlapping segments
        X = X[:, onset : onset + self.seq_len_samp]  # ( 208, 360 )

        subject_idx = np.where(self.valid_subjects == key_no_task.split("_")[0])[0][0]

        return X, self.Y[i], subject_idx