import os, sys, shutil
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import glob
import json
from natsort import natsorted
import scipy.io
import mne, mne_bids
from tqdm import tqdm
import ast
from typing import Union, Tuple
from psutil import virtual_memory as vm

from utils.wav2vec_util import load_wav2vec_model, getW2VLastFourLayersAvg
from utils.preproc_utils import (
    check_preprocs,
    scaleAndClamp_single,
    baseline_correction_single,
)
from termcolor import cprint
from pprint import pprint
from einops import rearrange
from constants import bar_format
from multiprocessing import Pool, Manager
from itertools import repeat
from pprint import pprint
from omegaconf import open_dict

mne.set_log_level(verbose="WARNING")

manager = Manager()
# glob_real_durations = manager.dict()
global_meg_onsets = manager.dict()
global_speech_onsets = manager.dict()


def to_second(onset):  # pandas Timestamp object
    return onset.minute * 60 + onset.second + onset.microsecond * 1e-6

def get_speech_onsets(df_annot):
    """
    Increment speech onsets that come for each separate audio file.
    """
    df_annot = pd.DataFrame(df_annot.description.apply(eval).to_list())

    _speech_onsets = df_annot['start'].to_numpy()
    speech_onsets = np.zeros_like(_speech_onsets)
    base = 0
    
    for i in range(len(speech_onsets)):
        speech_onsets[i] = base + _speech_onsets[i]
        # NOTE: avoid idx out of range at the final idx
        try:
            if _speech_onsets[i+1] < _speech_onsets[i]:
                base += _speech_onsets[i]
        except:
            pass
        
    return speech_onsets


class Gwilliams2022Dataset(Dataset):

    def __init__(self, args, train=True):
        super().__init__()
        
        self.train = train
        self.split_ratio = args.split_ratio
        
        self.wav2vec_model = args.wav2vec_model
        self.root_dir = args.root_dir + "/data/Gwilliams2022/"
        self.brain_orig_rate = 1000
        self.brain_resample_rate = args.preprocs["brain_resample_rate"]
        self.brain_filter_low = args.preprocs["brain_filter_low"]
        self.brain_filter_high = args.preprocs["brain_filter_high"]
        self.seq_len_samp = self.brain_resample_rate * args.preprocs["seq_len_sec"]
        self.clamp = args.preprocs["clamp"]
        self.clamp_lim = args.preprocs["clamp_lim"]
        self.baseline_len_samp = int(self.brain_resample_rate * args.preprocs["baseline_len_sec"])

        self.audio_resample_rate = args.preprocs["audio_resample_rate"]
        self.lowpass_filter_width = args.preprocs["lowpass_filter_width"]
        self.last4layers = args.preprocs["last4layers"]

        self.shift_brain = args.preprocs["shift_brain"]
        self.shift_len = args.preprocs["shift_len"]

        # NOTE: x_done and y_done are added to args.preprocs
        args, self.preproc_dir = check_preprocs(
            args,
            self.root_dir + "preprocessed/",
        )
        self.x_path = self.preproc_dir + "x_dict.npy"
        self.y_path = self.preproc_dir + "y_dict.npy"
        self.meg_onsets_path = self.preproc_dir + "meg_onsets.npy"
        self.speech_onsets_path = self.preproc_dir + "speech_onsets.npy"

        self.task_prefixes = ["lw", "cable", "easy", "the"]

        # ---------------------
        #     Make X (MEG)
        # ---------------------
        if args.rebuild_dataset or not args.preprocs["x_done"]:
            self.meg_onsets = {}    # will be updated in brain_preproc
            self.speech_onsets = {} # will be updated in brain_preproc
            self.X = self.brain_preproc_handler()
            
            np.save(self.meg_onsets_path, self.meg_onsets)
            np.save(self.speech_onsets_path, self.speech_onsets)
            np.save(self.x_path, self.X)
            
            # Record done
            args.preprocs.update({"x_done": True})
            with open(self.preproc_dir + "settings.json", "w") as f:
                json.dump(dict(args.preprocs), f)
        else:
            self.meg_onsets = np.load(self.meg_onsets_path, allow_pickle=True).item()
            self.speech_onsets = np.load(self.speech_onsets_path, allow_pickle=True).item()
            self.X = np.load(self.x_path, allow_pickle=True).item()

        # ----------------------------------
        #      Make Y (embedded speech)
        # ----------------------------------
        if args.rebuild_dataset or not args.preprocs["y_done"]:
            self.Y = self.audio_preproc()
            
            np.save(self.y_path, self.Y)
            
            # Record done
            with open_dict(args):
                args.preprocs.y_done = True
            with open(self.preproc_dir + "settings.json", "w") as f:
                json.dump(dict(args.preprocs), f)
        else:
            self.Y = np.load(self.y_path, allow_pickle=True).item()
            # dict_keys(['task0', 'task1', 'task2', 'task3'])

        self.X, self.Y, self.meg_onsets, self.num_segments_foreach_task = self.batchfy()
        
        assert len(self.X) == len(self.meg_onsets)
        # assert sum([v.shape[0] for v in list(self.meg_onsets.values())]) % self.Y.shape[0] == 0

        self.valid_subjects = np.array(list(set([k.split("_")[0] for k in self.X.keys()])))
        self.num_subjects = len(self.valid_subjects)
        
        cprint(f"X keys: {self.X.keys()}", color='cyan')
        cprint(f"Y: {self.Y.shape}", color='cyan')
        cprint(f"num_subjects: {self.num_subjects} (each has 2 or 1 sessions)", color='cyan')
        print(self.valid_subjects)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):  # NOTE: i is id of a speech segment
        
        i_in_task, task = self.segment_to_task(i)
        
        key_no_task = np.random.choice(list(self.X.keys()))
        X = self.X[key_no_task][task] # ( 208, ~100000 )
        onset = self.meg_onsets[key_no_task][task][i_in_task] # scalar
        
        # Extract MEG segment
        X = X[:, onset:onset+self.seq_len_samp] # ( 208, 360 )
        
        # TODO: batch baseline correction in collator
        X = baseline_correction_single(
            X.unsqueeze(0), self.baseline_len_samp
        ).squeeze()
        
        subject_idx = np.where(self.valid_subjects == key_no_task.split("_")[0])[0][0]

        return X, self.Y[i], subject_idx

    def segment_to_task(self, i) -> Tuple[int, str]:
        
        nseg_task_accum = np.cumsum(self.num_segments_foreach_task)
        task = np.searchsorted(nseg_task_accum, i+1)
        
        i_in_task = i - np.sum(self.num_segments_foreach_task[:task])
        
        return int(i_in_task), f"task{task}"

    def segment_speech(self, data: torch.Tensor, key: str) -> torch.Tensor:

        onsets = self.speech_onsets[key]
            
        onsets = (onsets * self.brain_resample_rate).round().astype(int)

        data = [data[:, onset:onset+self.seq_len_samp] for onset in onsets]

        return torch.stack(data)

    def batchfy(self):
        # self.X.keys() -> ['subject01_sess0_task0', ..., 'subject27_sess1_task3']
        # self.Y.keys() -> ['task0', 'task1', 'task2', 'task3']
        assert natsorted(self.X.keys()) == list(self.X.keys()), "self.X.keys() not sorted"

        # ----------------------------------------------------
        #    Make Y (speech are concatenated along tasks)
        # ----------------------------------------------------
        cprint("=> Batchfying Y", color="cyan")

        Y_list = []

        for key, Y in tqdm(self.Y.items()):
            # Y: ( F=1024, len=37835 )
            if self.shift_brain:
                # NOTE: This doesn't shift audio. Just crops the end.
                Y = self.shift_brain_signal(Y, is_Y=True)

            Y = torch.from_numpy(Y.astype(np.float32))

            Y = self.segment_speech(Y, key)  # ( num_segment=~2000, F=1024, len=360 )
            
            if self.train:
                Y = Y[:int(len(Y) * self.split_ratio)]
            else:
                Y = Y[int(len(Y) * self.split_ratio):]

            Y_list.append(Y)
        
        num_segments_foreach_task = [len(y) for y in Y_list]
                        
        # ------------------------------------
        #      More preprocessing for MEG
        # ------------------------------------
        cprint("=> Segmenting X", color="cyan")
        
        self.drop_task_missing_sessions() # self.X.keys() 167 -> 156
        assert len(self.X.keys()) == len(self.meg_onsets.keys())
        assert len(self.X.keys()) % 4 == 0
        
        X_dict = {}
        meg_onsets_dict = {}

        for key, X in tqdm(self.X.items()):
            # NOTE: e.g. 'subject01_sess0_task0' -> 'task0', 'subject01_sess0'
            key_task = key.split("_")[-1]
            key_no_task = "_".join(key.split("_")[:-1])

            if self.shift_brain:
                X = self.shift_brain_signal(X, is_Y=False)
                
            X = scaleAndClamp_single(X, self.clamp_lim, self.clamp)  # ( ch=208, len~=100000 )
            
            
            meg_onsets = self.meg_onsets[key]
            # To idx in samples
            meg_onsets = (meg_onsets * self.brain_resample_rate).round().astype(int)
            
            if self.train:
                meg_onsets = meg_onsets[:int(len(meg_onsets) * self.split_ratio)]
            else:
                meg_onsets = meg_onsets[int(len(meg_onsets) * self.split_ratio):]
            
            if not key_no_task in X_dict.keys():
                X_dict[key_no_task] = {key_task: X}
                meg_onsets_dict[key_no_task] = {key_task: meg_onsets}
            else:
                X_dict[key_no_task].update({key_task: X})
                meg_onsets_dict[key_no_task].update({key_task: meg_onsets})

        return X_dict, torch.cat(Y_list), meg_onsets_dict, num_segments_foreach_task

    def shift_brain_signal(self, data, is_Y: bool):
        """
        Rates of X and Y should be 120Hz, meaning Y is processed by wav2vec then upsampled
        - shift (ms): how much to shift M/EEG forward
        """
        shift = int(self.brain_resample_rate * (self.shift_len / 1000))

        if is_Y:
            return data[:, :-shift]
        else:
            return data[:, shift:]
        
    def drop_task_missing_sessions(self) -> None:
        sess_strs = list(set(["_".join(key.split("_")[:-1]) for key in self.X.keys()]))

        for sess_str in sess_strs:
            if len([key for key in self.X.keys() if sess_str in key]) < 4:
                for key in list(self.X.keys()):
                    if sess_str in key:
                        self.X.pop(key)
                        self.meg_onsets.pop(key)

    @staticmethod
    def brain_preproc(dat):
        subject_idx, d, speech_onsets, meg_onsets, session_idx, task_idx = dat
        num_channels = d["num_channels"]
        brain_orig_rate = d["brain_orig_rate"]
        brain_filter_low = d["brain_filter_low"]
        brain_filter_high = d["brain_filter_high"]
        brain_resample_rate = d["brain_resample_rate"]
        root_dir = d["root_dir"]
        preproc_dir = d["preproc_dir"]

        description = (f"subject{str(subject_idx+1).zfill(2)}_sess{session_idx}_task{task_idx}")

        bids_path = mne_bids.BIDSPath(
            subject=str(subject_idx + 1).zfill(2),
            # '01', '02', ...
            session=str(session_idx),
            task=str(task_idx),
            datatype="meg",
            root=root_dir,
        )
        
        try:
            raw = mne_bids.read_raw_bids(bids_path)
        except:
            cprint("No .con data was found", color="yellow")
            return 1

        cprint(description, color="cyan")
        
        df = raw.to_data_frame()
        df_annot = raw.annotations.to_data_frame()
        df_annot_pd = pd.DataFrame(df_annot.description.apply(eval).to_list())
        
        words = np.where(df_annot_pd['kind'] == 'word')[0]
        
        _meg_onsets = np.array([to_second(onset) for onset in df_annot.onset])
        _speech_onsets = get_speech_onsets(df_annot)
        
        _meg_onsets, _speech_onsets = _meg_onsets[words], _speech_onsets[words]
        cprint(f"MEG onsets: {_meg_onsets.shape}, speech onsets: {_speech_onsets.shape}", color='cyan')

        task_str = f"task{task_idx}"
        # Ensure that speech onsets are same across subjects&sessions
        if task_str in speech_onsets.keys():
            assert np.allclose(speech_onsets[task_str], _speech_onsets), "Speech onsets are different"

        meg_onsets.update({description: _meg_onsets})
        speech_onsets.update({task_str: _speech_onsets})


        meg_raw = np.stack([df[key] for key in df.keys() if "MEG" in key])  # ( 224, 396000 )
        # NOTE: (kind of) confirmed that last 16 channels are REF
        meg_raw = meg_raw[:num_channels]  # ( 208, 396000 )
        
        meg_filtered = mne.filter.filter_data(
            meg_raw,
            sfreq=brain_orig_rate,
            l_freq=brain_filter_low,
            h_freq=brain_filter_high,
        )

        # To 120 Hz
        meg_resampled = mne.filter.resample(
            meg_filtered,
            down=brain_orig_rate / brain_resample_rate,
        )  # ( 208, 37853 )

        # scaler = RobustScaler().fit(meg_resampled)
        # meg_scaled = scaler.transform(meg_resampled)
        # cprint(meg_scaled.shape, color="cyan")

        # save to disk
        # X = np.load(x_path, allow_pickle=True).item()
        # X.update({description: meg_resampled})
        # np.save(x_path, X)

        np.save(
            f"{preproc_dir}_parts/{description}",
            meg_resampled,
        )
        return 0

    def brain_preproc_handler(self, num_subjects=27, num_channels=208):
        tmp_dir = self.preproc_dir + "_parts/"
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
            
        consts = dict(
            num_channels=num_channels,
            brain_orig_rate=self.brain_orig_rate,
            brain_filter_low=self.brain_filter_low,
            brain_filter_high=self.brain_filter_high,
            brain_resample_rate=self.brain_resample_rate,
            root_dir=self.root_dir,
            preproc_dir=self.preproc_dir,
        )

        subj_list = []
        for subj in range(num_subjects):
            for session_idx in range(2):
                for task_idx in range(4):
                    subj_list.append((
                        subj,
                        consts,
                        global_speech_onsets,
                        global_meg_onsets,
                        session_idx,
                        task_idx
                    ))

        # subj_list = [(i, c, rd) for i, c, rd in zip(range(num_subjects), repeat(consts), repeat(real_durations))]
        with Pool(processes=20) as p:
            res = list(tqdm(
                p.imap(Gwilliams2022Dataset.brain_preproc, subj_list),
                total=len(subj_list),
                bar_format=bar_format,
            ))
            
        cprint("MEG preprocessing done.", color='green')

        # NOTE: global shared struct
        self.meg_onsets = dict(global_meg_onsets)
        self.speech_onsets = dict(global_speech_onsets)

        # NOTE: assemble files into one and clean up
        fnames = natsorted(os.listdir(tmp_dir))
        # cprint(fnames, color='yellow')
        X = dict()
        for fname in fnames:  # NOTE: data MUST be task0, ... taskN, task0, ..., taskN (N=4)
            key = os.path.splitext(fname)[0]
            X[key] = np.load(tmp_dir + fname, allow_pickle=True)

        cprint("removing temp files for EEG data", color="white")
        shutil.rmtree(tmp_dir)

        return X


    @torch.no_grad()
    def audio_preproc(self):
        wav2vec = load_wav2vec_model(self.wav2vec_model)
        wav2vec.eval()

        Y = {}
        assert os.path.exists(f"{self.root_dir}stimuli/audio"), "The path `data/Gwilliams2022/stimuli/audio` DOESN'T EXIST."
        
        for task_idx in self.speech_onsets.keys():  # 4 tasks for each subject
            task_idx_ID = int(task_idx[-1])

            audio_paths = natsorted(glob.glob(
                f"{self.root_dir}stimuli/audio/{self.task_prefixes[task_idx_ID]}*.wav"
            ))

            audio_raw = []
            for path in audio_paths:
                waveform, sample_rate = torchaudio.load(path)

                # cutoff = int(sample_rate * self.real_durations[task_idx][f])
                # if waveform.shape[1] > cutoff:
                #     waveform = waveform[:, :cutoff]
                # else:
                #     print(yellow("No audio cutoff"))

                # Upsample to 16000Hz
                waveform = F.resample(
                    waveform,
                    orig_freq=sample_rate,
                    new_freq=self.audio_resample_rate,
                    lowpass_filter_width=self.lowpass_filter_width,
                )
                # cprint(f"Audio after resampling: {waveform.shape}", color="cyan")

                if self.last4layers:
                    embeddings = getW2VLastFourLayersAvg(wav2vec, waveform)
                else:
                    embeddings = wav2vec.feature_extractor(waveform).squeeze()
                # cprint(f"Audio embedding: {embeddings.shape}", color="cyan")

                rate_after_wav2vec = self.audio_resample_rate * embeddings.shape[-1] / waveform.shape[-1] # 49.9737...
                # cprint(rate_after_wav2vec, color="cyan")

                # NOTE: torchaudio resample doesn't accept float freqs
                embeddings = mne.filter.resample(
                    embeddings.numpy().astype(np.float64),
                    up=self.brain_resample_rate / rate_after_wav2vec, # To 120 Hz
                    axis=-1,
                )
                cprint(
                    f"Audio embedding upsampled to {self.brain_resample_rate}Hz: {embeddings.shape}",
                    color="cyan"
                )

                audio_raw.append(embeddings)

            audio_raw = np.concatenate(audio_raw, axis=-1)

            print(audio_raw.shape)

            Y.update({task_idx: audio_raw})

        return Y