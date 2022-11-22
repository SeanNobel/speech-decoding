import os, sys
from re import sub
import torch
import torchaudio
import torchaudio.functional as F
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
from utils.bcolors import cyan, yellow
from utils.wav2vec_util import load_wav2vec_model, getW2VLastFourLayersAvg
from utils.preproc_utils import baseline_correction_single, baseline_correction
from utils.preproc_utils import scaleAndClamp_single, scaleAndClamp
from termcolor import cprint
from pprint import pprint
from einops import rearrange
from sklearn.preprocessing import RobustScaler, StandardScaler

mne.set_log_level(verbose="WARNING")


class Brennan2018Dataset(torch.utils.data.Dataset):

    def __init__(self, args):
        super().__init__()

        self.seq_len_sec = args.preprocs["seq_len_sec"]
        self.baseline_len_sec = args.preprocs["baseline_len_sec"]
        self.preceding_chunk_for_baseline = args.preprocs["preceding_chunk_for_baseline"]
        self.clamp = args.preprocs["clamp"]
        self.clamp_lim = args.preprocs["clamp_lim"]

        wav2vec_model = args.wav2vec_model
        force_recompute = args.force_recompute,
        last4layers = args.preprocs["last4layers"]
        mode = args.preprocs["mode"]
        self.subject_wise = args.preprocs['subject_wise']

        Y_path = f"data/Brennan2018/Y_embeds/embd_{wav2vec_model}.pt"

        if (not os.path.exists(Y_path)) or force_recompute[0]:
            torch.save(self.audio_preproc(
                wav2vec_model,
                last4layers=last4layers,
                mode=mode,
            ), Y_path)

        self.Y = torch.load(Y_path)  # load the upsampled (to 120 Hz) embeddings (of the entire recording)

        X_path = "data/Brennan2018/processed_X.pt"

        if (not os.path.exists(X_path)) or force_recompute[0]:
            cprint(f'Pre-processing EEG...', color='red')
            self.X, srate = self.brain_preproc(audio_embd_len=self.Y.shape[-1])
            torch.save({
                'X': self.X,
                'srate': srate,
            }, X_path)

        cprint(f'Loading preprocessed EEG...', color='green', attrs=['bold'])
        preprocessed_eeg = torch.load(X_path)
        self.X = preprocessed_eeg['X']
        srate = preprocessed_eeg['srate']  # ( 33, 60, 99712 )
        cprint(f"Using existing pre-processed data {self.X.shape}, srate={srate}", 'red', 'on_yellow')

        self.X, self.Y = self.shift_brain_signal(self.X, self.Y, srate=srate)

        cprint(f"X: {self.X.shape}, Y: {self.Y.shape}", color='red', attrs=['bold'])
        # X: ( 33, 60, 86791 ) -> ( B, 60, 1024 )
        # Y: ( 1024, 86791 ) -> ( B, 1024, 1024 ) # w2v embeddings

        # length of sequence in samples
        self.seq_len_samp = int(self.seq_len_sec * srate)

        # length of baseline period in samples
        self.baseline_len_samp = int(self.seq_len_samp * self.baseline_len_sec / self.seq_len_sec)

        cprint(f'Building batches of {self.seq_len_sec} s ({self.seq_len_samp} samples).', color='blue')
        self.X, self.Y, self.subject_idxs = self.batchfy(self.X, self.Y)
        cprint(f'X: {self.X.shape} | Y: {self.Y.shape} | {self.subject_idxs.shape}', color='blue')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idxs[i]

    def batchfy(self, X: torch.Tensor, Y: torch.Tensor):
        # NOTE: self.seq_len_samp is `bptt`

        assert X.shape[-1] == Y.shape[-1]
        trim_len = X.shape[-1] - X.shape[-1] % self.seq_len_samp

        X = X[:, :, :trim_len]  # ( 33, 60, 99584 ) (subj, chans, num_embeddings)
        # NOTE: the previous implementation was hard to understand and possibly wrong
        if not self.subject_wise:
            X = scaleAndClamp(X, self.clamp_lim, self.clamp)
        else:
            X = scaleAndClamp_single(X, self.clamp_lim, self.clamp)

        Y = Y[:, :trim_len]  # ( 512, 99584 )       (emsize, num_embeddings)

        X = X.reshape(X.shape[0], X.shape[1], -1, self.seq_len_samp)  # ( 8, 60, 243, 356 ) (sub, ch, chunks, bptt)
        if not self.subject_wise:
            X = baseline_correction(X, self.baseline_len_samp, self.preceding_chunk_for_baseline)
        else:
            X = baseline_correction_single(X, self.baseline_len_samp, self.preceding_chunk_for_baseline)

        Y = Y.reshape(Y.shape[0], -1, self.seq_len_samp)  # ( 1024, 243, 356 )  (emsize, chunks, bptt)

        Y = Y.unsqueeze(0).expand(X.shape[0], *Y.shape)  # ( 33, 512, 389, 256 )

        X = X.permute(0, 2, 1, 3)  # (8, 243, 60, 356) (sub, chunks, ch, bptt)
        Y = Y.permute(0, 2, 1, 3)  # (8, 243, 1024, 356) (sub, chunks, emsz, bptt)

        subject_idxs = torch.arange(X.shape[0]).unsqueeze(1).expand(-1, X.shape[1])  # ( 33, 389 )
        subject_idxs = subject_idxs.flatten()  # ( 19061, )

        X = X.reshape(-1, X.shape[-2], X.shape[-1])  # ( 19061, 60, 256 ) (samples, ch, emsize)
        Y = Y.reshape(-1, Y.shape[-2], Y.shape[-1])  # ( 19061, 512, 256 ) (samples, ch, emsize)

        return X, Y, subject_idxs

    @staticmethod
    def audio_preproc(
        wav2vec_model: str,
        last4layers: bool,
        mode: str,
    ):
        # waveform: ( 1, 31908132 ), sample_rate: 44100

        waveform, sample_rate = torchaudio.load("data/Brennan2018/merged_audio.wav")
        cprint(f"Audio before resampling: {waveform.shape}", color='yellow')  # shape of the original audio

        # NOTE: the base model was pre-trained on audio sampled @ 16kHz
        resample_rate = 16000
        waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=128)
        cprint(f"Audio after resampling: {waveform.shape}", color='red')  # shape of the resampled audio
        len_audio_s = waveform.shape[1] / resample_rate
        cprint(f"Audio length: {len_audio_s} s.", color='yellow')

        model = load_wav2vec_model(wav2vec_model)
        model.eval()

        # NOTE: for the large W2V2, the embedding dim is 1024
        if last4layers:
            cprint(f'Generating audio embeddings', 'yellow', 'on_red')
            embeddings = getW2VLastFourLayersAvg(model, waveform, mode=mode)  # torch.Size([1024, 36170])
        else:
            embeddings = model.feature_extractor(waveform).squeeze()  # (512, 36176 @16kHz) ( 512, 99712 @44.1kHz)

        embedding_srate = embeddings.shape[-1] / len_audio_s
        cprint(f'Original  embedding shape {embeddings.shape} | srate (out out w2v): {embedding_srate:.3f} Hz', 'red')

        res_embeddings = F.resample(embeddings, orig_freq=10, new_freq=24)  # to upsample from ~50 to ~120 Hz
        cprint(f'Resampled embedding shape {res_embeddings.shape} | srate: {120}', color='red', attrs=['bold'])

        # NOTE: "Paper says: we use standard normalization for both representations"
        scaler = StandardScaler().fit(res_embeddings.T)
        return torch.from_numpy(scaler.transform(res_embeddings.T).T).float()

    @staticmethod
    def brain_preproc(audio_embd_len):
        # NOTE: look at comprehension-scores.txt
        # excluded_subjects = [1, 6, 8, 22, 23, 26, 27, 28, 29, 30, 31, 32, 42, 45, 46, 48]

        matfile_paths = natsorted(glob.glob("data/Brennan2018/raw/*.mat"))
        # matfile_paths = [matfile_paths[i] for i in range(21) if not i in [1, 6, 8]]
        # matfile_paths = [matfile_paths[i] for i in [i for i in range(40)]]
        # cprint('using only subjects #[0, 1, 3, 4, 5, 6, 7, 48]', "blue", "on_yellow", attrs=['bold'])
        pprint(matfile_paths)
        # matfile_paths = np.delete(matfile_paths, excluded_subjects)

        # NOTE: find the shortest EEG and trim all EEG datasets to that length
        a = []
        pbar = tqdm(matfile_paths)
        for i, matfile_path in enumerate(pbar):
            mat_raw = scipy.io.loadmat(matfile_path)["raw"][0, 0]
            eeg_raw = mat_raw["trial"][0, 0][:60]  # drop non-EEG channels
            a.append(eeg_raw.shape)
        trim_eeg_to = np.stack(a)[:, 1].flatten().min()

        X = []
        pbar = tqdm(matfile_paths)
        for i, matfile_path in enumerate(pbar):
            pbar.set_description(f'Filtering subject {i} ')
            mat_raw = scipy.io.loadmat(matfile_path)["raw"][0, 0]
            eeg_raw = mat_raw["trial"][0, 0][:60, :trim_eeg_to]  # drop non-EEG channels
            fsample = mat_raw["fsample"][0, 0]  # 500 Hz
            assert fsample == 500, f"{matfile_path} has the wrong srate: {fsample}."
            # label = [e[0] for e in mat_raw["label"].squeeze()]

            eeg_filtered = mne.filter.filter_data(
                eeg_raw,
                sfreq=fsample,
                l_freq=1.0,
                h_freq=60,
            )

            # NOTE: This resamples EEG from 500Hz down to around 135Hz
            # NOTE: Two conditions must be met here: (1) that w2v and brain_encoder get the same length of data, AND (2) that the outputs of w2v and brain_encoder have the SAME dimension (this is required by CLIPLoss). Since the brain_encoder outputs the same number of time samples, we just need to resample EEG to so that the resampled EEG has the same number of time samples as the NUMBER of embeddings coming out of the FE.
            downsampling_factor = eeg_filtered.shape[-1] / audio_embd_len
            eeg_resampled = mne.filter.resample(
                eeg_filtered,
                down=downsampling_factor,
            )

            new_srate = fsample / downsampling_factor
            cprint(f'Downsampling EEG from {fsample} Hz to {new_srate:.4f} Hz', color='cyan')

            X.append(eeg_resampled)
        for i, x in enumerate(X):
            cprint(f"Samples in EEG DS {i}: {x.shape[-1]} | total wav embeddings: {audio_embd_len}", color='magenta')

        X = np.stack(X)
        # ( num_subjects, num_channels, num_embeddings ) *you get for the entire recording

        return torch.from_numpy(X).float(), new_srate

    @staticmethod
    def shift_brain_signal(X, Y, srate, shift_ms=150):
        """
        - X: ( 33, 60, 99712 ) Y: ( 512, 99712 )
        - resampled_rate (Hz): rates of M/EEG after resampling and speech after wav2vec2.0 encoding
        - shift (ms): how much to shift M/EEG forward
        """
        shift = int(srate * (shift_ms / 1000))  # 19

        X = X[:, :, shift:]  # ( 33, 60, 99692 )
        Y = Y[:, :-shift]  # ( 512, 99692 )

        return X, Y
