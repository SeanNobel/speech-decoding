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
from utils.bcolors import cyan, yellow
from utils.wav2vec_util import load_wav2vec_model, getW2VLastFourLayersAvg

# from utils.preproc_utils import scaleAndClamp_single, scaleAndClamp
from termcolor import cprint
from pprint import pprint
from einops import rearrange
from sklearn.preprocessing import RobustScaler, StandardScaler

mne.set_log_level(verbose="WARNING")


class Brennan2018Dataset(Dataset):

    def __init__(self, args, train=True):
        super().__init__()

        self.root_dir = args.root_dir
        self.wav2vec_model = args.wav2vec_model
        self.seq_len_sec = args.preprocs.seq_len_sec
        self.baseline_len_sec = args.preprocs.baseline_len_sec
        self.clamp = args.preprocs.clamp
        self.clamp_lim = args.preprocs.clamp_lim
        self.brain_filter_low = args.preprocs.brain_filter_low
        self.brain_filter_high = args.preprocs.brain_filter_high

        force_recompute = args.rebuild_dataset
        last4layers = args.preprocs.last4layers
        self.subject_wise = args.preprocs.subject_wise

        Y_path = f"{self.root_dir}/data/Brennan2018/Y_embeds/embd_wav2vec.pt"

        if (not os.path.exists(Y_path)) or force_recompute:
            torch.save(self.audio_preproc(last4layers=last4layers), Y_path)

        # load the upsampled (to 120 Hz) embeddings (of the entire recording)
        self.Y = torch.load(Y_path)

        # load or rebuild the array of pre-processed EEG. Shape: (subj, chan, time)
        X_path = f"{self.root_dir}/data/Brennan2018/processed_X.pt"
        if (not os.path.exists(X_path)) or force_recompute:
            cprint(f"Pre-processing EEG...", color="red")
            self.X, srate = self.brain_preproc(self.Y.shape[-1])
            torch.save(
                {
                    "X": self.X,
                    "srate": srate,
                },
                X_path,
            )

        cprint(f"Loading preprocessed EEG...", color="green", attrs=["bold"])
        preprocessed_eeg = torch.load(X_path)
        self.X = preprocessed_eeg["X"]
        srate = preprocessed_eeg["srate"]  # ( 33, 60, 99712 )
        cprint(
            f"Using existing pre-processed data {self.X.shape}, srate={srate}",
            "red",
            "on_yellow",
        )

        self.num_subjects = self.X.shape[0]
        cprint(f"Number of subjects: {self.num_subjects}", color="yellow")

        self.X, self.Y = self.shift_brain_signal(self.X, self.Y, srate=srate)

        cprint(
            f"X (EEG): {self.X.shape}, Y (audio embeds): {self.Y.shape}",
            color="red",
            attrs=["bold"],
        )
        # X: ( 33, 60, 86791 ) -> ( B, 60, 1024 )
        # Y: ( 1024, 86791 ) -> ( B, 1024, 1024 ) # w2v embeddings

        # length of sequence in samples
        self.seq_len_samp = int(self.seq_len_sec * srate)

        # length of baseline period in samples
        self.baseline_len_samp = int(self.seq_len_samp * self.baseline_len_sec / self.seq_len_sec)

        # compute the number if samples that divide evenly by the number of samples in 1 segement
        trim_len = (self.X.shape[-1] // self.seq_len_samp) * self.seq_len_samp

        # compute the number of segements in the entire dataset
        num_segments = trim_len // self.seq_len_samp

        # trim the length of EEG and embeddings so that they can be evenly divided by num time samp in 1 segment
        self.X = self.X[..., :trim_len]
        self.Y = self.Y[..., :trim_len]

        # scale and clamp either all the subejects taken together, or each subject's data separately
        self.X = self.scaleAndClamp()

        # make segments
        # NOTE: now X is a tuple of 358 matrices of size torch.Size([subj, ch, time]))
        self.X = self.X.split(num_segments, dim=-1)
        self.Y = self.Y.split(num_segments, dim=-1)

        # NOTE: baseline corection becomes naturally subject-specific
        self.X = self.baseline_correction()

    def scaleAndClamp(self):
        """
        returns:
            X (size=subj, chan, time) scaled channel-wise, subject-wise and clamped 
        """
        if self.subject_wise:
            res = []
            for subjID in range(self.X.shape[0]):
                scaler = RobustScaler().fit(self.X[subjID, :, :].T)  # NOTE: must be samples x features
                _X = torch.from_numpy(scaler.transform(self.X[subjID, :, :].T)).to(
                    torch.float)  # must be samples x features !!!
                if self.clamp:
                    _X.clamp_(min=-self.clamp_lim, max=self.clamp_lim)
                res.append(_X.to(torch.float))
            return torch.stack(res).permute(0, 2, 1)  # NOTE: make (subj, ch, time) again
        else:
            num_subjects = self.X.shape[0]
            T = rearrange(self.X, "s c t -> (t s) c")  # flatten subjects
            T = RobustScaler().fit_transform(T)  # NOTE: must be samples x features
            T = torch.from_numpy(T).float()
            if self.clamp:
                T.clamp_(min=-self.clamp_lim, max=self.clamp_lim)
            return rearrange(T, "(t s) c -> s c t", s=num_subjects)

    def baseline_correction(self):
        baseline_corrected_X = []
        # NOTE: now X is a tuple of 358 matrices of size torch.Size([subj, ch, time]))
        for chunk_id in range(len(self.X)):
            baseline = self.X[chunk_id][..., :self.baseline_len_samp].mean(axis=-1, keepdim=True)
            baseline_corrected_X.append(self.X[chunk_id] - baseline)
        return baseline_corrected_X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i, return_chunkids=True):
        random_subject = np.random.choice(self.num_subjects)
        if return_chunkids:
            return self.X[i][random_subject], self.Y[i], random_subject, i
        else:
            return self.X[i][random_subject], self.Y[i], random_subject

    def audio_preproc(self, last4layers: bool):
        audio_paths = natsorted(glob.glob(f"{self.root_dir}/data/Brennan2018/audio/*.wav"))
        waveform = [torchaudio.load(path) for path in audio_paths]

        sample_rates = np.array([w[1] for w in waveform])
        # is all 44.1kHz
        assert np.all(sample_rates == sample_rates[0])
        sample_rate = sample_rates[0]

        # waveform: ( 1, 31908132 )
        waveform = torch.cat([w[0] for w in waveform], dim=1)

        cprint(f"Audio before resampling: {waveform.shape}", color="yellow")  # shape of the original audio

        # NOTE: the base model was pre-trained on audio sampled @ 16kHz
        resample_rate = 16000
        waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=128)
        cprint(f"Audio after resampling: {waveform.shape}", color="red")  # shape of the resampled audio
        len_audio_s = waveform.shape[1] / resample_rate
        cprint(f"Audio length: {len_audio_s} s.", color="yellow")

        model = load_wav2vec_model(self.wav2vec_model)
        model.eval()

        # NOTE: for the large W2V2, the embedding dim is 1024
        if last4layers:
            cprint(f"Generating audio embeddings", "yellow", "on_red")
            embeddings = getW2VLastFourLayersAvg(model, waveform)  # torch.Size([1024, 36170])
        else:
            embeddings = model.feature_extractor(waveform).squeeze()  # (512, 36176 @16kHz) ( 512, 99712 @44.1kHz)

        embedding_srate = embeddings.shape[-1] / len_audio_s
        cprint(
            f"Original  embedding shape {embeddings.shape} | srate (out out w2v): {embedding_srate:.3f} Hz",
            "red",
        )

        res_embeddings = mne.filter.resample(
            embeddings.numpy().astype(np.float64),
            up=2.4,  # FIXME: this upsamling factor must be computed, not hard-coded
            axis=-1,
        )
        cprint(
            f"Resampled embedding shape {res_embeddings.shape} | srate: {120}",
            color="red",
            attrs=["bold"],
        )

        # NOTE: "Paper says: we use standard normalization for both representations"
        # scaler = StandardScaler().fit(res_embeddings.T)
        # return torch.from_numpy(scaler.transform(res_embeddings.T).T).float()

        return torch.from_numpy(res_embeddings).float()

    def brain_preproc(self, audio_embd_len):
        # NOTE: look at comprehension-scores.txt
        excluded_subjects = [
            "S02",
            "S07",
            "S09",
            "S23",
            "S24",
            "S27",
            "S28",
            "S29",
            "S30",
            "S31",
            "S32",
            "S33",
            "S43",
            "S46",
            "S47",
            "S49",
        ]
        MP = []
        matfile_paths = natsorted(glob.glob(f"{self.root_dir}/data/Brennan2018/raw/*.mat"))

        for i in matfile_paths:
            if not i.split(".")[0][-3:] in excluded_subjects:
                MP.append(i)
        matfile_paths = MP

        pprint(matfile_paths)

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
            pbar.set_description(f"Filtering subject {i} ")
            mat_raw = scipy.io.loadmat(matfile_path)["raw"][0, 0]
            eeg_raw = mat_raw["trial"][0, 0][:60, :trim_eeg_to]  # drop non-EEG channels
            fsample = mat_raw["fsample"][0, 0]  # 500 Hz
            assert fsample == 500, f"{matfile_path} has the wrong srate: {fsample}."
            # label = [e[0] for e in mat_raw["label"].squeeze()]

            eeg_filtered = mne.filter.filter_data(
                eeg_raw,
                sfreq=fsample,
                l_freq=self.brain_filter_low,
                h_freq=self.brain_filter_high,
            )

            # NOTE: This resamples EEG from 500Hz down to around 135Hz
            # NOTE: Two conditions must be met here: (1) that w2v and brain_encoder get the same length of data, AND (2) that the outputs of w2v and brain_encoder have the SAME dimension (this is required by CLIPLoss). Since the brain_encoder outputs the same number of time samples, we just need to resample EEG to so that the resampled EEG has the same number of time samples as the NUMBER of embeddings coming out of the FE.
            downsampling_factor = eeg_filtered.shape[-1] / audio_embd_len
            eeg_resampled = mne.filter.resample(
                eeg_filtered,
                down=downsampling_factor,
            )

            new_srate = fsample / downsampling_factor
            cprint(
                f"Downsampling EEG from {fsample} Hz to {new_srate:.4f} Hz",
                color="cyan",
            )

            X.append(eeg_resampled)
        for i, x in enumerate(X):
            cprint(
                f"Samples in EEG DS {i}: {x.shape[-1]} | total wav embeddings: {audio_embd_len}",
                color="magenta",
            )

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
