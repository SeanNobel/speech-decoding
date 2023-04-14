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

mne.set_log_level(verbose="WARNING")

from tqdm import tqdm
import ast
from typing import Union
from termcolor import cprint
from pprint import pprint
from einops import rearrange
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import List, Tuple

from transformers import Wav2Vec2Model

from speech_decoding.utils.wav2vec_util import get_last4layers_avg
from speech_decoding.utils.preproc_utils import (
    shift_brain_signal,
    continuous,
    baseline_correction,
    scale_and_clamp,
    pad_y_time,
    interpolate_y_time,
)
from speech_decoding.constants import BRAIN_RESAMPLE_RATE, AUDIO_RESAMPLE_RATE

# fmt: off
EXCLUDED_SUBJECTS = [
    "S02", "S07", "S09", "S23", "S24", "S27", "S28", "S29", "S30", "S31",
    "S32", "S33", "S43", "S46", "S47", "S49",
]
# fmt: on

"""
IMPORTANT NOTE:
As I couldn't find the information of word onsets relative to the EEG recording,
I'm implementing assuming that EEG recordings started at the exact same time as speech.
Also, dispite that the speech is split into multiple wave files, I assume that there's no
gap between them while presenting to the subjects.
"""


class Brennan2018Dataset(Dataset):
    def __init__(self, args, train=True):
        super().__init__()

        # Both
        force_recompute = args.rebuild_dataset
        self.root_dir = args.root_dir
        self.subject_wise = args.preprocs.subject_wise
        self.seq_len_sec = args.preprocs.seq_len_sec
        # EEG
        self.brain_filter_low = args.preprocs.brain_filter_low
        self.brain_filter_high = args.preprocs.brain_filter_high
        self.baseline_len_sec = args.preprocs.baseline_len_sec
        self.clamp = args.preprocs.clamp
        self.clamp_lim = args.preprocs.clamp_lim
        self.brain_num_samples = int(self.seq_len_sec * BRAIN_RESAMPLE_RATE)
        self.baseline_num_samples = int(self.baseline_len_sec * BRAIN_RESAMPLE_RATE)
        # Audio
        self.lowpass_filter_width = args.preprocs.lowpass_filter_width
        self.wav2vec = Wav2Vec2Model.from_pretrained(args.wav2vec_model)
        # Paths
        onsets_path = f"{self.root_dir}/data/Brennan2018/AliceChapterOne-EEG.csv"
        X_path = f"{self.root_dir}/data/Brennan2018/X.pt"
        Y_path = f"{self.root_dir}/data/Brennan2018/Y.pt"

        # Rebuild dataset
        if force_recompute or not (os.path.exists(X_path) and os.path.exists(Y_path)):
            cprint(f"> Preprocessing EEG and audio.", color="cyan")

            matfile_paths = natsorted(glob.glob(f"{self.root_dir}/data/Brennan2018/raw/*.mat"))
            audio_paths = natsorted(glob.glob(f"{self.root_dir}/data/Brennan2018/audio/*.wav"))

            X = self.brain_preproc(matfile_paths)
            audio = self.audio_preproc(audio_paths)

            X, audio = shift_brain_signal(
                X, audio, srate_x=BRAIN_RESAMPLE_RATE, srate_y=AUDIO_RESAMPLE_RATE
            )
            cprint(f">> X (EEG): {X.shape}, Audio: {audio.shape}", color="cyan")

            cprint("> Segmenting EEG and audio using onsets.", color="cyan")
            X, audio = self.segment(X, audio, onsets_path)
            cprint(f"X (EEG): {X.shape}, Audio: {audio.shape}", color="cyan")

            # NOTE: baseline corection becomes naturally subject-specific
            cprint("> Baseline correction EEG.", color="cyan")
            X = baseline_correction(X, self.baseline_num_samples)

            cprint("> Scaling and clamping EEG.", color="cyan")
            self.X = scale_and_clamp(X, self.clamp_lim)

            cprint("> Embedding audio with wave2vec2.0.", color="cyan")
            self.Y = self.embed_audio(audio)
            cprint(f"Y (audio): {self.Y.shape}", color="cyan")

            torch.save(self.X, X_path)
            torch.save(self.Y, Y_path)

        else:
            self.X = torch.load(X_path)
            self.Y = torch.load(Y_path)
            cprint(
                f"> Using pre-processed EEG {self.X.shape} | srate={BRAIN_RESAMPLE_RATE}",
                "cyan",
            )
            cprint(f"> Using pre-processed audio embeddings {self.Y.shape}", "cyan")

        self.num_subjects = self.X.shape[1]
        cprint(f"Number of subjects: {self.num_subjects}", color="cyan")

        if args.preprocs.y_upsample == "interpolate":
            self.Y = interpolate_y_time(self.Y, self.brain_num_samples)
        elif args.preprocs.y_upsample == "pad":
            self.Y = pad_y_time(self.Y, self.brain_num_samples)
        else:
            raise ValueError(f"Unknown upsampling strategy: {args.preprocs.y_upsample}")

    def embed_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: ( segment, 1, time@16kHz//segment )
        Returns:
            Y: Embedded (avg of last four activations) -> standard normalized
            | ( segment, features@w2v, time@w2v-freq//segment )
        """
        Y = []

        for segment in tqdm(audio):
            Y.append(get_last4layers_avg(self.wav2vec, segment).squeeze().T)

        return torch.stack(Y)

    def segment(
        self, X: torch.Tensor, audio: torch.Tensor, onsets_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X: ( subject, channel, time@120Hz )
            audio: ( 1, time@16kHz )
        Returns:
            X: ( segment, subject, channel, time@120Hz//segment )
            audio: ( segment, 1, time@16kHz//segment )
        """
        onsets = pd.read_csv(onsets_path).onset.to_numpy()
        onsets = continuous(onsets)

        # fmt: off
        X = torch.stack(
            [
                X[:, :, int(onset * BRAIN_RESAMPLE_RATE) : int((onset + self.seq_len_sec) * BRAIN_RESAMPLE_RATE)]
                for onset in onsets
            ]
        )

        audio = torch.stack(
            [
                audio[:, int(onset * AUDIO_RESAMPLE_RATE) : int((onset + self.seq_len_sec) * AUDIO_RESAMPLE_RATE)]
                for onset in onsets
            ]
        )
        # fmt: on

        return X, audio

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i, return_chunkids=True):
        random_subject = np.random.choice(self.num_subjects)

        if return_chunkids:
            return self.X[i, random_subject], self.Y[i], random_subject, i
        else:
            return self.X[i, random_subject], self.Y[i], random_subject

    def audio_preproc(self, audio_paths: List[str]) -> torch.Tensor:
        """
        Loads audio, resamples it to 16kHz, and extracts the last 4 layers
        of the pretrained wave2vec2.0 model.
        """
        # ----------------------
        #      Data Loading
        # ----------------------
        waveform = [torchaudio.load(path) for path in audio_paths]

        sample_rates = np.array([w[1] for w in waveform])
        # is all 44.1kHz
        assert np.all(sample_rates == sample_rates[0])
        sample_rate = sample_rates[0]

        # waveform: ( 1, 31908132 )
        waveform = torch.cat([w[0] for w in waveform], dim=1)

        cprint(
            f"Audio before resampling: {waveform.shape} | {sample_rate}Hz", color="cyan"
        )  # shape of the original audio

        # ----------------------
        #       Resampling
        # ----------------------
        # NOTE: the base model was pre-trained on audio sampled @ 16kHz
        waveform = F.resample(
            waveform,
            sample_rate,
            AUDIO_RESAMPLE_RATE,
            lowpass_filter_width=self.lowpass_filter_width,
        )
        cprint(
            f"Audio after resampling: {waveform.shape} | {AUDIO_RESAMPLE_RATE}Hz",
            color="cyan",
        )  # shape of the resampled audio
        len_audio_s = waveform.shape[1] / AUDIO_RESAMPLE_RATE
        cprint(f"Audio length: {len_audio_s} s.", color="cyan")

        return waveform

    def brain_preproc(self, matfile_paths: List[str]) -> torch.Tensor:
        """
        Returns: ( subject, channel, time@120Hz )
        """
        # NOTE: exclude bad subjects (look at comprehension-scores.txt)
        matfile_paths = [
            path for path in matfile_paths if not path.split(".")[0][-3:] in EXCLUDED_SUBJECTS
        ]

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

            # NOTE: in the paper they don't mention anything about filtering
            eeg_filtered = mne.filter.filter_data(
                eeg_raw,
                sfreq=fsample,
                l_freq=self.brain_filter_low,
                h_freq=self.brain_filter_high,
            )

            eeg_filtered = torch.from_numpy(eeg_filtered.astype(np.float32))

            # NOTE: This resamples EEG from 500Hz down to around 135Hz
            # NOTE: Two conditions must be met here: (1) that w2v and brain_encoder get the same length of data, AND (2) that the outputs of w2v and brain_encoder have the SAME dimension (this is required by CLIPLoss). Since the brain_encoder outputs the same number of time samples, we just need to resample EEG to so that the resampled EEG has the same number of time samples as the NUMBER of embeddings coming out of the FE.
            # downsampling_factor = eeg_filtered.shape[-1] / audio_embd_len
            # eeg_resampled = mne.filter.resample(
            #     eeg_filtered,
            #     down=downsampling_factor,
            # )
            # NOTE: in the paper they say they downsampled to exactly 120Hz with Torchaudio, so I'll stick to that
            eeg_resampled = F.resample(
                waveform=eeg_filtered,
                orig_freq=fsample,
                new_freq=BRAIN_RESAMPLE_RATE,
                lowpass_filter_width=self.lowpass_filter_width,  # TODO: check
            )

            X.append(eeg_resampled)

        return torch.stack(X)
