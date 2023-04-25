import os, sys
from re import sub
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import glob
from natsort import natsorted
import scipy.io
import mne

mne.set_log_level(verbose="WARNING")

from tqdm import tqdm
import ast
from termcolor import cprint
from typing import List, Tuple, Union

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
# NOTE: exclude some subjects (look at comprehension-scores.txt)
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
gap between them while being presented to the subjects.
"""


class Brennan2018Dataset(Dataset):
    def __init__(self, args):
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
        # Data Paths
        self.matfile_paths = natsorted(glob.glob(f"{self.root_dir}/data/Brennan2018/raw/*.mat"))
        self.audio_paths = natsorted(glob.glob(f"{self.root_dir}/data/Brennan2018/audio/*.wav"))
        self.onsets_path = f"{self.root_dir}/data/Brennan2018/AliceChapterOne-EEG.csv"
        # Save Paths
        X_path = f"{self.root_dir}/data/Brennan2018/X.pt"
        Y_path = f"{self.root_dir}/data/Brennan2018/Y.pt"

        # Rebuild dataset
        if force_recompute or not (os.path.exists(X_path) and os.path.exists(Y_path)):
            cprint(f"> Preprocessing EEG and audio.", color="cyan")

            self.X, self.Y = self.rebuild_dataset(
                self.matfile_paths, self.audio_paths, self.onsets_path
            )

            torch.save(self.X, X_path)
            torch.save(self.Y, Y_path)

        # Load dataset
        else:
            self.X = torch.load(X_path)
            self.Y = torch.load(Y_path)
            cprint(
                f"> Using pre-processed EEG {self.X.shape} | srate={BRAIN_RESAMPLE_RATE}",
                color="cyan",
            )
            cprint(f"> Using pre-processed audio embeddings {self.Y.shape}", "cyan")

        self.num_subjects = self.X.shape[1]
        cprint(f">>> Number of subjects: {self.num_subjects}", color="cyan")

        cprint(f">> Upsampling audio embedding with: {args.preprocs.y_upsample}", color="cyan")
        if args.preprocs.y_upsample == "interpolate":
            self.Y = interpolate_y_time(self.Y, self.brain_num_samples)
        elif args.preprocs.y_upsample == "pad":
            self.Y = pad_y_time(self.Y, self.brain_num_samples)
        else:
            raise ValueError(f"Unknown upsampling strategy: {args.preprocs.y_upsample}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i, return_chunkids=True):
        random_subject = np.random.choice(self.num_subjects)

        if return_chunkids:
            return self.X[i, random_subject], self.Y[i], random_subject, i
        else:
            return self.X[i, random_subject], self.Y[i], random_subject

    def rebuild_dataset(self, matfile_paths: List[str], audio_paths: List[str], onsets_path: str):
        """
        Returns:
            X: ( segment, subject, channel, time@120Hz//segment )
            Y: ( segment, features@w2v, time@w2v-freq//segment )
        """
        # ----------------------
        #     Audio Loading
        # ----------------------
        waveform = [torchaudio.load(path) for path in audio_paths]

        audio_rate = self.get_audio_rate(waveform)

        waveform = torch.cat([w[0] for w in waveform], dim=1)  # ( 1, time@44.1kHz )

        audio = self.resample_audio(waveform, audio_rate)

        cprint(
            f">>> Resampled audio {audio_rate}Hz -> {AUDIO_RESAMPLE_RATE}Hz | shape: {waveform.shape} -> {audio.shape}",
            color="cyan",
        )

        # ----------------------
        #      EEG Loading
        # ----------------------
        matfile_paths = [
            path for path in matfile_paths if not path.split(".")[0][-3:] in EXCLUDED_SUBJECTS
        ]
        mat_raws = [scipy.io.loadmat(path)["raw"][0, 0] for path in matfile_paths]
        eeg_raws = [mat_raw["trial"][0, 0] for mat_raw in mat_raws]

        eeg_rate = self.get_eeg_rate(mat_raws)

        # ----------------------
        #     Preprocessing
        # ----------------------
        X = self.brain_preproc(eeg_raws, eeg_rate)
        cprint(f">>> Preprocessed X (EEG): {X.shape}", color="cyan")

        X, audio = shift_brain_signal(
            X, audio, srate_x=BRAIN_RESAMPLE_RATE, srate_y=AUDIO_RESAMPLE_RATE
        )
        cprint(f">>> Shifted X (EEG): {X.shape} | Audio: {audio.shape}", color="cyan")

        cprint(">> Segmenting EEG and audio using onsets.", color="cyan")
        X, audio = self.segment(X, audio, onsets_path)
        cprint(f">>> X (EEG): {X.shape} | Audio: {audio.shape}", color="cyan")

        # NOTE: baseline corection becomes naturally subject-specific
        cprint(">> Baseline correcting EEG.", color="cyan")
        X = baseline_correction(X, self.baseline_num_samples)

        cprint(">> Scaling and clamping EEG.", color="cyan")
        X = scale_and_clamp(X, self.clamp_lim)

        cprint(">> Embedding audio with wave2vec2.0.", color="cyan")
        Y = self.embed_audio(audio)
        cprint(f">>> Y (audio): {Y.shape}", color="cyan")

        return X, Y

    def resample_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resamples audio to 16kHz. (16kHz is required by wav2vec2.0)"""

        waveform = F.resample(
            waveform,
            sample_rate,
            AUDIO_RESAMPLE_RATE,
            lowpass_filter_width=self.lowpass_filter_width,
        )

        len_audio_s = waveform.shape[1] / AUDIO_RESAMPLE_RATE
        cprint(f">>> Audio length: {len_audio_s} s.", color="cyan")

        return waveform

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

    def brain_preproc(self, eeg_raws: List[np.ndarray], fsample: int) -> torch.Tensor:
        """
        Returns: ( subject, channel, time@120Hz )
        """

        trim_eeg_to = min([eeg_raw.shape[1] for eeg_raw in eeg_raws])

        X = []
        pbar = tqdm(eeg_raws)
        for i, eeg_raw in enumerate(pbar):
            pbar.set_description(f"Preprocessing subject {i} EEG.")

            # NOTE: # Drop non-EEG channels and trim to the shortest subject
            eeg_raw = eeg_raw[:60, :trim_eeg_to]

            # NOTE: in the paper they don't mention anything about filtering
            eeg_filtered = mne.filter.filter_data(
                eeg_raw,
                sfreq=fsample,
                l_freq=self.brain_filter_low,
                h_freq=self.brain_filter_high,
            )

            eeg_filtered = torch.from_numpy(eeg_filtered.astype(np.float32))

            # NOTE: in the paper they say they downsampled to exactly 120Hz with Torchaudio, so I'll stick to that
            eeg_resampled = F.resample(
                waveform=eeg_filtered,
                orig_freq=fsample,
                new_freq=BRAIN_RESAMPLE_RATE,
                lowpass_filter_width=self.lowpass_filter_width,  # TODO: check
            )

            X.append(eeg_resampled)

        return torch.stack(X)

    def get_audio_rate(self, waveform: List[Tuple[torch.Tensor, int]]) -> int:
        sample_rates = np.array([w[1] for w in waveform])
        # is all 44.1kHz
        assert np.all(sample_rates == sample_rates[0])

        return sample_rates[0]

    def get_eeg_rate(self, mat_raws: List) -> int:
        sample_rates = np.array([mat_raw["fsample"][0, 0] for mat_raw in mat_raws])
        # is all 500Hz
        assert np.all(sample_rates == sample_rates[0]), "Wrong EEG sampling rate detected."

        return sample_rates[0]
