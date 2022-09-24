import os, sys
import torch
import torchaudio
import numpy as np
import pandas as pd
import glob
from natsort import natsorted
import scipy.io
import mne, mne_bids
from sklearn.preprocessing import RobustScaler
import tqdm
import ast
from typing import Union
from utils.bcolors import cyan, yellow
from utils.wav2vec_util import load_wav2vec_model
mne.set_log_level(verbose="WARNING")


def baseline_correction(X):
    """Assumes that X (M/EEG) is already resampled to 120Hz"""
    return X

def shift_brain_signal(X, Y, resampled_rate=135, shift=150):
    """
    - X: ( 33, 60, 99712 ) Y: ( 512, 99712 )
    - resampled_rate (Hz): rates of M/EEG after resampling and speech after wav2vec2.0 encoding
    - shift (ms): how much to shift M/EEG forward
    """
    # TODO: find actual resampled_rate (need to fix resampling amount for subjects)

    shift = int(resampled_rate * (shift / 1000)) # 19

    X = X[:, :, shift:] # ( 33, 60, 99692 )
    Y = Y[:, :-shift] # ( 512, 99692 )

    return X, Y


class Brennan2018Dataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, wav2vec_model):
        super().__init__()

        self.seq_len = seq_len

        Y_path = f"data/Brennan2018/Y_embeds/embd_{wav2vec_model}.pt"

        if not os.path.exists(Y_path):
            torch.save(self.audio_preproc(wav2vec_model), Y_path)

        self.Y = torch.load(Y_path) # ( 512, 99712 )
        # self.Y.requires_grad = False

        X_path = "data/Brennan2018/processed_X.pt"

        if os.path.exists(X_path):
            self.X = torch.load(X_path) # ( 33, 60, 99712 )
        else:
            self.X = self.brain_preproc(audio_embd_len=self.Y.shape[-1])
            torch.save(self.X, X_path)

        self.X, self.Y = shift_brain_signal(self.X, self.Y)

        print(f"X: {self.X.shape}, Y: {self.Y.shape}")
        # X: ( 33, 60, 99692 ) -> ( B, 60, 256 )
        # Y: ( 512, 99692 ) -> ( B, 512, 256 )
        self.X, self.Y, self.subject_idxs = self.batchfy(self.X, self.Y, self.seq_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idxs[i]

    @staticmethod
    def batchfy(X:torch.Tensor, Y:torch.Tensor, seq_len:int):
        assert X.shape[-1] == Y.shape[-1]
        trim_len = X.shape[-1] - X.shape[-1] % seq_len

        X = X[:, :, :trim_len] # ( 33, 60, 99584 )
        Y = Y[:, :trim_len] # ( 512, 99584 )

        X = X.reshape(X.shape[0], X.shape[1], -1, seq_len) # ( 33, 60, 389, 256 )
        Y = Y.reshape(Y.shape[0], -1, seq_len) # ( 512, 389, 256 )

        Y = Y.unsqueeze(0).expand(X.shape[0], *Y.shape) # ( 33, 512, 389, 256 )
        
        X = X.permute(0,2,1,3) # ( 33, 389, 60, 256 )
        Y = Y.permute(0,2,1,3) # ( 33, 389, 512, 256 )

        subject_idxs = torch.arange(X.shape[0]).unsqueeze(1).expand(-1, X.shape[1]) # ( 33, 389 )
        subject_idxs = subject_idxs.flatten() # ( 19061, )

        X = X.reshape(-1, X.shape[-2], X.shape[-1]) # ( 19061, 60, 256 )
        Y = Y.reshape(-1, Y.shape[-2], Y.shape[-1]) # ( 19061, 512, 256 )

        return X, Y, subject_idxs

    @staticmethod
    def audio_preproc(wav2vec_model: str):
        # waveform: ( 1, 31908132 ), sample_rate: 44100
        waveform, sample_rate = torchaudio.load("data/Brennan2018/merged_audio.wav")

        model = load_wav2vec_model(wav2vec_model)
        model.eval()

        # FIXME: in the paper, activations of the last four transformer layers were averaged
        return model.feature_extractor(waveform).squeeze() # ( 512, 99712 )

    @staticmethod
    def brain_preproc(audio_embd_len):
        # NOTE: look at comprehension-scores.txt
        excluded_subjects = [1, 6, 8, 22, 23, 26, 27, 28, 29, 30, 31, 32, 42, 45, 46, 48]

        matfile_paths = natsorted(glob.glob("data/Brennan2018/raw/*.mat"))
        matfile_paths = np.delete(matfile_paths, excluded_subjects)

        X = []
        for matfile_path in matfile_paths:
            mat_raw = scipy.io.loadmat(matfile_path)["raw"][0,0]
            eeg_raw = mat_raw["trial"][0,0][:60]
            fsample = mat_raw["fsample"][0,0]
            # label = [e[0] for e in mat_raw["label"].squeeze()]

            eeg_filtered = mne.filter.filter_data(
                eeg_raw, sfreq=fsample, l_freq=1.0, h_freq=None
            )

            # NOTE: This resamples EEG from 500Hz down to around 135Hz
            eeg_resampled = mne.filter.resample(
                eeg_filtered, down=eeg_filtered.shape[-1]/audio_embd_len,
            )

            scaler = RobustScaler().fit(eeg_resampled)
            eeg_scaled = scaler.transform(eeg_resampled)

            X.append(eeg_scaled)

        X = np.stack(X) # ( 33, 60, 99712 )
        
        return torch.from_numpy(X.astype(np.float32))


class Gwilliams2022Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wav2vec_model: str,
        num_subjects=27,
        seq_len=3, # length of a segment in seconds
        resample_rate=120,
        audio_upsample=38530,
        shift_brain=True,
    ):
        super().__init__()

        self.wav2vec_model = wav2vec_model
        self.resample_rate = resample_rate
        self.meg_len = resample_rate * seq_len
        # NOTE: upsample audio so that it becomes (roughly) 120Hz after wav2vec2.0
        self.audio_upsample = audio_upsample
        # TODO: decide whether to wav2vec before or after
        self.audio_len = audio_upsample * seq_len # self.meg_len
        self.shift_brain = shift_brain

        self.x_path = "data/Gwilliams2022/processed_x_dict.npy"
        self.y_path = "data/Gwilliams2022/processed_y_dict.npy"
        real_dur_path = "data/Gwilliams2022/real_durations.npy"

        # Make X
        if os.path.exists(self.x_path):
            self.X = np.load(self.x_path, allow_pickle=True).item()
            self.real_durations = np.load(real_dur_path, allow_pickle=True).item()
        else:
            self.real_durations = {} # will be updated in self.brain_preproc
            self.X = self.brain_preproc(num_subjects)
            np.save(real_dur_path, self.real_durations)

        # Make Y
        if os.path.exists(self.y_path):
            self.Y = np.load(self.y_path, allow_pickle=True).item()
        else:
            self.Y = self.audio_preproc()

        # NOTE: this also updates self.X, self.Y
        self.subject_idxs = self.batchfy()

        self.Y.requires_grad = False

        print(f"X: {self.X.shape}, Y: {self.Y.shape}, subject_idxs: {self.subject_idxs.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idxs[i]

    def batchfy(self):
        X_list = []; Y_list = []; subject_idxs_list = []
        for i, key in enumerate(self.X.keys()):
            X = self.X[key]
            Y = self.Y[key.split("_")[-1]]
            if self.shift_brain:
                X, Y = self.shift_brain_signal(X, Y)

            # print(f"X: {X.shape}")
            trim_len = X.shape[-1] % self.meg_len
            X = X[:, :-trim_len]
            X = X.reshape(X.shape[0], -1, self.meg_len).transpose(1,0,2)
            X_list.append(torch.from_numpy(X.astype(np.float32)))

            # print(f"Y: {Y.shape}")
            trim_len = Y.shape[-1] % self.audio_len
            Y = Y[:, :-trim_len]
            Y = Y.reshape(-1, self.audio_len) # .unsqueeze(1)
            # Y = Y.reshape(Y.shape[0], -1, self.audio_len).permute(1,0,2)
            Y_list.append(Y)

            assert X.shape[0] == Y.shape[0]

            subj_idx = int(key.split("_")[0][-2:]) - 1 # 0, 1,...
            subj_idx *= torch.ones(X.shape[0], dtype=torch.uint8)
            subject_idxs_list.append(subj_idx)

        self.X = torch.cat(X_list, dim=0)
        self.Y = torch.cat(Y_list, dim=0)

        return torch.cat(subject_idxs_list)

    def brain_preproc(self, num_subjects, num_channels=208):
        np.save(self.x_path, {})
        for subject_idx in range(num_subjects):
            for session_idx in range(2): # 2 sessions for each subject
                for task_idx in range(4): # 4 tasks for each subject

                    description = f"subject{str(subject_idx+1).zfill(2)}_sess{session_idx}_task{task_idx}"
                    print(cyan(description))

                    bids_path = mne_bids.BIDSPath(
                        subject=str(subject_idx+1).zfill(2), # '01', '02', ...
                        session=str(session_idx),
                        task=str(task_idx),
                        datatype="meg",
                        root="data/Gwilliams2022/",
                    )
                    try:
                        raw = mne_bids.read_raw_bids(bids_path)
                    except:
                        print(yellow("No .con data was found"))
                        continue

                    df = raw.to_data_frame()
                    meg_raw = np.stack([df[key] for key in df.keys() if "MEG" in key]) # ( 224, 396000 )
                    # TODO: 16 channels are references, but need to confirm that last 16 are
                    meg_raw = meg_raw[:num_channels] # ( 208, 396000 )

                    df_annot = raw.annotations.to_data_frame()
                    meg_trimmed, real_durations = self.trim_nosound_regions(meg_raw, df_annot) # ( 208, <396000 )
                    self.update_real_durations(real_durations, task_idx)

                    # To 120 Hz
                    meg_resampled = mne.filter.resample(meg_trimmed, down=1000/self.resample_rate) # ( 208, 37853 )

                    meg_filtered = mne.filter.filter_data(
                        meg_resampled, sfreq=self.resample_rate, l_freq=0.5, h_freq=30,
                    )

                    scaler = RobustScaler().fit(meg_filtered)
                    meg_scaled = scaler.transform(meg_filtered)
                    print(cyan(meg_scaled.shape))

                    # save to disk
                    X = np.load(self.x_path, allow_pickle=True).item()
                    X.update({description: meg_scaled})
                    np.save(self.x_path, X)

        return X

    @torch.no_grad()
    def audio_preproc(self):
        # wav2vec = load_wav2vec_model(self.wav2vec_model)
        # wav2vec.eval()

        task_prefixes = ["lw1", "cable", "easy", "the"]

        Y = {}
        for task_idx in range(4): # 4 tasks for each subject

            audio_paths = natsorted(glob.glob(
                f"data/Gwilliams2022/stimuli/audio/{task_prefixes[task_idx]}*.wav"))

            audio_raw = []
            for f, path in enumerate(audio_paths):
                waveform, sample_rate = torchaudio.load(path)

                cutoff = int(sample_rate * self.real_durations[f"task{task_idx}"][f])
                if waveform.shape[1] > cutoff:
                    waveform = waveform[:, :cutoff]
                else:
                    print(yellow("No audio cutoff"))

                # Upsample
                waveform = torchaudio.functional.resample(
                    waveform=waveform,
                    orig_freq=sample_rate,
                    new_freq=self.audio_upsample,
                )

                # FIXME: in the paper, activations of the last four transformer layers were averaged
                # waveform = wav2vec.feature_extractor(waveform).squeeze()

                audio_raw.append(waveform)

            audio_raw = torch.cat(audio_raw, dim=-1)

            print(audio_raw.shape)

            Y.update({f"task{task_idx}": audio_raw})

        np.save(self.y_path, Y)

        return Y

    @staticmethod
    def to_second(onset): # pandas Timestamp object
        return onset.minute * 60 + onset.second + onset.microsecond * 1e-6

    def trim_nosound_regions(self, meg_raw, df_annot):
        prev_sound_id = -1.0
        starts_ends_t = []
        for t, desc in enumerate(df_annot.description):
            desc = ast.literal_eval(desc)
            if desc['sound_id'] != prev_sound_id:
                prev_sound_id = desc['sound_id']

                starts_ends_t += [t-1, t]
                    
        starts_ends_t = starts_ends_t[1:] + [t]
        starts_ends_t = np.reshape(starts_ends_t, (-1, 2))

        meg_trimmed = []
        real_durations = []
        for start_t, end_t in starts_ends_t:
            start = self.to_second(df_annot.onset[start_t])
            end = self.to_second(df_annot.onset[end_t]) + df_annot.duration[end_t]

            meg_trimmed.append(meg_raw[:, int(start*1000):int(end*1000)])

            real_durations.append(end - start)

        meg_trimmed = np.concatenate(meg_trimmed, axis=1)

        return meg_trimmed, real_durations

    def update_real_durations(self, real_durations, task_idx) -> None:
        task_str = f"task{task_idx}"
        if task_str in self.real_durations.keys():
            if not np.allclose(self.real_durations[task_str], real_durations):
                print(yellow("Real durations are different"))
                print(yellow(real_durations))
                print(yellow(self.real_durations[task_str]))

        self.real_durations.update({task_str: real_durations})

    def shift_brain_signal(self, X: np.ndarray, Y: torch.Tensor, shift=150):
        """
        - X: ( 208, 37853 ) Y: ( 1, 12153904 )
        - shift (ms): how much to shift M/EEG forward
        """
        # TODO: find actual resampled_rate (need to fix resampling amount for subjects)

        x_shift = int(self.resample_rate * (shift / 1000)) # 18
        y_shift = int(self.audio_upsample * (shift / 1000)) # 5779

        X = X[:, x_shift:] # ( 208, 37835 )
        Y = Y[:, :-y_shift] # ( 1, 12148125 )

        return X, Y


class ToyDataset():
    def __init__(self, num_samples=10000, seq_len=256, X_dim=60, Y_dim=512):
        super().__init__()

        linspaces = torch.stack([
            torch.linspace(st, st+10, seq_len) for st in torch.rand(num_samples) * 10
        ])

        self.Y = torch.stack([linspaces * torch.rand(1) for _ in range(Y_dim)])
        # self.X = torch.stack([linspaces * torch.rand(1) for _ in range(X_dim)])
        self.X = self.Y[:X_dim]

        self.Y = torch.cos(self.Y.permute(1,0,2))
        self.X = torch.cos(self.X.permute(1,0,2))

        self.subject_idxs = torch.randint(33, size=(num_samples,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idxs[i]


if __name__ == '__main__':
    
    # dataset = Brennan2018Dataset(seq_len=256)
    # print(dataset.Y.requires_grad)

    # dataset = ToyDataset()
    # print(dataset.Y.shape)

    dataset = Gwilliams2022Dataset(wav2vec_model="xlsr_53_56k")
