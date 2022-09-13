import os, sys
import torch
import torchaudio
import numpy as np
import pandas as pd
import fairseq
import glob
from natsort import natsorted
import scipy.io
import mne, mne_bids
from sklearn.preprocessing import RobustScaler
from omegaconf import DictConfig, open_dict
import tqdm


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
        self.Y.requires_grad = False


        X_path = "data/Brennan2018/processed_X.pt"

        if os.path.exists(X_path):
            self.X = torch.load(X_path) # ( 33, 60, 99712 )
        else:
            self.X = self.brain_preproc(audio_embd_len=self.Y.shape[-1])
            torch.save(self.X, X_path)

        # NOTE: trying scaling X becauase it looks roughly 10 times larger than Y
        # self.X /= 10

        self.X, self.Y = shift_brain_signal(self.X, self.Y)

        print(f"X: {self.X.shape}, Y: {self.Y.shape}")
        # X: ( 33, 60, 99692 ) -> ( B, 60, 256 )
        # Y: ( 512, 99692 ) -> ( B, 512, 256 )
        self.X, self.Y, self.subject_idxs = self.batchfy(self.X, self.Y, self.seq_len)

    def __len__(self):
        return len(self.X)
        # return 4096

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idxs[i]

    # def __getitem__(self, i):
    #     X = self.X[np.random.randint(self.subj_num)] # ( 60, 99712 )

    #     t_0 = np.random.randint(X.shape[-1] - self.seq_len)
    
    #     return (
    #         X[:, t_0:t_0+self.seq_len], # ( 60, 1024 )
    #         self.Y[:, t_0:t_0+self.seq_len], # ( 512, 1024 )
    #     )

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

        cp_path = f"weights/{wav2vec_model}.pt"

        # pop unneeded keys from config dict of newer models so that I can load it without an error
        if wav2vec_model != "wav2vec_small":
            cp = torch.load(cp_path)
            unneeded_keys = ['eval_wer','eval_wer_config', 'eval_wer_tokenizer', 'eval_wer_post_process', 'autoregressive']
            cfg = DictConfig(cp['cfg'])
            with open_dict(cfg):
                for k in unneeded_keys:
                    cfg.task.pop(k)
            cp['cfg'] = cfg
            torch.save(cp, cp_path) # overwrite .pt

        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        model = model[0]
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
    def __init__(self, num_subjects=11, resample_rate=120):
        super().__init__()

        x_path = "data/Gwilliams2022/processed_X.pt"

        if os.path.exists(x_path):
            self.X = torch.load(x_path)
        else:
            self.X = self.brain_preproc(num_subjects, resample_rate)
            torch.save(self.X, x_path)

    @staticmethod
    def brain_preproc(num_subjects, resample_rate):
        X = []
        for subject_idx in tqdm.tqdm(range(num_subjects)):
            bids_path = mne_bids.BIDSPath(
                subject=str(subject_idx+1).zfill(2), # '01', '02', ...
                session="0", # TODO: there are 2 sessions for each subject
                task="0", # TODO: there are 4 tasks for each session
                datatype="meg",
                root="data/Gwilliams2022/",
            )
            raw = mne_bids.read_raw_bids(bids_path)
            raw.resample(sfreq=resample_rate) # To 120Hz

            df = raw.to_data_frame()
            meg_resampled = np.stack([df[key] for key in df.keys() if "MEG" in key]) # ( 224, 39600 )

            meg_filtered = mne.filter.filter_data(
                meg_resampled, sfreq=resample_rate, l_freq=0.5, h_freq=30,
            )

            scaler = RobustScaler().fit(meg_filtered)
            meg_scaled = scaler.tranform(meg_filtered)

            X.append(meg_scaled)

        X = np.stack(X)

        return torch.from_numpy(X.astype(np.float32))


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

    dataset = Gwilliams2022Dataset()
    print(dataset.X.shape)