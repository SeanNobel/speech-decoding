import os, sys
import torch
import torchaudio
import fairseq
import glob
from natsort import natsorted
import scipy.io
import mne
import numpy as np
from omegaconf import DictConfig, open_dict

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
            self.X = torch.load(X_path) # ( 49, 60, 99712 )
        else:
            self.X = self.brain_preproc(audio_embd_len=self.Y.shape[-1])
            torch.save(self.X, X_path)

        self.subj_num = self.X.shape[0]

        print(f"X: {self.X.shape}, Y: {self.Y.shape}")
        # X: ( 49, 60, 99712 ) -> ( B, 60, 256 )
        # Y: ( 512, 99712 ) -> ( B, 512, 256 )
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

        X = X[:, :, :trim_len] # ( 49, 60, 99584 )
        Y = Y[:, :trim_len] # ( 512, 99584 )

        X = X.reshape(X.shape[0], X.shape[1], -1, seq_len) # ( 49, 60, 389, 256 )
        Y = Y.reshape(Y.shape[0], -1, seq_len) # ( 512, 389, 256 )

        Y = Y.unsqueeze(0).expand(X.shape[0], *Y.shape) # ( 49, 512, 389, 256 )
        
        X = X.permute(0,2,1,3) # ( 49, 389, 60, 256 )
        Y = Y.permute(0,2,1,3) # ( 49, 389, 512, 256 )

        subject_idxs = torch.arange(X.shape[0]).unsqueeze(1).expand(-1, X.shape[1]) # ( 49, 389 )
        subject_idxs = subject_idxs.flatten() # ( 19061, )

        X = X.reshape(-1, X.shape[-2], X.shape[-1]) # ( 19061, 60, 256 )
        Y = Y.reshape(-1, Y.shape[-2], Y.shape[-1]) # ( 19061, 512, 256 )

        return X, Y, subject_idxs

    @staticmethod
    def audio_preproc(wav2vec_model: str):
        # waveform: ( 1, 31908132 )
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

            eeg_resampled = mne.filter.resample(
                eeg_filtered, down=eeg_filtered.shape[-1]/audio_embd_len,
            )

            X.append(eeg_resampled)

        X = np.stack(X) # ( 49, 60, 99712 )
        
        return torch.from_numpy(X.astype(np.float32))


if __name__ == '__main__':
    
    dataset = Brennan2018Dataset(seq_len=256)

    print(dataset.Y.requires_grad)