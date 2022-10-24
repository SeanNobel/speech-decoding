import os, sys
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
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import ast
from typing import Union
from utils.bcolors import cyan, yellow
from utils.wav2vec_util import load_wav2vec_model, getW2VLastFourLayersAvg
from utils.preproc_utils import check_preprocs
from termcolor import cprint
from pprint import pprint

mne.set_log_level(verbose="WARNING")


def baseline_correction(X):
    """Assumes that X (M/EEG) is already resampled to 120Hz"""
    return X


class Brennan2018Dataset(torch.utils.data.Dataset):

    def __init__(self, args):
        super().__init__()

        seq_len_seconds = args.seqLengthInSeconds
        wav2vec_model = args.wav2vec_model
        force_recompute = args.force_recompute,
        last4layers = args.preprocs["last4layers"]
        before = args.preprocs["before"]

        Y_path = f"data/Brennan2018/Y_embeds/embd_{wav2vec_model}.pt"

        if (not os.path.exists(Y_path)) or force_recompute[0]:
            torch.save(self.audio_preproc(
                wav2vec_model,
                last4layers=last4layers,
                before=before,
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
        else:
            cprint(f'Loading existing preprocessed EEG...', color='green')
            preprocessed_eeg = torch.load(X_path)
            self.X = preprocessed_eeg['X']
            srate = preprocessed_eeg['srate']  # ( 33, 60, 99712 )
            cprint(f"Using existing pre-processed data {self.X.shape}, srate={srate}", 'red', 'on_yellow')

        self.X, self.Y = self.shift_brain_signal(self.X, self.Y, srate=srate)

        print(f"X: {self.X.shape}, Y: {self.Y.shape}")
        # X: ( 33, 60, 99692 ) -> ( B, 60, 256 )
        # Y: ( 512, 99692 ) -> ( B, 512, 256 ) # w2v embeddings
        seq_len = int(seq_len_seconds * srate)
        cprint(f'Building batches of {seq_len_seconds} s ({seq_len} samples).', color='blue')
        self.X, self.Y, self.subject_idxs = self.batchfy(self.X, self.Y, seq_len)
        cprint(f'X: {self.X.shape} | Y: {self.Y.shape} | {self.subject_idxs.shape}', color='blue')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idxs[i]

    @staticmethod
    def batchfy(X: torch.Tensor, Y: torch.Tensor, seq_len: int):
        # NOTE: seq_len is `bptt`

        assert X.shape[-1] == Y.shape[-1]
        trim_len = X.shape[-1] - X.shape[-1] % seq_len

        X = X[:, :, :trim_len]  # ( 33, 60, 99584 ) (subj, chans, num_embeddings)
        Y = Y[:, :trim_len]  # ( 512, 99584 )       (emsize, num_embeddings)

        X = X.reshape(X.shape[0], X.shape[1], -1, seq_len)  # ( 33, 60, 389, 256 )
        Y = Y.reshape(Y.shape[0], -1, seq_len)  # ( 512, 389, 256 )

        Y = Y.unsqueeze(0).expand(X.shape[0], *Y.shape)  # ( 33, 512, 389, 256 )

        X = X.permute(0, 2, 1, 3)  # ( 33, 389, 60, 256 )
        Y = Y.permute(0, 2, 1, 3)  # ( 33, 389, 512, 256 )

        subject_idxs = torch.arange(X.shape[0]).unsqueeze(1).expand(-1, X.shape[1])  # ( 33, 389 )
        subject_idxs = subject_idxs.flatten()  # ( 19061, )

        X = X.reshape(-1, X.shape[-2], X.shape[-1])  # ( 19061, 60, 256 ) (samples, ch, emsize)
        Y = Y.reshape(-1, Y.shape[-2], Y.shape[-1])  # ( 19061, 512, 256 ) (samples, ch, emsize)

        return X, Y, subject_idxs

    @staticmethod
    def audio_preproc(
        wav2vec_model: str,
        last4layers: bool,
        before: bool,
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

        # FIXME: in the paper, activations of the last four transformer layers were averaged
        if last4layers:
            cprint(f'Generating audio embeddings', 'yellow', 'on_red')
            embeddings = getW2VLastFourLayersAvg(model, waveform, before=before)
        else:
            embeddings = model.feature_extractor(waveform).squeeze()  # (512, 36176 @16kHz) ( 512, 99712 @44.1kHz)

        embedding_srate = embeddings.shape[-1] / len_audio_s
        print(f'Original  embedding shape {embeddings.shape} | srate (out out w2v): {embedding_srate}')
        res_embeddings = F.resample(embeddings, orig_freq=10, new_freq=24)  # to upsample from ~50 to ~120 Hz
        print(f'Resampled embedding shape {res_embeddings.shape} | srate: {120}')
        return res_embeddings

    @staticmethod
    def brain_preproc(audio_embd_len):
        # NOTE: look at comprehension-scores.txt
        # excluded_subjects = [1, 6, 8, 22, 23, 26, 27, 28, 29, 30, 31, 32, 42, 45, 46, 48]

        matfile_paths = natsorted(glob.glob("data/Brennan2018/raw/*.mat"))
        # matfile_paths = [matfile_paths[i] for i in range(21) if not i in [1, 6, 8]]
        matfile_paths = [matfile_paths[i] for i in [0, 1, 3, 4, 5, 6, 7, 48]]
        cprint('using only subjects #[0, 1, 3, 4, 5, 6, 7, 48]', "blue", "on_yellow", attrs=['bold'])
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

            scaler = RobustScaler().fit(eeg_resampled)
            eeg_scaled = scaler.transform(eeg_resampled)

            X.append(eeg_scaled)
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


class Gwilliams2022Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data_dir="data/Gwilliams2022/preprocessed/"):
        super().__init__()

        self.wav2vec_model = args.wav2vec_model

        self.brain_orig_rate = 1000
        self.brain_resample_rate = args.preprocs["brain_resample_rate"]
        self.brain_filter_low = args.preprocs["brain_filter_low"]
        self.brain_filter_high = args.preprocs["brain_filter_high"]
        self.segment_len = self.brain_resample_rate * args.preprocs["seq_len"]

        self.audio_resample_rate = args.preprocs["audio_resample_rate"]
        self.lowpass_filter_width = args.preprocs["lowpass_filter_width"]

        self.last4layers = args.preprocs["last4layers"]
        self.before = args.preprocs["before"]

        self.shift_brain = args.preprocs["shift_brain"]
        self.shift_len = args.preprocs["shift_len"]

        # NOTE: x_done and y_done are added to args.preprocs
        args, preproc_dir = check_preprocs(args, data_dir)
        self.x_path = preproc_dir + "x_dict.npy"
        self.y_path = preproc_dir + "y_dict.npy"
        real_dur_path = preproc_dir + "real_durations.npy"

        # Make X
        if args.preprocs["x_done"]:
            cprint("Found x_dict.npy. Skipping preprocessing.", color="cyan")
            self.X = np.load(self.x_path, allow_pickle=True).item()
            self.real_durations = np.load(real_dur_path, allow_pickle=True).item()
        else:
            self.real_durations = {}  # will be updated in self.brain_preproc
            self.X = self.brain_preproc(args.num_subjects)  # ???
            np.save(real_dur_path, self.real_durations)
            args.preprocs.update({"x_done": True})
            with open(preproc_dir + "settings.json", 'w') as f:
                json.dump(args.preprocs, f)

        # Make Y if it doesn't already exist
        if args.preprocs["y_done"]:
            cprint("Found y_dict.npy. Skipping preprocessing.", color="cyan")
            self.Y = np.load(self.y_path, allow_pickle=True).item()
        else:
            self.Y = self.audio_preproc()
            args.preprocs.update({"y_done": True})
            with open(preproc_dir + "settings.json", 'w') as f:
                json.dump(args.preprocs, f)

        # NOTE: this also updates self.X, self.Y. self.Y becomes a list
        self.subject_idxs, self.task_id_list = self.batchfy()

        # self.Y.requires_grad = False

        print(
            f"X: {self.X.shape}, Y (list): {len(self.Y)}, subject_idxs: {self.subject_idxs.shape}, task_id_list: {len(self.task_id_list)}"
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        task_id = self.task_id_list[i]
        i_in_task = self.get_i_in_task(i, task_id)

        return self.X[i], self.Y[task_id][i_in_task], self.subject_idxs[i]

    # NOTE: this is very hard coding but should be correct.
    def get_i_in_task(self, i, task_id):
        i_in_task = 0
        while i > 0 and self.task_id_list[i] == self.task_id_list[i - 1]:
            i_in_task += 1
            i -= 1

        # avoid error by same task sucsession by missing session
        return i_in_task % self.Y[task_id].shape[0]

    @staticmethod
    def data_reform(data: np.ndarray, segment_len) -> torch.Tensor:
        trim_len = data.shape[-1] % segment_len
        data = data[:, :-trim_len]
        data = data.reshape(data.shape[0], -1, segment_len).transpose(1, 0, 2)

        return torch.from_numpy(data.astype(np.float32))

    def batchfy(self):
        X_list = []
        Y_list = []
        subject_idxs_list = []
        task_id_list = []

        cprint("Batchfying X", color="cyan")
        for key, X in tqdm(self.X.items()):
            # Y = self.Y[key.split("_")[-1]]
            if self.shift_brain:
                # X, Y = self.shift_brain_signal(X, Y)
                X = self.shift_brain_signal(X, is_Y=False)

            X = self.data_reform(X, self.segment_len)

            X_list.append(X)
            # Y_list.append(self.data_reform(Y, self.segment_len))

            # assert X_list[-1].shape[0] == Y_list[-1].shape[0]
            # if i > 0 and i % 4 == 0:
            #     print(
            #         f"Shapes: {Y_list[-1].shape}, {Y_list[-5].shape} | equal: {torch.equal(Y_list[-1], Y_list[-5])}"
            #     )

            subj_idx = int(key.split("_")[0][-2:]) - 1  # 0, 1,...
            subj_idx *= torch.ones(X.shape[0], dtype=torch.uint8)
            subject_idxs_list.append(subj_idx)

            task_id = int(key[-1])  # 0 or 1 or 2 or 3
            task_id_list += [task_id] * X.shape[0]

        cprint("Batchfying Y", color="cyan")
        for key, Y in tqdm(self.Y.items()):
            if self.shift_brain:
                Y = self.shift_brain_signal(Y, is_Y=True)

            Y = self.data_reform(Y, self.segment_len)

            Y_list.append(Y)

        self.X = torch.cat(X_list, dim=0)
        # self.Y = torch.cat(Y_list, dim=0)
        self.Y = Y_list

        return torch.cat(subject_idxs_list), task_id_list

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

    def brain_preproc(self, num_subjects, num_channels=208):
        np.save(self.x_path, {})
        for subject_idx in range(num_subjects):
            for session_idx in range(2):  # 2 sessions for each subject
                for task_idx in range(4):  # 4 tasks for each subject

                    description = f"subject{str(subject_idx+1).zfill(2)}_sess{session_idx}_task{task_idx}"
                    cprint(description, color="cyan")

                    bids_path = mne_bids.BIDSPath(
                        subject=str(subject_idx + 1).zfill(2),
                        # '01', '02', ...
                        session=str(session_idx),
                        task=str(task_idx),
                        datatype="meg",
                        root="data/Gwilliams2022/",
                    )
                    try:
                        raw = mne_bids.read_raw_bids(bids_path)
                    except:
                        cprint("No .con data was found", color="yellow")
                        continue

                    df = raw.to_data_frame()
                    meg_raw = np.stack([df[key] for key in df.keys() if "MEG" in key])  # ( 224, 396000 )
                    # TODO: 16 channels are references, but need to confirm that last 16 are
                    meg_raw = meg_raw[:num_channels]  # ( 208, 396000 )

                    df_annot = raw.annotations.to_data_frame()
                    meg_trimmed, real_durations = self.trim_nosound_regions(meg_raw, df_annot)  # ( 208, <396000 )
                    self.update_real_durations(real_durations, task_idx)

                    meg_filtered = mne.filter.filter_data(
                        meg_trimmed,
                        sfreq=self.brain_orig_rate,
                        l_freq=self.brain_filter_low,
                        h_freq=self.brain_filter_high,
                    )

                    # To 120 Hz
                    meg_resampled = mne.filter.resample(meg_filtered,
                                                        down=self.brain_orig_rate /
                                                        self.brain_resample_rate)  # ( 208, 37853 )

                    scaler = RobustScaler().fit(meg_resampled)
                    meg_scaled = scaler.transform(meg_resampled)
                    cprint(meg_scaled.shape, color="cyan")

                    # save to disk
                    X = np.load(self.x_path, allow_pickle=True).item()
                    X.update({description: meg_scaled})
                    np.save(self.x_path, X)

        return X

    @torch.no_grad()
    def audio_preproc(self):
        wav2vec = load_wav2vec_model(self.wav2vec_model)
        wav2vec.eval()

        task_prefixes = ["lw1", "cable", "easy", "the"]

        Y = {}
        assert os.path.exists(
            'data/Gwilliams2022/stimuli/audio'), "The path `data/Gwilliams2022/stimuli/audio` DOESN'T EXIST."
        for task_idx in range(4):  # 4 tasks for each subject

            audio_paths = natsorted(glob.glob(f"data/Gwilliams2022/stimuli/audio/{task_prefixes[task_idx]}*.wav"))

            audio_raw = []
            for f, path in enumerate(audio_paths):
                waveform, sample_rate = torchaudio.load(path)

                cutoff = int(sample_rate * self.real_durations[f"task{task_idx}"][f])
                if waveform.shape[1] > cutoff:
                    waveform = waveform[:, :cutoff]
                else:
                    print(yellow("No audio cutoff"))

                # Upsample
                waveform = F.resample(waveform,
                                      sample_rate,
                                      self.audio_resample_rate,
                                      lowpass_filter_width=self.lowpass_filter_width)
                cprint(f"Audio after resampling: {waveform.shape}", color="cyan")

                if self.last4layers:
                    embeddings = getW2VLastFourLayersAvg(wav2vec, waveform, before=self.before)
                else:
                    embeddings = wav2vec.feature_extractor(waveform).squeeze()
                cprint(f"Audio embedding: {embeddings.shape}", color="cyan")

                rate_after_wav2vec = self.audio_resample_rate * embeddings.shape[-1] / waveform.shape[-1]  # 49.9737...
                cprint(rate_after_wav2vec, color="cyan")

                # NOTE: torchaudio resample doesn't accept float freqs
                # embeddings = F.resample(embeddings,
                #                         orig_freq=rate_after_wav2vec,
                #                         new_freq=self.brain_resample_rate)
                embeddings = mne.filter.resample(embeddings.numpy().astype(np.float64),
                                                 up=self.brain_resample_rate / rate_after_wav2vec,
                                                 axis=-1)
                cprint(f"Audio embedding upsampled: {embeddings.shape}", color="cyan")

                audio_raw.append(embeddings)

            audio_raw = np.concatenate(audio_raw, axis=-1)

            print(audio_raw.shape)

            Y.update({f"task{task_idx}": audio_raw})

        np.save(self.y_path, Y)

        return Y

    @staticmethod
    def to_second(onset):  # pandas Timestamp object
        return onset.minute * 60 + onset.second + onset.microsecond * 1e-6

    def trim_nosound_regions(self, meg_raw, df_annot):
        prev_sound_id = -1.0
        starts_ends_t = []
        for t, desc in enumerate(df_annot.description):
            desc = ast.literal_eval(desc)
            if desc['sound_id'] != prev_sound_id:
                prev_sound_id = desc['sound_id']

                starts_ends_t += [t - 1, t]

        starts_ends_t = starts_ends_t[1:] + [t]
        starts_ends_t = np.reshape(starts_ends_t, (-1, 2))

        meg_trimmed = []
        real_durations = []
        for start_t, end_t in starts_ends_t:
            start = self.to_second(df_annot.onset[start_t])
            end = self.to_second(df_annot.onset[end_t]) + df_annot.duration[end_t]

            meg_trimmed.append(meg_raw[:, int(start * 1000):int(end * 1000)])

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


class ToyDataset():

    def __init__(self, num_samples=10000, seq_len=256, X_dim=60, Y_dim=512):
        super().__init__()

        linspaces = torch.stack([torch.linspace(st, st + 10, seq_len) for st in torch.rand(num_samples) * 10])

        self.Y = torch.stack([linspaces * torch.rand(1) for _ in range(Y_dim)])
        # self.X = torch.stack([linspaces * torch.rand(1) for _ in range(X_dim)])
        self.X = self.Y[:X_dim]

        self.Y = torch.cos(self.Y.permute(1, 0, 2))
        self.X = torch.cos(self.X.permute(1, 0, 2))

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

    from configs.args import args
    dataset = Gwilliams2022Dataset(args)
