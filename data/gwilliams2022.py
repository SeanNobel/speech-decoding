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
from utils.preproc_utils import check_preprocs, scaleAndClamp_single, baseline_correction_single
from termcolor import cprint
from pprint import pprint
from einops import rearrange
from sklearn.preprocessing import RobustScaler

mne.set_log_level(verbose="WARNING")


class Gwilliams2022Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data_dir="data/Gwilliams2022/preprocessed/"):
        super().__init__()

        self.wav2vec_model = args.wav2vec_model

        self.brain_orig_rate = 1000
        self.brain_resample_rate = args.preprocs["brain_resample_rate"]
        self.brain_filter_low = args.preprocs["brain_filter_low"]
        self.brain_filter_high = args.preprocs["brain_filter_high"]
        self.seq_len_samp = self.brain_resample_rate * args.preprocs["seq_len_sec"]
        self.clamp = args.preprocs["clamp"]
        self.clamp_lim = args.preprocs["clamp_lim"]
        self.preceding_chunk_for_baseline = args.preprocs["preceding_chunk_for_baseline"]
        self.baseline_len_samp = int(self.brain_resample_rate * args.preprocs["baseline_len_sec"])

        self.audio_resample_rate = args.preprocs["audio_resample_rate"]
        self.lowpass_filter_width = args.preprocs["lowpass_filter_width"]

        self.last4layers = args.preprocs["last4layers"]
        self.mode = args.preprocs["mode"]

        self.shift_brain = args.preprocs["shift_brain"]
        self.shift_len = args.preprocs["shift_len"]

        # NOTE: x_done and y_done are added to args.preprocs
        args, preproc_dir = check_preprocs(args, data_dir)
        self.x_path = preproc_dir + "x_dict.npy"
        self.y_path = preproc_dir + "y_dict.npy"
        real_dur_path = preproc_dir + "real_durations.npy"

        # ------------
        #    Make X
        # ------------
        if args.preprocs["x_done"]:
            self.X = np.load(self.x_path, allow_pickle=True).item()
            self.real_durations = np.load(real_dur_path, allow_pickle=True).item()
        else:
            self.real_durations = {}  # will be updated in self.brain_preproc
            self.X = self.brain_preproc(args.num_subjects)  # ???

            np.save(real_dur_path, self.real_durations)
            args.preprocs.update({"x_done": True})
            with open(preproc_dir + "settings.json", 'w') as f:
                json.dump(args.preprocs, f)

        # -------------------------------------------
        #     Make Y if it doesn't already exist
        # -------------------------------------------
        if args.force_recompute_y or not args.preprocs['y_done']:
            self.Y = self.audio_preproc()
            args.preprocs.update({"y_done": True})
            with open(preproc_dir + "settings.json", 'w') as f:
                json.dump(args.preprocs, f)
        else:
            self.Y = np.load(self.y_path, allow_pickle=True).item()

        self.X_list, self.Y = self.batchfy()

        self.num_subjects = len(self.X_list)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):  # NOTE: i is id of a speech segment
        subject_idx = np.random.randint(self.num_subjects)

        subject_X = self.X_list[subject_idx]
        num_sessions = subject_X.shape[0]  # 1 or 2
        session_idx = np.random.randint(num_sessions)

        return subject_X[session_idx, i], self.Y[i], subject_idx

    @staticmethod
    def data_reform(data: torch.Tensor, segment_len) -> torch.Tensor:
        trim_len = data.shape[-1] % segment_len
        data = data[:, :-trim_len]
        data = data.reshape(data.shape[0], -1, segment_len).permute(1, 0, 2)

        return data

    def batchfy(self):
        # self.X.keys() -> ['subject01_sess0_task0', ..., 'subject27_sess1_task3']
        # self.Y.keys() -> ['task0', 'task1', 'task2', 'task3']

        assert natsorted(self.X.keys()) == list(self.X.keys()), 'self.X.keys() is not sorted'

        # ------------------------------------------------------
        #     Make X_dict (MEG are concatenated along tasks)
        # ------------------------------------------------------
        cprint("=> Batchfying X", color="cyan")

        X_dict = {}

        for key, X in tqdm(self.X.items()):
            if self.shift_brain:
                X = self.shift_brain_signal(X, is_Y=False)

            X = scaleAndClamp_single(X, self.clamp_lim, self.clamp)  # ( ch=208, len=37835 )

            X = self.data_reform(X, self.seq_len_samp)  # ( num_segment=~500, ch=208, len=360 )

            X = baseline_correction_single(X, self.baseline_len_samp, self.preceding_chunk_for_baseline)

            key_wo_task = '_'.join(key.split('_')[:-1])  # e.g. 'subject01_sess0_task0' -> 'subject01_sess0'
            if not key_wo_task in X_dict.keys():
                X_dict.update({key_wo_task: X})
            else:
                _X = torch.cat((X_dict[key_wo_task], X))
                X_dict.update({key_wo_task: _X})

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

            Y = self.data_reform(Y, self.seq_len_samp)  # ( num_segment=~500, F=1024, len=360 )

            Y_list.append(Y)

        Y = torch.cat(Y_list)

        # -----------------------------------------------------------------------
        #   Drop sessions in which some tasks are missing -> X_dict to X_list
        # -----------------------------------------------------------------------
        X_list = []

        subj_strs = natsorted(set(k.split('_')[0] for k in X_dict.keys()))
        for subj_str in subj_strs:
            # subj_str = k.split('_')[0]
            # num_sessions_subj = sum([(subj_str in k) for k in X_dict.keys()])

            sess_list = []
            for k, X in X_dict.items():
                if subj_str in k:
                    if X.shape[0] == Y.shape[0]:
                        cprint(f"{k}: {X_dict[k].shape}", color='cyan')
                        sess_list.append(X)
                    else:
                        cprint(f"{k}: {X_dict[k].shape} -> dropped", color='yellow')

            if len(sess_list) > 0:
                X_list.append(torch.stack(sess_list))

        print('---------------------------')
        cprint(f"Using {len(X_list)} subjects (some of them have only 1 session)", color='cyan')
        cprint(str([x.shape[0] for x in X_list]), color='cyan')

        print('---------------------------')
        cprint(f"Audio: {Y.shape}", color='cyan')

        return X_list, Y  #, torch.cat(subject_idxs_list)

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
            for session_idx in range(2):  # 2 sessions for each subject (but sometimes only 1 or even 0)
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
                    # NOTE: (kind of) confirmed that last 16 channels are REF
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

                    # scaler = RobustScaler().fit(meg_resampled)
                    # meg_scaled = scaler.transform(meg_resampled)
                    # cprint(meg_scaled.shape, color="cyan")

                    # save to disk
                    X = np.load(self.x_path, allow_pickle=True).item()
                    X.update({description: meg_resampled})
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

                # Upsample to 16000Hz
                waveform = F.resample(waveform,
                                      orig_freq=sample_rate,
                                      new_freq=self.audio_resample_rate,
                                      lowpass_filter_width=self.lowpass_filter_width)
                cprint(f"Audio after resampling: {waveform.shape}", color="cyan")

                if self.last4layers:
                    embeddings = getW2VLastFourLayersAvg(wav2vec, waveform, mode=self.mode)
                else:
                    embeddings = wav2vec.feature_extractor(waveform).squeeze()
                cprint(f"Audio embedding: {embeddings.shape}", color="cyan")

                rate_after_wav2vec = self.audio_resample_rate * embeddings.shape[-1] / waveform.shape[-1]  # 49.9737...
                cprint(rate_after_wav2vec, color="cyan")

                # NOTE: torchaudio resample doesn't accept float freqs
                # To 120 Hz
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

    def _batchfy(self):
        """
        Legacy. In this way there are more than two overlapping segments in a single batch.
        """
        X_list = []
        Y_list = []
        subject_idxs_list = []
        task_idxs_list = []
        i_in_task_list = []

        # self.X.keys() -> ['subject01_sess0_task0', ..., 'subject27_sess1_task3']
        # self.Y.keys() -> ['task0', 'task1', 'task2', 'task3']

        cprint("=> Batchfying X", color="cyan")
        for key, X in tqdm(self.X.items()):
            if self.shift_brain:
                X = self.shift_brain_signal(X, is_Y=False)

            X = scaleAndClamp_single(X, self.clamp_lim, self.clamp)  # ( ch=208, len=37835 )

            X = self.data_reform(X, self.seq_len_samp)  # ( num_segment=~500, ch=208, len=360 )

            X = baseline_correction_single(X, self.baseline_len_samp, self.preceding_chunk_for_baseline)

            X_list.append(X)
            num_segments = X.shape[0]

            subj_idx = int(key.split("_")[0][-2:]) - 1  # 0, 1,...
            subj_idx *= torch.ones(num_segments, dtype=torch.int64)
            subject_idxs_list.append(subj_idx)

            task_idx = int(key[-1])  # 0 or 1 or 2 or 3
            task_idxs_list += [task_idx] * num_segments

            i_in_task_list.append(torch.arange(num_segments))  # torch.int64

        cprint("=> Batchfying Y", color="cyan")
        for key, Y in tqdm(self.Y.items()):
            # Y: ( F=1024, len=37835 )
            if self.shift_brain:
                # NOTE: This doesn't shift audio. Just crops the end.
                Y = self.shift_brain_signal(Y, is_Y=True)

            Y = torch.from_numpy(Y.astype(np.float32))

            Y = self.data_reform(Y, self.seq_len_samp)  # ( num_segment=~500, F=1024, len=360 )

            Y_list.append(Y)

        return (torch.cat(X_list, dim=0), Y_list, torch.cat(subject_idxs_list),
                torch.tensor(task_idxs_list, dtype=torch.int64), torch.cat(i_in_task_list))


if __name__ == "__main__":

    from configs.args import args
    dataset = Gwilliams2022Dataset(args)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    for _ in dataloader:
        break