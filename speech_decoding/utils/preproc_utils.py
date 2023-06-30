import os
import glob
import json
from operator import is_
import numpy as np
from termcolor import cprint
import torch
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from omegaconf import open_dict
from typing import Union
from tqdm import tqdm


def continuous(onsets: np.ndarray) -> np.ndarray:
    """
    Increments speech onsets that start from zero in each separate audio file.
    (add final timestamp in the previous audio file)
    """
    base = 0

    for i in range(len(onsets)):
        update_base = i < len(onsets) - 1 and onsets[i + 1] < onsets[i]

        if update_base:
            next_base = base + onsets[i]

        onsets[i] += base

        if update_base:
            base = next_base

    return onsets


def shift_brain_signal(
    X: Union[torch.Tensor, np.ndarray],
    Y: Union[torch.Tensor, np.ndarray],
    srate_x: int = 120,
    srate_y: int = 16000,
    shift_ms: int = 150,
):
    """
    Args:
        X: preprocessed MEG/EEG | ( subject, channel, time@120Hz )
        Y: preprocessed audio before wave2vec embedding | ( 1, time@16kHz )
        shift_ms: how much to shift MEG/EEG forward in ms
    """
    shift_s = shift_ms / 1000

    X = X[:, :, int(srate_x * shift_s) :]
    Y = Y[:, : -int(srate_y * shift_s)]

    return X, Y


@torch.no_grad()
def baseline_correction(X: torch.Tensor, baseline_num_samples: int) -> torch.Tensor:
    """
    args:
        X: ( chunks, channel, time@120Hz//segment ) or ( segment, subject, channel, time@120Hz//segment )
    return:
        X: same shape as input
    """
    orig_dim = X.dim()

    if orig_dim == 4:
        num_segments = X.shape[0]
        X = X.clone().flatten(end_dim=1)  # ( chunks, channel, time )

    baseline = X[:, :, :baseline_num_samples].mean(dim=-1)  # ( chunks, channel )

    X = (X.permute(2, 0, 1) - baseline).permute(1, 2, 0)  # ( chunks, channel, time )

    if orig_dim == 4:
        X = X.reshape(num_segments, -1, X.shape[-2], X.shape[-1])

    return X


@torch.no_grad()
def scale_and_clamp(X: torch.Tensor, clamp_lim: Union[int, float], channel_wise=True):
    """subject-wise scaling and clamping of M/EEG
    args:
        X: ( chunks, channel, time@120Hz//segment ) or ( segment, subject, channel, time@120Hz//segment )
        clamp_lim: float, abs limit (will be applied for min and max)
        channel_wise: if True, also scales and clamp each channel independently
    returns:
        X: scaled and clampted | same shape as input
    """
    if not channel_wise:
        orig_shape = X.shape

        X = X.flatten(end_dim=-2)  # ( segment * subject * channel, time//segment )

        X = RobustScaler().fit_transform(X).astype(np.float32)

        X = torch.from_numpy(X).clamp(min=-clamp_lim, max=clamp_lim)

        return X.reshape(orig_shape)

    else:
        orig_dim = X.dim()

        if orig_dim == 4:
            num_segments = X.shape[0]
            X = X.clone().flatten(end_dim=1)
            # ( segment * subject, channel, time//segment )

        res = []

        for chunk_id in tqdm(range(X.shape[0])):
            # NOTE: must be samples x features!
            scaler = RobustScaler().fit(X[chunk_id].T)

            _X = torch.from_numpy(scaler.transform(X[chunk_id].T).T)

            _X.clamp_(min=-clamp_lim, max=clamp_lim)

            res.append(_X.to(torch.float))

        X = torch.stack(res)  # ( chunks, channel, time )

        if orig_dim == 4:
            X = X.reshape(num_segments, -1, X.shape[-2], X.shape[-1])

        return X


def pad_y_time(Y: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    args:
        Y: ( segment, features@w2v, time@w2v-freq//segment )
    returns:
        Y: ( segment, features@w2v, time@w2v-freq//segment + pad )
    """
    return F.pad(Y, (0, num_samples - Y.shape[-1]), "constant", 0)


def interpolate_y_time(Y: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    args:
        Y: ( segment, features@w2v, time@w2v-freq//segment )
    returns:
        Y: ( segment, features@w2v, time@120Hz//segment )
    """
    return F.interpolate(Y, size=num_samples, mode="linear")


# NOTE: Works only for Gwilliams2022 dataset
def check_preprocs(args, data_dir):
    is_processed = False
    preproc_dirs = glob.glob(data_dir + "*/")

    for preproc_dir in preproc_dirs:
        preproc_name = os.path.basename(os.path.dirname(preproc_dir))

        try:
            with open(preproc_dir + "settings.json") as f:
                settings = json.load(f)

            x_done = settings.pop("x_done") if "x_done" in settings else False
            y_done = settings.pop("y_done") if "y_done" in settings else False
        except:
            cprint("No settings.json under preproc dir", color="yellow")
            continue

        try:
            # NOTE: I didn't want to recompute the MEG dataset. Will get rid of this hack.
            # NOTE: should be just:
            # is_processed = np.all([v == args.preprocs[k] for k, v in settings.items() if not k in excluded_keys])
            excluded_keys = ["preceding_chunk_for_baseline", "mode"]
            is_processed = np.all(
                [
                    v == args.preprocs[k]
                    for k, v in settings.items()
                    if k not in excluded_keys
                ]
            )
            if is_processed:
                cprint(
                    f"All preproc params matched to {preproc_name} -> using",
                    color="cyan",
                )
                break
        except:
            cprint(f"Preproc param name mismatch for {preproc_name}", color="yellow")
            continue

    if not is_processed:
        cprint("No matching preprocessing. Starting a new one.")

        preproc_dir = data_dir + str(len(preproc_dirs)) + "/"
        os.mkdir(preproc_dir)

        # args.preprocs.update({"x_done": False, "y_done": False})
        with open_dict(args):
            args.preprocs.x_done = False
            args.preprocs.y_done = False

        with open(preproc_dir + "settings.json", "w") as f:
            json.dump(dict(args.preprocs), f)

    else:
        # args.preprocs.update({"x_done": x_done, "y_done": y_done})
        with open_dict(args):
            args.preprocs.x_done = x_done
            args.preprocs.y_done = y_done

    return args, preproc_dir
