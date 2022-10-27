import os
import glob
import json
from operator import is_
import numpy as np
from termcolor import cprint
import torch
from sklearn.preprocessing import RobustScaler


# NOTE currently only works for gwilliams2022.yml
def check_preprocs(args, data_dir):
    is_processed = False
    preproc_dirs = glob.glob(data_dir + "*/")

    for preproc_dir in preproc_dirs:
        try:
            with open(preproc_dir + "settings.json") as f:
                settings = json.load(f)

            x_done = settings.pop("x_done") if "x_done" in settings else False
            y_done = settings.pop("y_done") if "y_done" in settings else False
        except:
            cprint("No settings.json under preproc dir", color="yellow")
            continue

        try:
            is_processed = np.all([v == args.preprocs[k] for k, v in settings.items()])
            if is_processed:
                cprint(f"Using preprocessing {preproc_dir}", color="cyan")
                break
        except:
            cprint("Preproc hyperparameter name mismatch", color="yellow")
            continue

    if not is_processed:
        preproc_dir = data_dir + str(len(preproc_dirs)) + "/"
        os.mkdir(preproc_dir)

        args.preprocs.update({"x_done": False, "y_done": False})

        with open(preproc_dir + "settings.json", 'w') as f:
            json.dump(args.preprocs, f)

    else:
        args.preprocs.update({"x_done": x_done, "y_done": y_done})

    return args, preproc_dir


def scaleAndClamp(X, clamp_lim, clamp):
    """ subject-wise scaling and clamping of EEG
        args:
            clamp_lim: float, abs limit (will be applied for min and max)
            clamp: bool, whether to clamp or not
        returns:
            X (size=subj, chan, time) scaled and clampted channel-wise, subject-wise
    """
    res = []
    for subjID in range(X.shape[0]):
        scaler = RobustScaler().fit(X[subjID, :, :].T)  # NOTE: must be samples x features
        _X = torch.from_numpy(scaler.transform(X[subjID, :, :].T)).to(torch.float)  # must be samples x features !!!
        if clamp:
            _X.clamp_(min=-clamp_lim, max=clamp_lim)
        res.append(_X.to(torch.float))
    return torch.stack(res).permute(0, 2, 1)  # NOTE: make (subj, ch, time) again


def baseline_correction(X, baseline_len_samp, preceding_chunk_for_baseline):
    """ subject-wise baselining
        args:
            baseline_len_samp: int, number of time steps to compute the baseline
            preceding_chunk_for_baseline: bool, whether to use the current or previous chunk
        returns:
            X (size=subj, chan, time) baseline-corrected channel-wise, subject-wise
    """

    with torch.no_grad():
        for subj_id in range(X.shape[0]):
            for chunk_id in range(X.shape[2]):
                if preceding_chunk_for_baseline:
                    if chunk_id == 0:
                        continue
                    baseline = X[subj_id, :, chunk_id - 1, -baseline_len_samp:].mean(axis=1)
                else:
                    baseline = X[subj_id, :, chunk_id, :baseline_len_samp].mean(axis=1)
                X[subj_id, :, chunk_id, :] -= baseline.view(-1, 1)
            cprint(f'subj_id: {subj_id} {X[subj_id].max().item()}', color='magenta')
    return X
