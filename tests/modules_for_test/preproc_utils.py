import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from termcolor import cprint


def scale_and_clamp_single_test(X: np.ndarray, clamp_lim, clamp) -> torch.Tensor:
    """args:
    X: ( ch, time )
    """
    X = X.T

    X = RobustScaler().fit_transform(X)  # NOTE: must be samples x features
    X = torch.from_numpy(X).to(torch.float)

    if clamp:
        X.clamp_(min=-clamp_lim, max=clamp_lim)

    return X.T  # NOTE: make ( ch, time ) again


@torch.no_grad()
def baseline_correction_test(X, baseline_len_samp):
    """subject-wise baselining
    args:
        X (subject, chan, time) channel-wise, subject-wise
        baseline_len_samp: int, number of time steps to compute the baseline
    returns:
        X (size=subj, chan, time) baseline-corrected channel-wise, subject-wise
    """

    for subj_id in range(X.shape[0]):
        for chunk_id in range(X.shape[2]):
            baseline = X[subj_id, :, chunk_id, :baseline_len_samp].mean(axis=1)
            X[subj_id, :, chunk_id, :] -= baseline.view(-1, 1)
        cprint(
            f"subj_id: {subj_id} | max amlitude: {X[subj_id].max().item():.4f}",
            color="magenta",
        )
    return X


@torch.no_grad()
def baseline_correction_single_test(X: torch.Tensor, baseline_len_samp):
    """args:
        X: ( chunks, ch, time )
    returns:
        X ( chunks, ch, time ) baseline-corrected channel-wise
    """
    X = X.permute(1, 0, 2).clone()  # ( ch, chunks, time )

    for chunk_id in range(X.shape[1]):
        baseline = X[:, chunk_id, :baseline_len_samp].mean(axis=1)

        X[:, chunk_id, :] -= baseline.view(-1, 1)

    return X.permute(1, 0, 2)
