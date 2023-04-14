from speech_decoding.utils.preproc_utils import *
from tests.modules_for_test.preproc_utils import *

import pandas as pd
import matplotlib.pyplot as plt


def test_baseline_correction():
    input = torch.rand(2, 33, 60, 360)

    assert baseline_correction(input, 60).shape == input.shape

    # NOTE: test if the result matches the old implementation
    test_output = baseline_correction_single_test(input.flatten(end_dim=1), 60)
    test_output = test_output.reshape(input.shape)
    assert torch.equal(baseline_correction(input, 60), test_output)


def test_scale_and_clamp():
    input = torch.rand(2, 33, 60, 360)

    assert scale_and_clamp(input, 20).shape == input.shape
    # NOTE: check stochasticity of RobustScaler
    assert torch.equal(scale_and_clamp(input, 20), scale_and_clamp(input, 20))

    # NOTE: test if the result matches the old implementation
    # test_output = torch.stack(
    #     [scale_and_clamp_single_test(i, 20, True) for i in input.flatten(end_dim=1).numpy()]
    # ).reshape(input.shape)
    # assert torch.equal(scale_and_clamp(input, 20), test_output)


def test_continuous():
    onsets = pd.read_csv("data/Brennan2018/AliceChapterOne-EEG.csv").onset.to_numpy()

    onsets = continuous(onsets)

    assert (np.diff(onsets) >= 0).all()

    plt.plot(onsets)
    plt.savefig("data/Brennan2018/word-onsets-continuous.png")
