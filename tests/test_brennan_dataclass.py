import torch
from hydra import initialize, compose

from speech_decoding.dataclass.brennan2018 import Brennan2018Dataset

with initialize(version_base=None, config_path="../configs/"):
    args = compose(config_name="config.yaml")


def test_drop_last_segments():
    _X = torch.rand(2129, 33, 60, 360)
    _audio = torch.rand(2129, 1, 48000)
    onsets_path = "/home/sensho/speech_decoding/data/Brennan2018/AliceChapterOne-EEG.csv"

    X, audio, sentence_idxs = Brennan2018Dataset._drop_last_segments(
        _X, _audio, onsets_path
    )

    print(X.shape)

    assert X.shape[0] < _X.shape[0]
    assert X.shape[1:] == _X.shape[1:]
