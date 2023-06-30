import torch
from transformers import Wav2Vec2Model
from speech_decoding.utils.wav2vec_util import get_last4layers_avg


def test_get_last4layers_avg():
    input = torch.rand(1, 48000)
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    emb = get_last4layers_avg(model, input)

    print(emb.shape)


if __name__ == "__main__":
    test_get_last4layers_avg()
